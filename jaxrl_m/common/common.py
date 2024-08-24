from typing import Any, Dict, Tuple, Union
import flax
from flax.core import FrozenDict
import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import optax
import functools

from jaxrl_m.common.typing import Callable, PRNGKey
from jaxrl_m.common.typing import Optional
from jaxrl_m.common.typing import Params
from jaxrl_m.common.typing import Sequence


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def orthogonal_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def final_layer_init(init_w: float = 1e-3):
    return nn.initializers.uniform(-init_w, init_w)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    activate_final: bool = False
    small_init_final: bool = False
    dropout: float = 0.0

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        layers = [
            nn.Dense(size, kernel_init=default_init()) for size in self.hidden_dims[:-1]
        ]
        if self.small_init_final:
            layers.append(
                nn.Dense(
                    self.hidden_dims[-1],
                    kernel_init=final_layer_init(),
                    bias_init=nn.initializers.constant(0),
                )
            )
        else:
            layers.append(nn.Dense(self.hidden_dims[-1], kernel_init=default_init()))
        self.layers = layers

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = self.layers[i](x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation(x)
                x = self.dropout_layer(x, deterministic=not train)
        return x


class TaskMLP(MLP):
    def __call__(self, observations: jnp.ndarray, task: jnp.ndarray, train: bool=False):
        return super().__call__(task, train)

class JaxRLTrainState(struct.PyTreeNode):
    """
    Custom TrainState class to replace `flax.training.train_state.TrainState`.

    Adds support for holding target params and updating them via polyak
    averaging. Adds the ability to hold an rng key for dropout.

    Also generalizes the TrainState to support an arbitrary pytree of
    optimizers, `txs`. When `apply_gradients()` is called, the `grads` argument
    must have `txs` as a prefix. This is backwards-compatible, meaning `txs` can
    be a single optimizer and `grads` can be a single tree with the same
    structure as `self.params`.

    Also adds a convenience method `apply_loss_fns` that takes a pytree of loss
    functions with the same structure as `txs`, computes gradients, and applies
    them using `apply_gradients`.

    Attributes:
        step: The current training step.
        apply_fn: The function used to apply the model.
        params: The model parameters.
        target_params: The target model parameters.
        txs: The optimizer or pytree of optimizers.
        opt_states: The optimizer state or pytree of optimizer states.
        rng: The internal rng state.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Params
    target_params: Params
    txs: Any = struct.field(pytree_node=False)
    opt_states: Any
    rng: PRNGKey

    @staticmethod
    def _tx_tree_map(*args, **kwargs):
        return jax.tree_map(
            *args,
            is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
            **kwargs,
        )

    def target_update(self, tau: float) -> "JaxRLTrainState":
        """
        Performs an update of the target params via polyak averaging. The new
        target params are given by:

            new_target_params = tau * params + (1 - tau) * target_params
        """
        new_target_params = jax.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), self.params, self.target_params
        )
        return self.replace(target_params=new_target_params)

    def apply_gradients(self, *, grads: Any) -> "JaxRLTrainState":
        """
        Only difference from flax's TrainState is that `grads` must have
        `self.txs` as a tree prefix (i.e. where `self.txs` has a leaf, `grads`
        has a subtree with the same structure as `self.params`.)
        """
        updates_and_new_states = self._tx_tree_map(
            lambda tx, opt_state, grad: tx.update(grad, opt_state, self.params),
            self.txs,
            self.opt_states,
            grads,
        )
        updates = self._tx_tree_map(lambda _, x: x[0], self.txs, updates_and_new_states)
        new_opt_states = self._tx_tree_map(
            lambda _, x: x[1], self.txs, updates_and_new_states
        )

        
        
        updates_flat = []
        self._tx_tree_map(
            lambda _, update: updates_flat.append(update), self.txs, updates
        )

        
        updates_acc = jax.tree_map(
            lambda *xs: jnp.sum(jnp.array(xs), axis=0), *updates_flat
        )
        new_params = optax.apply_updates(self.params, updates_acc)

        return self.replace(
            step=self.step + 1, params=new_params, opt_states=new_opt_states
        )

    def apply_loss_fns(
        self, loss_fns: Any, pmap_axis: str = None, has_aux: bool = False
    ) -> Union["JaxRLTrainState", Tuple["JaxRLTrainState", Any]]:
        """
        Convenience method to compute gradients based on `self.params` and apply
        them using `apply_gradients`. `loss_fns` must have the same structure as
        `txs`, and each leaf must be a function that takes two arguments:
        `params` and `rng`.
        
        This method automatically provides fresh rng to each loss function and
        updates this train state's internal rng key.

        Args:
            loss_fns: loss function or pytree of loss functions with same
                structure as `self.txs`. Each loss function must take `params`
                as the first argument and `rng` as the second argument, and return
                a scalar value.
            pmap_axis: if not None, gradients (and optionally auxiliary values)
                will be averaged over this axis
            has_aux: if True, each `loss_fn` returns a tuple of (loss, aux) where
                `aux` is a pytree of auxiliary values to be returned by this
                method.

        Returns:
            If `has_aux` is True, returns a tuple of (new_train_state, aux).
            Otherwise, returns the new train state.
        """
        
        treedef = jax.tree_util.tree_structure(loss_fns)
        new_rng, *rngs = jax.random.split(self.rng, treedef.num_leaves + 1)
        rngs = jax.tree_util.tree_unflatten(treedef, rngs)

        
        grads_and_aux = jax.tree_map(
            lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(self.params, rng),
            loss_fns,
            rngs,
        )

        
        self = self.replace(rng=new_rng)

        
        if pmap_axis is not None:
            grads_and_aux = jax.lax.pmean(grads_and_aux, axis_name=pmap_axis)

        if has_aux:
            grads = jax.tree_map(lambda _, x: x[0], loss_fns, grads_and_aux)
            aux = jax.tree_map(lambda _, x: x[1], loss_fns, grads_and_aux)
            return self.apply_gradients(grads=grads), aux
        else:
            return self.apply_gradients(grads=grads_and_aux)

    @classmethod
    def create(
        cls, *, apply_fn, params, txs, target_params=None, rng=jax.random.PRNGKey(0)
    ):
        """
        Initializes a new train state.

        Args:
            apply_fn: The function used to apply the model, typically `model_def.apply`.
            params: The model parameters, typically from `model_def.init`.
            txs: The optimizer or pytree of optimizers.
            target_params: The target model parameters.
            rng: The rng key used to initialize the rng chain for `apply_loss_fns`.
        """
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            txs=txs,
            opt_states=cls._tx_tree_map(lambda tx: tx.init(params), txs),
            rng=rng,
        )
