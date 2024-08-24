import functools as ft
import flax.linen as nn
import jax.numpy as jnp

from functools import partial
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple
from jaxrl_m.common.common import MLP



import numpy as np
from jaxrl_m.common.common import MLP

ModuleDef = Any


class AddSpatialCoordinates(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        grid = jnp.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        )

        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        return jnp.concatenate([x, grid], axis=-1)


class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, features):
        if self.temperature == -1:
            from jax.nn import initializers

            temperature = self.param(
                "softmax_temperature", initializers.ones, (1), jnp.float32
            )
        else:
            temperature = 1.0

        
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        assert len(features.shape) == 4
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        features = features.transpose(0, 3, 1, 2).reshape(
            batch_size, num_featuremaps, self.height * self.width
        )

        softmax_attention = nn.softmax(features / temperature)
        expected_x = jnp.sum(
            self.pos_x * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(
            self.pos_y * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2 * num_featuremaps])

        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """
        features is B x H x W X C
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2)
        )
        features = jnp.reshape(features, [batch_size, -1])

        if no_batch_dim:
            features = features[0]

        return features


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = False
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    use_multiplicative_cond: bool = False
    num_spatial_blocks: int = 8
    task_units: int = 0
    mlp_kwargs: dict = None
    g_only: bool = False
    pre_fuse_mlp: bool = True
    return_ftmaps: bool = False
    fuse_ftmaps: bool = False
    task_highway: bool = False
    normalize_output: bool = False

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        task: jnp.ndarray = None,
        train: bool = True,
        cond_var=None,
    ):
        
        x = observations.astype(jnp.float32) / 127.5 - 1.0

        if task is not None and self.task_units == 0 and not self.fuse_ftmaps:
            task = task.astype(jnp.float32) / 127.5 - 1.0
            if self.g_only:
                x = task
            else:
                x = jnp.concatenate([x, task], axis=-1)

        

        if task is not None and self.task_units > 0 and not self.fuse_ftmaps:
            t = task.astype(jnp.float32)
            if self.task_highway:
                orig_t = t
            if self.pre_fuse_mlp:
                t = nn.Dense(self.task_units)(t)
                t = nn.relu(t)
                t = nn.Dense(self.task_units)(t)
                t = nn.relu(t)
            task = t

        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "batch":
            
            
            raise NotImplementedError()
            norm = partial(
                nn.BatchNorm,
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                dtype=self.dtype,
            )
        elif self.norm == "group":
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        
        
        
        
        
        
        
        
        
        
        elif self.norm == "layer":
            norm = partial(
                nn.LayerNorm,
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            raise ValueError("norm not found")

        act = getattr(nn, self.act)

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        if self.return_ftmaps:
            feature_maps = []

        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act if j < block_size - 1 else (lambda x: x),
                )(x)
                if self.use_multiplicative_cond:
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    cond_out = nn.Dense(
                        x.shape[-1], kernel_init=nn.initializers.xavier_normal()
                    )(cond_var)
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    x = x * x_mult

            if task is not None and self.task_units > 0 and not self.fuse_ftmaps:
                gamma = nn.Dense(x.shape[-1])(t)
                gamma = jnp.expand_dims(gamma, axis=(-2, -3))
                gamma = jnp.repeat(gamma, x.shape[-2], axis=-2)
                gamma = jnp.repeat(gamma, x.shape[-3], axis=-3)

                beta = nn.Dense(x.shape[-1])(t)
                beta = jnp.expand_dims(beta, axis=(-2, -3))
                beta = jnp.repeat(beta, x.shape[-2], axis=-2)
                beta = jnp.repeat(beta, x.shape[-3], axis=-3)

                x = act(gamma * x + beta)

            if self.fuse_ftmaps:
                
                curr_layer_ftmap = task[i]
                x = jnp.concatenate([x, curr_layer_ftmap], axis=-1)

            if self.return_ftmaps:
                feature_maps.append(x)

        if self.return_ftmaps:
            return feature_maps

        if self.pooling_method == "spatial_learned_embeddings":
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
        elif self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(
                height, width, channel, pos_x, pos_y, self.softmax_temperature
            )(x)
        elif self.pooling_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pooling_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError("pooling method not found")

        if self.mlp_kwargs is not None:
            x = MLP(**self.mlp_kwargs)(x)

        
        if self.task_highway:
            x = x + orig_t

        if self.normalize_output:
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
         
        return x


resnetv1_configs = {
    "resnetv1-18": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock
    ),
    "resnetv1-34": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock
    ),
    "resnetv1-50": ft.partial(
        ResNetEncoder, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
    ),
    "resnetv1-18-deeper": ft.partial(
        ResNetEncoder, stage_sizes=(3, 3, 3, 3), block_cls=ResNetBlock
    ),
    "resnetv1-18-deepest": ft.partial(
        ResNetEncoder, stage_sizes=(4, 4, 4, 4), block_cls=ResNetBlock
    ),
    "resnetv1-18-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(2, 2, 2, 2),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-18-bridge-task": ft.partial(
        ResNetEncoder,
        stage_sizes=(2, 2, 2, 2),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
        task_units=256,
        
    ),
    "resnetv1-34-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-34-bridge-task": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
        task_units=256,
        
    ),
    "resnetv1-34-bridge-task-x-units": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-50-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
        mlp_kwargs=dict(
            hidden_dims=(512, ),
        ),
    ),
}
