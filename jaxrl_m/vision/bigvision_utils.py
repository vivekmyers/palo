













"""Utils very specific to this project, not generic."""

import collections
import contextlib
import dataclasses
import functools
import io
import json
import multiprocessing
import multiprocessing.pool
import os
import re
import sys
import time
from typing import Mapping

from absl import flags
from absl import logging


import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections as mlc
import numpy as np









def pad_shard_unpad(wrapped, static_argnums=(0,), static_argnames=()):
    """Wraps a function with code that pads, shards, then un-shards, un-pads.

    Args:
      wrapped: the function to be wrapped. Signature is `params, *args, *kwargs`.
      static_argnums: indices of arguments to `wrapped` that should _not_ be
        padded and sharded, but instead be forwarded as-is. The default is (0,)
        because by far the most common use-case is to pass `params` first.
      static_argnames: names of kwargs to `wrapped` that should _not_ be padded
        and sharded, but instead be forwarded as-is.

    Returns:
      A new function that pads and shards its arguments before passing them to
      the wrapped function, and un-shards and un-pads the returned pytree.

      This is useful for calling a pmap'ed function with inputs that aren't
      divisible by the number of devices. A typical use is:
        @pad_shard_unpad
        @jax.pmap
        def forward(params, x): ...

    Notes:
      The padding is done in host-memory before being passed to the function, and
      the values returned by the function are transferred back to host memory.

      The returned function is augmented with a new keyword-only argument
      `min_device_batch` that, if specified, forces padding inputs to at least
      this size per device. This can be useful to avoid recompiles for the last
      batch and reduce memory fragmentation.
    """

    def pad_shard_unpad_wrapper(*args, min_device_batch=None, **kw):
        d = jax.local_device_count()  
        batch_sizes = {
            a.shape[0] for i, a in enumerate(args) if i not in static_argnums
        } | {v.shape[0] for k, v in kw.items() if k not in static_argnames}
        assert len(batch_sizes) == 1, f"Inconsistent batch-sizes: {batch_sizes}"
        b = batch_sizes.pop()

        def maybe_pad(x, actually_pad=True):
            if not actually_pad:
                return x  
            _, *shape = x.shape
            db, rest = divmod(b, d)
            if rest:
                x = np.concatenate([x, np.zeros((d - rest, *shape), x.dtype)], axis=0)
                db += 1
            if min_device_batch and db < min_device_batch:
                x = np.concatenate(
                    [x, np.zeros((d * (min_device_batch - db), *shape), x.dtype)]
                )
                db = min_device_batch
            return x.reshape(d, db, *shape)

        args = [maybe_pad(a, i not in static_argnums) for i, a in enumerate(args)]
        kw = {k: maybe_pad(v, k not in static_argnames) for k, v in kw.items()}
        out = wrapped(*args, **kw)

        def unpad(x):
            
            return einops.rearrange(jax.device_get(x), "d b ... -> (d b) ...")[:b]

        return jax.tree_map(unpad, out)

    return pad_shard_unpad_wrapper


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = labels[..., None] == jnp.arange(num_classes)[None]
    x = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def npload(fname):
    with open(fname, "rb") as f:
        data = f.read()
    return dict(np.load(io.BytesIO(data), allow_pickle=False))


def load_checkpoint(tree, npz):
    """Loads a jax pytree from a npz file.

    Args:
      tree: deprecated, use None.
        Bwd-compat for old format that only stored values: the pytree structure.
      npz: Either path to the checkpoint file (.npz), or a dict-like.

    Returns:
      A pytree that is the checkpoint.
    """
    if isinstance(npz, str):  
        npz = npload(npz)
    keys, values = zip(*list(npz.items()))
    if tree:
        checkpoint = tree.unflatten(values)
    else:
        checkpoint = recover_tree(keys, values)
    return checkpoint


def load_params(tree, npz):
    """Loads a parameters from a npz checkpoint.

    Args:
      tree: deprecated, use None.
        Bwd-compat for old format that only stored values: the pytree structure.
      npz: Either path to the checkpoint file (.npz), or a dict-like.

    Returns:
      A pytree that is the checkpoint.

    Notes:
      The filename can contain an indicator like `/path/to/file.npz:keyname`, in
      which case ["opt"]["params"]["keyname"] will become ["opt"]["params"] in
      the returned checkpoint. This allows ANY model that uses this function to
      load itself from a checkpoint that contains multiple sub-models, such as
      checkpoints generated from Argus or Distillation trainers.
    """
    key = None  
    if isinstance(npz, str):
        if (":" in npz and "://" not in npz) or (  
            "://" in npz and npz.count(":") == 2
        ):  
            npz, key = npz.rsplit(":", 1)
    checkpoint = load_checkpoint(tree, npz)
    if "params" in checkpoint:
        
        params = checkpoint["params"]
    elif "opt" in checkpoint:
        
        params = checkpoint["opt"]["target"]
    else:
        
        params = checkpoint
    if key is not None:
        params = tree_get(params, key)
    return params


def prefetch_scalar(it, nprefetch=1, devices=None):
    n_loc_dev = len(devices) if devices else jax.local_device_count()
    repl_iter = (np.ones(n_loc_dev) * i for i in it)
    return flax_utils.prefetch_to_device(repl_iter, nprefetch, devices)


def sigmoid_xent(*, logits, labels, reduction=True):
    
    
    
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    nll = -jnp.sum(labels * log_p + (1.0 - labels) * log_not_p, axis=-1)
    return jnp.mean(nll) if reduction else nll


def softmax_xent(*, logits, labels, reduction=True, kl=False, axis=-1):
    log_p = jax.nn.log_softmax(logits, axis=axis)
    nll = -jnp.sum(labels * log_p, axis=axis)
    if kl:
        nll += jnp.sum(labels * jnp.log(jnp.clip(labels, 1e-8)), axis=axis)
    return jnp.mean(nll) if reduction else nll


def weighted_softmax_xent(
    *, logits, labels, reduction=True, weights=None, label_smoothing=0.0, normalize=True
):
    """Compute weighted cross entropy.

    Args:
     logits: [batch, length, num_classes] float array.
     labels: categorical targets [batch, length] int array.
     reduction: reduce across batch dim.
     weights: None or array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.
     normalize: normalize each "sentence" loss by the number of tokens in it.

    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != labels.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(labels.shape))
        )
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    soft_targets = onehot(
        labels, vocab_size, on_value=confidence, off_value=low_confidence
    )

    loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)

    normalizing_factor = labels.shape[1]
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum(axis=1)

    loss = loss.sum(axis=1)
    if normalize:
        loss = loss / normalizing_factor

    return loss.mean() if reduction else loss


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    
    if accum_steps and accum_steps > 1:
        assert (
            images.shape[0] % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}"
        step_size = images.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            imgs = jax.lax.dynamic_slice(
                images, (i * step_size, 0, 0, 0), (step_size,) + images.shape[1:]
            )
            lbls = jax.lax.dynamic_slice(
                labels, (i * step_size, 0), (step_size, labels.shape[1])
            )
            li, gi = loss_and_grad_fn(params, imgs, lbls)
            l, g = l_and_g
            return (l + li, jax.tree_map(lambda x, y: x + y, g, gi))

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_map(lambda x: x / accum_steps, (l, g))
    else:
        return loss_and_grad_fn(params, images, labels)


def itstime(
    step,
    every_n_steps,
    total_steps,
    host=None,
    last=True,
    first=True,
    drop_close_to_last=0.25,
):
    """Returns True if it's time to execute an action.

    Args:
      step: the current step representing "now".
      every_n_steps: the action should run every this many steps.
      total_steps: the step number of the last step of training.
      host: host number. If provided, only run if we are this process.
      last: whether to run on the last step or not.
      first: whether to run on the first step or not.
      drop_close_to_last: if a step would run, but is this close (in terms of
        fraction of every_n_step) to the last one, skip.

    Returns:
      True if the action should be executed, False if not.
    """

    
    
    
    close_to_last = False
    if drop_close_to_last and every_n_steps:
        close_to_last = abs(step - total_steps) < drop_close_to_last * every_n_steps

    is_host = host is None or jax.process_index() == host
    is_step = every_n_steps and (step % every_n_steps == 0) and not close_to_last
    is_last = every_n_steps and step == total_steps
    is_first = every_n_steps and step == 1
    return is_host and (is_step or (last and is_last) or (first and is_first))


def checkpointing_timeout(writer, timeout):
    
    if writer is not None:
        try:
            writer.get(timeout=timeout)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                "Checkpoint writing seems to be a bottleneck. Make sure you do "
                "not do something wrong, like writing checkpoints to a distant "
                "cell. In a case you are OK with checkpoint writing being a "
                "bottleneck, you can configure `checkpoint_timeout` parameter"
            ) from e


def hms(s):
    """Format time in hours/minutes/seconds."""
    if s < 60:
        return f"{s:.0f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m:.0f}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{h:.0f}h{m:.0f}m"  


class Chrono:
    """Measures time and reports progress, hyper-specific to our train loops.

    Some concepts:
    1. This differentiates between three "types" of time:
      - training time: the time spent on actual training (fprop/bprop/update)
      - program time: overall time the program runs, including all overheads
      - pause time: the chronometer can be paused (eg during evals).
    2. This handles a "warmup": the first step is skipped for training time
        purposes, as it includes significant compilation overheads, which distort
        estimates.
    3. `accum`ulates (i.e. integrates) timings, and save/load them across
        restarts.
    """

    def __init__(self):
        self.program_start_time = time.time()
        self.train_start_time = None
        self.train_start_step = None  

        self.prev_time = None
        self.prev_step = None

        self.pause_start = None
        self.paused_time = 0

        self.warmup = 2  
        self.load()  
        self.note = "Chrono n/a"

    def inform(self, first_step, total_steps, global_bs, steps_per_epoch):
        """Provide some extra info that's only known later in the program."""
        self.prev_step = first_step
        self.first_step = first_step
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.global_bs = global_bs
        if total_steps:
            self.note = (
                f"Steps:{first_step}/{total_steps} [{first_step/total_steps:.1%}]"
            )

    def tick(self, step, measure, write_note):
        """A chronometer tick."""
        now = time.time()

        
        
        ds = step - self.prev_step  
        self.prev_step = step
        self.accum_examples_seen += ds * self.global_bs
        measure("examples_seen", self.accum_examples_seen)
        measure("epoch", step / self.steps_per_epoch)

        
        
        
        if self.warmup > 1:
            self.warmup -= 1
            write_note(self.note)  
            return
        if self.warmup == 1:
            self.train_start_time = self.prev_time = now
            self.train_start_step = step
            self.accum_program_time += now - self.program_start_time
            self.paused_time = 0  
            self.warmup = 0
            write_note(self.note)  
            return

        
        
        dt = now - self.prev_time - self.paused_time
        ncores = jax.device_count()  
        measure("img/sec/core", self.global_bs * ds / dt / ncores)

        
        self.accum_train_time += dt
        self.accum_pause_time += self.paused_time
        self.accum_program_time += dt + self.paused_time

        
        core_hours = self.accum_train_time * ncores / 60 / 60
        devtype = jax.devices()[0].device_kind
        measure(f"core_hours_{devtype}", core_hours)
        measure("core_hours", core_hours)  

        
        
        dt = now - self.train_start_time  
        steps_timed = step - self.train_start_step
        steps_todo = self.total_steps - step
        self.note = f"Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]"
        self.note += f"\nWalltime:{hms(self.accum_program_time)}"
        self.note += f" ({hms(self.accum_pause_time)} eval)"
        self.note += f"\nETA:{hms(dt / steps_timed * steps_todo)}"
        self.note += f"\nTotal train time:{hms(dt / steps_timed * self.total_steps)}"
        write_note(self.note)

        self.prev_time = now
        self.paused_time = 0

    def pause(self, wait_for=()):
        assert self.pause_start is None, "Don't pause twice."
        jax.block_until_ready(wait_for)
        self.pause_start = time.time()

    def resume(self):
        self.paused_time += time.time() - self.pause_start
        self.pause_start = None

    def save(self):
        return dict(
            accum_program_time=self.accum_program_time,
            accum_train_time=self.accum_train_time,
            accum_pause_time=self.accum_pause_time,
            accum_examples_seen=self.accum_examples_seen,
        )

    def load(self, ckpt={}):  
        self.accum_program_time = ckpt.get("accum_program_time", 0.0)
        self.accum_train_time = ckpt.get("accum_train_time", 0.0)
        self.accum_pause_time = ckpt.get("accum_pause_time", 0.0)
        self.accum_examples_seen = ckpt.get("accum_examples_seen", 0)


def _traverse_with_names(tree, with_inner_nodes=False):
    """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
    if dataclasses.is_dataclass(tree):
        tree = flax.serialization.to_state_dict(tree)
    
    
    
    if tree is None:
        return
    elif isinstance(tree, Mapping):
        keys = sorted(tree.keys())
        for key in keys:
            for path, v in _traverse_with_names(tree[key], with_inner_nodes):
                yield (key + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    elif isinstance(tree, (list, tuple)):
        for idx in range(len(tree)):
            for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
                yield (str(idx) + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    else:
        yield "", tree


def tree_flatten_with_names(tree):
    """Populates tree_flatten with leaf names.

    This function populates output of tree_flatten with leaf names, using a
    custom traversal that produces names is provided. The custom traversal does
    NOT have to traverse tree in the same order as jax, as we take care of
    automatically aligning jax' and custom traversals.

    Args:
      tree: python tree.

    Returns:
      A list of values with names: [(name, value), ...]
    """
    vals, tree_def = jax.tree_flatten(tree)

    
    
    tokens = range(len(vals))
    token_tree = tree_def.unflatten(tokens)
    val_names, perm = zip(*_traverse_with_names(token_tree))
    inv_perm = np.argsort(perm)

    
    assert len(val_names) == len(vals)

    return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(f, tree, *rest):
    """Like jax.tree_map but with a filter on the leaf path name.

    Args:
      f: A function with first parameter `name` (path-like "a/b/c") and remaining
        parameters values of `tree` and `*rest` corresponding to the given `name`
        Should return a new value for parameter `name`.
      tree: The tree of parameters `f` should be applied to.
      *rest: more trees of the exact same structure.

    Returns:
      A tree identical in structure to `tree` and `*rest` but with the leaves the
      result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
    """
    names_and_vals, tree_def = tree_flatten_with_names(tree)
    names, vals = zip(*names_and_vals)
    rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
    vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
    return tree_def.unflatten(vals)


def tree_map_with_regex(f, tree, regex_rules, not_f=lambda x: x, name=None):
    """Apply jax-style tree_map based on regex rules.

    Args:
      f: a function that is being applied to every variable.
      tree: jax tree of arrays.
      regex_rules: a list of tuples `(pattern, args)`, where `pattern` is a regex
        which used for variable matching and `args` are positional arguments
        passed to `f`. If some variable is not matched, we apply `not_f` transform
        which is id by default. If multiple patterns match, then only the first
        rule is applied.
      not_f: optional function which is applied to variables that do not match any
        pattern.
      name: a name of transform for logging purposes.

    Returns:
      a tree, transformed by `f` according to the given rules.
    """

    def _f(vname, v):
        for pattern, arg in regex_rules:
            if re.fullmatch(pattern, vname):
                if name and jax.process_index() == 0:
                    logging.info(
                        "Applying %s to %s with %s due to `%s`",
                        name,
                        vname,
                        arg,
                        pattern,
                    )
                return f(v, arg)
        return not_f(v)

    return tree_map_with_names(_f, tree)


def tree_get(tree, name):
    """Get an entry of pytree by flattened key name, eg a/b/c, with nice error.

    Args:
      tree: the pytree to be queried.
      name: the path to extract from the tree, see below for examples.

    Returns:
      A few examples:
        tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
        tree_get(tree, 'a') == 1
        tree_get(tree, 'b/c') == 2
        tree_get(tree, 'b') == {'c': 2, 'd': 3}
    """
    flattened = dict(_traverse_with_names(tree, with_inner_nodes=True))
    try:
        return flattened[name]
    except KeyError as e:

        class Msg(str):  
            def __repr__(self):
                return str(self)

        msg = "\n".join([name, "Available keys:", *flattened, ""])
        
        msg = mlc.ConfigDict(flattened)._generate_did_you_mean_message(
            name, msg
        )  
        raise KeyError(Msg(msg)) from e


def recover_dtype(a):
    """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
    if hasattr(a, "dtype") and a.dtype.type is np.void:
        assert a.itemsize == 2, "Unknown dtype!"
        return a.view(jax.numpy.bfloat16)
    else:
        return a





def save_checkpoint(checkpoint, path, step_copy=None, compressed=False):
    """Util for checkpointing: saves jax pytree objects to the disk.

    Args:
      checkpoint: arbitrary jax pytree to be saved.
      path: a path to save the checkpoint.
      step_copy: creates a copy of the checkpoint with `path-{step_copy}` name.
      compressed: whether to use np.savez or np.savez_compressed, useful if saving
        large buffers that are easily compressed (e.g. repeated or integers).
    """
    names_and_vals, _ = tree_flatten_with_names(checkpoint)
    io_buffer = io.BytesIO()

    if compressed:
        np.savez_compressed(io_buffer, **{k: v for k, v in names_and_vals})
    else:
        np.savez(io_buffer, **{k: v for k, v in names_and_vals})

    
    
    path_tmp = path + "-TEMPORARY"
    with gfile.GFile(path_tmp, "wb") as f:
        f.write(io_buffer.getvalue())
    gfile.rename(path_tmp, path, overwrite=True)

    if step_copy is not None:
        gfile.copy(path, f"{path}-{step_copy:09d}", overwrite=True)


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are saved by our programs
    without need to access the exact source code of the experiment. In particular,
    it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
    subtree of parameters.

    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.

    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def create_learning_rate_schedule(
    global_batch_size,
    total_steps,
    steps_per_epoch=None,
    base=0.0,
    decay_type="stair",
    scale_with_batchsize=False,
    warmup_steps=0,
    cooldown_steps=0,
    warmup_epochs=0,
    cooldown_epochs=0,
    **kw,
):
    """Creates learning rate schedule, see (internal link)

    Args:
      global_batch_size: The global batch-size optionally used for scaling.
      total_steps: The total number of steps to run.
      steps_per_epoch: How many steps form an epoch. Needed only if anything is
        passed in terms of epochs.
      base: The starting learning-rate (without warmup).
      decay_type: 'linear' or 'cosine', 'rsqrt', 'stair'.
      scale_with_batchsize: Whether or not to scale lr automatically.
      warmup_steps: how many steps to warm up for.
      cooldown_steps: how many steps to cool down for.
      warmup_epochs: how many epochs to warm up for.
      cooldown_epochs: how many epochs to cool down for.
      **kw: extra arguments specific to individual decay_types.

    Returns:
      A function learning_rate(step): float -> {"learning_rate": float}.
    """

    
    assert bool(warmup_epochs) + bool(warmup_steps) < 2, "Only one!"
    assert bool(cooldown_epochs) + bool(cooldown_steps) < 2, "Only one!"
    if warmup_epochs:
        warmup_steps = warmup_epochs * steps_per_epoch
    
    
    assert (total_steps <= 1) or (
        warmup_steps < total_steps
    ), "warmup_steps is >= total_steps"
    if cooldown_epochs:
        cooldown_steps = cooldown_epochs * steps_per_epoch

    def step_fn(step):
        """Step to learning rate function."""
        lr = base

        
        
        
        
        if scale_with_batchsize:
            lr = lr * global_batch_size / 256.0

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        if decay_type in ("linear", "polynomial"):
            power = kw.get("power", 1)
            zero = kw.get("end", kw.get("linear_end", 0))
            lr = zero + (lr - zero) * (1.0 - progress) ** power
        elif decay_type == "cosine":
            lr = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        elif decay_type == "rsqrt":
            timescale = kw.get("timescale", 10_000)
            shift = timescale - warmup_steps
            lr = jnp.where(
                warmup_steps < step, lr / jnp.sqrt((step + shift) / timescale), lr
            )
        elif decay_type == "stair":
            i = jnp.searchsorted(jnp.array(kw.get("steps", [])), step + 1)
            lr = lr * jnp.take(jnp.array([1.0] + list(kw.get("mults", []))), i)
        else:
            raise ValueError(f"Unknown lr type {decay_type}")

        if warmup_steps:
            lr = lr * jnp.minimum(1.0, step / warmup_steps)
        if cooldown_steps:
            lr = lr * jnp.minimum(1.0, (total_steps - step) / cooldown_steps)

        return jnp.asarray(lr, dtype=jnp.float32)

    return step_fn


def mixup(rng, *things, p=0.1, fold_in=None, n=2, **more_things):
    """Perform mixup https://arxiv.org/abs/1710.09412.

    Args:
      rng: The random key to use.
      *things: further arguments are the arrays to be mixed.
      p: the beta/dirichlet concentration parameter, typically 0.1 or 0.2.
      fold_in: One of None, "host", "device", or "sample". Whether to sample a
        global mixing coefficient, one per host, one per device, or one per
        example, respectively. The latter is usually a bad idea.
      n: with how many other images an image is mixed. Default mixup is n=2.
      **more_things: further kwargs are arrays to be mixed.  See also (internal link)
        for further experiments and investigations.

    Returns:
      A new rng key. A list of mixed *things. A dict of mixed **more_things.
    """
    rng, rng_m = jax.random.split(rng, 2)
    if fold_in == "host":
        rng_m = jax.random.fold_in(rng_m, jax.process_index())
    elif fold_in in ("device", "sample"):
        rng_m = jax.random.fold_in(rng_m, jax.lax.axis_index("batch"))
    ashape = (len(things[0]),) if fold_in == "sample" else (1,)
    alpha = jax.random.dirichlet(rng_m, jnp.array([p] * n), ashape)
    
    
    alpha = -jnp.sort(-alpha, axis=-1)

    def mix(batch):
        if batch is None:
            return None  

        def mul(a, b):  
            return b * jnp.expand_dims(a, tuple(range(1, b.ndim)))

        return sum(mul(alpha[:, i], jnp.roll(batch, i, axis=0)) for i in range(n))

    return rng, map(mix, things), {k: mix(v) for k, v in more_things.items()}


def sync_all_hosts():
    """Makes sure all hosts are synced."""
    if jax.process_count() > 1:
        x = jnp.ones([jax.local_device_count()])
        x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, "i"), "i")(x))
        assert x[0] == jax.device_count()


def check_and_compile_patterns(patterns):
    """Validates and compiles a list of param-patterns.

    The validation consists of checking for common mistakes, currently only that
    the pattern does not start with a slash, because unlike FLAX, our parameter
    names don't start with a slash.

    Args:
      patterns: a single (string) pattern (regex), or a list of patterns.

    Returns:
      A list of compiled and verified regexes.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    assert isinstance(patterns, (list, tuple)), patterns

    def check_and_compile(pattern):
        assert not pattern.startswith(
            "/"
        ), f"Big vision parameter names never start with '/': '{pattern}"
        return re.compile(pattern)

    return list(map(check_and_compile, patterns))


def make_mask_trees(tree, patterns, *, log=None):
    """Returns a boolean mask tree for every pattern (only first match)."""
    compiled_patterns = check_and_compile_patterns(patterns)

    def matchfirst(name, _):
        matches = []
        for pattern in compiled_patterns:
            matches.append(not any(matches) and bool(pattern.fullmatch(name)))
        if log is not None and True in matches and jax.process_index() == 0:
            logging.info(
                "%s: %s - matched by %s", log, name, patterns[matches.index(True)]
            )
        return np.array(matches)

    multimask = tree_map_with_names(matchfirst, tree)
    return [
        jax.tree_map(lambda matches, i=idx: matches[i], multimask)
        for idx in range(len(patterns))
    ]


@contextlib.contextmanager
def profile(name, ttl=3 * 365 * 24 * 3600):
    sess = startstop_prof_at_steps(None, name=name, ttl=ttl)
    yield
    startstop_prof_at_steps(sess, name=name, ttl=ttl)


def startstop_prof(sess, step=None, first_step=0, log_steps=1, surround=20, **kw):
    """Runs the profiler for `surround` steps around the next `log_steps`."""
    first_log = first_step + log_steps - (first_step % log_steps)
    
    start = max(first_log - surround // 2, first_step + 1)
    return startstop_prof_at_steps(sess, step, start, start + surround, **kw)


def startstop_prof_at_steps(
    sess,
    step=None,
    first_step=None,
    last_step=None,
    name="steps",
    ttl=3 * 365 * 24 * 3600,
):
    del sess, step, first_step, last_step, name, ttl
    pass  




class BigVisionMetricWriter:
    """A class for logging metrics."""

    def __init__(self, xid=-1, wid=-1, workdir=None):
        self.step_start(0)
        if jax.process_index() != 0:
            return  

        self.pool = multiprocessing.pool.ThreadPool(1)  
        self.fname = None
        if workdir:
            if xid != -1 and wid != -1:
                self.fname = os.path.join(
                    workdir, f"big_vision_{xid}_{wid}_metrics.txt"
                )
            else:
                self.fname = os.path.join(workdir, "big_vision_metrics.txt")

    def step_start(self, step):
        self.step = step
        self.step_metrics = {}

    def measure(self, name, value):
        """Logs the metric value."""
        if jax.process_index() != 0:
            return  

        
        
        value = np.array(value).squeeze()

        
        
        value = float(value) if value.ndim == 0 else value.shape

        logging.info(f"\u001b[35m[{self.step}]\u001b[0m {name} = {value}")
        logging.flush()
        self.step_metrics[name] = value

        return value  

    def step_end(self):
        """Ends a training step, write its full row."""
        if not self.step_metrics:
            return

        def write(metrics):
            with gfile.GFile(self.fname, "a") as f:
                f.write(json.dumps({"step": self.step, **metrics}) + "\n")

        if self.fname:
            self.pool.apply(lambda: None)  
            self.pool.apply_async(write, (self.step_metrics,))

    def close(self):
        self.step_end()
        if jax.process_index() == 0:
            self.pool.close()
            self.pool.join()


def maybe_cleanup_workdir(workdir, cleanup, info):
    """Potentially removes workdirs at end of run for cleanup."""
    if not workdir:
        return

    if not cleanup:
        info("Logs/checkpoints are in %s", workdir)
    elif jax.process_index() == 0:
        gfile.rmtree(workdir)
        try:  
            gfile.remove(os.path.join(workdir, ".."))
        except tf.errors.OpError:
            pass
