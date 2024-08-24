
"""Utilities shared across models."""

from absl import logging
import jaxrl_m.vision.bigvision_utils as u
import flax.linen as nn
import jax
import jax.numpy as jnp

def merge_params(loaded, inited, dont_load=()):
    """Makes `loaded` pytree match `init`, warning or failing on mismatch.

    Args:
      loaded: pytree of parameters, typically loaded from a checkpoint.
      inited: pytree of parameter, typically coming from model init.
      dont_load: List of regexes for parameters which shall not be taken
        from `loaded`, either because they should remain at their init value,
        or because they are missing on either side.

    Returns:
      If successful, a new pytree which matches the structure of `init`
      but contains values from `loaded`, except for `dont_load`.

      If structures don't match and mismatches are not covered by regexes in
      `dont_load` argument, then raises an exception with more information.
    """
    dont_load = u.check_and_compile_patterns(dont_load)

    def should_merge(name):
        return not any(pattern.fullmatch(name) for pattern in dont_load)

    loaded_flat, _ = u.tree_flatten_with_names(loaded)
    inited_flat, _ = u.tree_flatten_with_names(inited)
    loaded_flat = {k: v for k, v in loaded_flat}
    inited_flat = {k: v for k, v in inited_flat}

    
    merged = {}
    for name, init_val in inited_flat.items():
        
        if name in loaded_flat and should_merge(name):
            merged[name] = loaded_flat[name]
        else:
            logging.info("Ignoring checkpoint and using init value for %s", name)
            merged[name] = init_val

    def pp(title, names, indent="  "):  
        if names:
            return f"{title}:\n" + "\n".join(f"{indent}{k}" for k in sorted(names))
        else:
            return ""

    
    not_in_loaded = inited_flat.keys() - loaded_flat.keys()
    not_in_inited = loaded_flat.keys() - inited_flat.keys()
    logging.info(pp("Parameters in model but not in checkpoint", not_in_loaded))
    logging.info(pp("Parameters in checkpoint but not in model", not_in_inited))

    
    not_in_loaded = {k for k in not_in_loaded if should_merge(k)}
    not_in_inited = {k for k in not_in_inited if should_merge(k)}

    if not_in_loaded or not_in_inited:
        raise ValueError(
            pp("Params in checkpoint", loaded_flat.keys())
            + "\n"
            + pp("Params in model (code)", inited_flat.keys())
            + "\n"
            + pp(
                "Params in model (code) but not in checkpoint and not `dont_load`ed",
                not_in_loaded,
                indent=" - ",
            )
            + "\n"
            + pp(  
                "Params in checkpoint but not in model (code) and not `dont_load`ed",
                not_in_inited,
                indent=" + ",
            )
        )  

    return u.recover_tree(merged.keys(), merged.values())
