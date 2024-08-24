from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import jax.numpy as jnp
import flax
import tensorflow as tf


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray, tf.Tensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]

ModuleMethod = Union[str, Callable, None]
