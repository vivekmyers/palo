## Policy Adaptation via Language Optimization

Code for the paper [Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation](https://arxiv.org/abs/2408.16228).

### Environment
```
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
pip install -r requirements.txt
```
For GPU:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 
