## Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation
[![arXiv](https://img.shields.io/badge/arXiv-2408.16228-df2a2a.svg)](https://arxiv.org/pdf/2408.16228)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10V2qIVl3IONsCbul3TtZYwxA_-I4oOAG#scrollTo=XzHIDBfKBbDv)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://palo-website.github.io/)

[Vivek Myers](https://people.eecs.berkeley.edu/~vmyers/), Bill Chunyuan Zheng, [Oier Mees](https://www.oiermees.com/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Kuan Fang](https://kuanfang.github.io/)
<hr style="border: 2px solid gray;"></hr>

This repository contains the code for Policy Adaptation via Language Optimization (PALO), which combines a handful of demonstrations of a task with proposed language decompositions sampled from a VLM to quickly enable rapid nonparametric adaptation, avoiding the need for a larger fine-tuning dataset.  
### Environment
```
conda create -n palo python=3.10
conda activate palo
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

## Using PALO

### Running Optimization

To get the best language decomposition from PALO, you can run the following commands:

```
python palo/optimize.py --instruction [Your Instruction Here] --trajectory_path [Your data here] \
 --checkpoint_path "./agent/checkpoint/" --im_size 224 --config_dir "./agent/config.pkl"
```

## Citation
PLease consider citing our work if you find it useful:
```bibtex
@inproceedings{myers2024policy,
  title={Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation},
  author={Vivek Myers and Bill Chunyuan Zheng and Oier Mees and Sergey Levine and Kuan Fang},
  booktitle={Conference on Robot Learning},
  year={2024}
}
```
