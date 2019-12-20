# Invertible Gaussian Reparameterization: Revisting the Gumbel-Softmax
**Andres Potapczynski**, **Gabriel Loaiza-Ganem** and **John P. Cunningham** 
[(webpage)](http://stat.columbia.edu/~cunningham/). <br>

* For inquiries: apotapczynski@gmail.com 

This repo contains a TensorFlow 2.0 implementation of the Invertible Gaussian Reparameterization.

**Abstract**<br>
*The Gumbel-Softmax is a continuous distribution over the simplex that is often used as a relaxation
 of discrete distributions. Because it can be readily interpreted and easily reparameterized, the
 Gumbel-Softmax enjoys widespread use. We show that this relaxation experiences two shortcomings
 that affect its performance, namely: numerical instability caused by its temperature hyperparameter
 and noisy KL estimates. The first requires the temperature values to be set too high,
 creating a poor correspondence between continuous components and their respective discrete
 complements. The second, which is of fundamental importance to variational autoencoders, severely
 hurts performance. We propose a flexible and reparameterizable family of distributions that
 circumvents these issues by transforming Gaussian noise into one-hot approximations through an
 invertible function. Our construction improves numerical stability, and outperforms the
 Gumbel-Softmax in a variety of experiments while generating samples that are closer to their
 discrete counterparts and achieving lower-variance gradients. Furthermore, with a careful choice of the
 invertible function we extend the reparameterization trick to distributions with countably infinite
 support.*

## Overview

The goal of this documentation is to provide a guide to replicate the results from the paper and to clarify the structure
of the repository.

### Replicating the results

The scripts to replicate the experiments of the paper are under the `vae_experiments` folder. For example, to replicate the
results for the binarized MNIST discrete model experiment, in the console you can run (assuming that your folder location
is on the root of the repo)
```
python vae_experiments/mnist_vae.py
```
This experiment is run by a function named `run_vae` this function takes as arguments two parameters. (1) is a
dictionary that contains all the hyperparameter specifications of the model `hyper` and (2) a flag to test
that the code is running properly `run_with_sample` (set to False in order to run the experiment with all the data).
The contents of that the hyperparameter dictionary expects are detailed below:

| Key                               | Value (Example)                            | Description     |
| :-------------------------------- | :------------------------------: | :-----------------------: |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `model_type` | `<str> ('ExpGSDis')`  | The name of the model to use. |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `sample_size`  | `<int> (1)`  | The number of samples that are taken from the noise distribution at each iteration |
| `n_required` | `<int> ('10')`  | The number of categories needed for each discrete variable |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model |

### Understanding the structure of the repository
