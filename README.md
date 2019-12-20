# Invertible Gaussian Reparameterization: Revisting the Gumbel-Softmax
**Andres Potapczynski**, **Gabriel Loaiza-Ganem** and **John P. Cunningham** 
[(webpage)](http://stat.columbia.edu/~cunningham/). <br>

* For inquiries: apotapczynski@gmail.com 

This repo contains a TensorFlow 2.0 implementation of the Invertible Gaussian Reparameterization.<br>

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
of the repo.

### Requirements
Below is a list of the main requirements. Installing them via `pip` will fetch the dependencies as well.
```
numba==0.46.0
plotly==4.2.1
scipy==1.3.1
seaborn==0.9.0
tensorflow==2.0.0
tensorflow-datasets==1.3.0
```

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
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model. |
| `model_type` | `<str> ('ExpGSDis')`  | The name of the model to use. Look at `./Models/train_vae.py`  for all the model options. |
| `temp` | `<float> (0.25)`  | The value of the temperature hyperparameter.|
| `sample_size`  | `<int> (1)`  | The number of samples that are taken from the noise distribution at each iteration. |
| `n_required` | `<int> ('10')`  | The number of categories needed for each discrete variable. |
| `num_of_discrete_param` | `<int> (1)`  | The number of parameters for the discrete variables. |
| `num_of_discrete_var` | `<int> (2)`  | The number of discrete variables in the model. |
| `num_of_norm_param` | `<int> (0)`  | The number of parameters for the continuous variables. If the value is 0 then no continuous variable is incorporated into the model. |
| `num_of_norm_var` | `<int> (0)`  | The number of continuous variables in the model. |
| `latent_norm_n` | `<int> (0)`  | The dimensionality of the continuous variables in the model. |
| `architecture` | `<str> ('dense')`  | The neural network architecture employed. All the options are in `./Models/VAENet.py`.|
| `learning_rate` | `<float> (0.001)`  | The learning rate used when training. |
| `batch_n` | `<int> (64)`  | The batch size taken per iteration. |
| `epochs` | `<int> (100)`  | The number of epochs used when training. |

### Understanding the structure of the repository

* `Models`: contains the training functions (`train_vae.py`) and the neural network architectures (`VAENet.py`).
* `Utils`: Besides general utils, it contains all the distributions (`Distributions.py`) and data loading routines (`load_data.py`)
* `vae_experiments`: Contains all the scripts to run the VAE experiments to replicate the paper results.
* `structure_output_prediction`: Contains all the scripts to run the SOP experiments.
* `Tests`: Contains various tests for relevant classes in the repo. The name indicates which class is being tested.


