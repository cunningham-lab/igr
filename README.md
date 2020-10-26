# [Invertible Gaussian Reparameterization: Revisting the Gumbel-Softmax](https://arxiv.org/abs/1912.09588)
**Andres Potapczynski**, **Gabriel Loaiza-Ganem** and **John P. Cunningham**
[(webpage)](http://stat.columbia.edu/~cunningham/). <br>

* For inquiries: apotapczynski@gmail.com

This repo contains a TensorFlow 2.0 implementation of the Invertible Gaussian Reparameterization.

<br>**Abstract**<br>
*The Gumbel-Softmax is a continuous distribution over the simplex that is often used as a
relaxation of discrete distributions. Because it can be readily interpreted and easily
reparameterized, it enjoys widespread use. We propose a modular and more flexible family
of reparameterizable distributions where Gaussian noise is transformed into a one-hot
approximation through an invertible function. This invertible function is composed of a
modified softmax and can incorporate diverse transformations that serve different
specific purposes. For example, the stick-breaking procedure allows us to extend the
reparameterization trick to distributions with countably infinite support, thus enabling
the use of our distribution along nonparametric models, or normalizing flows let us
increase the flexibility of the distribution. Our construction enjoys theoretical
advantages over the Gumbel-Softmax, such as closed form KL, and significantly
outperforms it in a variety of experiments.*

## Overview
The goal of this documentation is to clarify the structure of the repo and to provide a guide to replicate the
results from the paper. To avoid reading all details, I recommend that you find the files that are linked to
the experiment that you want to run (see Replicating Figures / Tables section) and only then check the
information about the folder of interest in the (General Information section).

### Requirements
Briefly, the requirements for the project are Python >= 3.6 and to pip install the the packages
needed (which will fetch all the dependencies as well). Make sure that you have the latest `pip`
version.  This repo was develop using TensorFlow 2.0.1, but it runs for 2.1.0 as well. The only
package that requires a specific version is TensorFlow Probability.  Moreover, we also added a
Singularity definition file `igr_singularity.def` if you want to create an image to run the project
in a HPC cluster (it will only require that the host has the 10.0 CUDA drivers available). In
summary you could run the following in you terminal to ensure that the installation works (after
adding the repo to your
`$PYTHONPATH`
`export PYTHONPATH=$PYTHONPATH:"path/to/igr"`)
```
python3 -m pip install --update pip
python3 -m pip install -r requirements.txt
cd ./igr/
mkdir Log
python3 vae_experiments/experiments_grid.py
```
from wherever you cloned the repo. It should successfully run an experiment with an small sample that should
take 5 - 10 seconds.

## General Information

### Structure of the Repository

* `Log`: is the directory where the outputs from the experiments are logged and saved (weights of the NNs).
  The contents of this directory are disposable, move the results that you want to save into the
  `Results` directory for further analysis (see below).
* `Models`: contains the training functions (`train_vae.py`, `SOPOptimizer,py`), the optimizer classes
(`OptVAE`, `SOPOptimizer`), and the neural network architectures (`VAENet.py` `SOP.py`)
for both the VAE and for the SOP experiments.
* `Results`: this directory serves two purposes. First, it holds all the figures created by the scripts in the
  repo and then it contains the outputs from the experiments that are used as input for some figures /
  tables.
* `Tests`: contains various tests for relevant classes in the repo. The name indicates which class is being tested.
* `Utils`: has key functions / classes used throughout the repo: `Distributions.py` contains the GS and
  IGR samplers, `MinimizeEmpiricalLoss.py` contains the optimizer class for learning the parameters that
  approximate a discrete distribution, `general.py` contains the functions used for approximating discrete
  distributions, for evaluating simplex proximity and for initializing the distribution's parameters,
  `load_data.py` contains all the utils needed to load the data, and `viz_vae.py` contains the functions to
  plot the performance over time.
* `approximations`: contains all the scripts to approximate discrete distributions and to learn the IGR
  priors (more details in the Replicating section).
* `vae_experiments`: contains all the scripts to run the VAE experiments to
replicate the paper results (more details in the Replicating section).
* `structure_output_prediction`: contains all the scripts to run the SOP experiments (more details in the Replicating section).

### Variables Explanation
All the scripts and saving references assume that you are running the files from the root folder of the repo
(as in the first test to check that the installation was correct). In other words, from the terminal run the
scripts as
```
python3 path/to/script.py
```
or set your IDE to run files from the root location. Throughout the code, the discrete tensors have the
following shape `(batch_n, categories_n, sample_size, num_of_vars)`. Where `batch_n` is the batch size used at
each iteration (selected by the user), `categories_n` is the dimension of each of the discrete variables,
`sample_size` is the number of samples taken during the reparameterization trick and `num_of_vars` the number
of discrete variables that the model has. Finally, below is a list that expands on the name of variables and
hyperparameters that appear frequently in different experiment scripts.

| Name                               | Type and Value (Example)                            | Description     |
| :-------------------------------- | :------------------------------: | :-----------------------: |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model. |
| `model_type` | `<str> ('ExpGSDis')`  | The name of the model to use. Look at `./Models/train_vae.py`  for all the model options. |
| `temp` | `<float> (0.25)`  | The value of the temperature hyperparameter.|
| `sample_size`  | `<int> (1)`  | The number of samples that are taken from the noise distribution at each iteration. |
| `n_required` | `<int> ('10')`  | The number of categories needed for each discrete variable (identical to `categories_n`. |
| `num_of_discrete_param` | `<int> (1)`  | The number of parameters for the discrete variables. Alpha for GS
mu and sigma for IGR|
| `num_of_discrete_var` | `<int> (2)`  | The number of discrete variables in the model. |
| `num_of_norm_param` | `<int> (0)`  | The number of parameters for the continuous variables. |
| `num_of_norm_var` | `<int> (0)`  | The number of continuous variables in the model. |
| `latent_norm_n` | `<int> (0)`  | The dimensionality of the continuous variables in the model. |
| `architecture` | `<str> ('dense')`  | The neural network architecture employed. All the options are in `./Models/VAENet.py`.|
| `learning_rate` | `<float> (0.001)`  | The learning rate used when training. |
| `batch_n` | `<int> (64)`  | The batch size taken per iteration. |
| `epochs` | `<int> (100)`  | The number of epochs used when training. |
| `run_jv` | `<bool> (False)`  | Whether to run the JointVAE model or not. |
| `gamma` | `<int> (100)`  | The penalty coefficient in the JointVAE loss. |
| `cont_c_linspace` | `<tuple> ((0., 5., 25000))`  | The lower bound, upper bound, and how many iters to get from lower to upper. |
| `disc_c_linspace` | `<tuple> ((0., 5., 25000))`  | The lower bound, upper bound, and how many iters to get from lower to upper. |
| `check_every` | `<int> (1)`  | How often (in terms of epochs) the test loss is evaluated. |
| `run_with_sample` | `<bool> (True)`  | Test the experiment with a small sample. |
| `num_of_repetitions` | `<int> (1)`  | Determine how many times to run the experiment (useful for doing CV). |
| `truncation_options` | `<str> ('quantile')`  | The statistical procedure to determine the categories needed in a given batch. |
| `threshold` | `<float> (0.99)`  | The precision parameter for the IGR-SB. |
| `prior_file` | `<str> ('./Results/mu_xi_unif_10_IGR_SB_Finite.pkl')`  | Location of the prior parameters file. |
| `run_closed_form_kl` | `<bool> (True)`  | Whether to use the Gaussian close form KL (only available to the IGR). |
| `width_height` | `<tuple> ((14, 28, 1))`  | The size of the images for the SOP experiment. |
| `iter_per_epoch` | `<int> (937)`  | The number of iterations per epoch in SOP (the VAE experiments infer this). |


### Implementation Nuances

The IGR-SB extends the reparameterization trick to distributions with an countably infinite
support. However, to implement the IGR-SB with finite resources, there is a threshold imposed on the
stick breaking procedure that determines a finite number of categories that are needed for a proper
estimation. Additionally, the number of categories has a maximum that is set as a
reference (this is called `latent_discrete_n` in the `hyper` dictionary). This maximum can be moved
accordingly to fit the problem. For example, for CelebA we set a maximum number of categories to 50
but the thresholding procedure ended up selecting 20-30. Moving the maximum beyond 50 would have
resulted in waste of memory allocated but would have not yield any quantitative difference. However,
setting the maximum to 10 would have truncated the stick-breaking procedure too soon and would have
resulted in a loss of performance. To avoid this situation, we recommend monitoring if the threshold
is met. If not, then increasing the maximum would be needed.
