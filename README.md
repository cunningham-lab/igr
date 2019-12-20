# Invertible Gaussian Reparameterization: Revisting the Gumbel-Softmax
TensorFlow 2.0 implementation of the Invertible Gaussian Reparameterization.

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
