import tensorflow as tf
from rebar_toy import RELAX
# from rebar_toy import toy_loss
from rebar_toy import toy_loss_2

lr = 1 * 1.e-2
batch_n, categories_n, sample_size, num_of_vars = 1, 1, 1, 3
shape = (batch_n, categories_n, sample_size, num_of_vars)
# max_iter = 1 * int(1.e4)
# check_every = int(1.e3)
max_iter = 3 * int(1.e3)
check_every = int(1.e2)

# relax = RELAX(toy_loss, lr, shape)
relax = RELAX(toy_loss_2, lr, shape)

for idx in range(max_iter):
    grads, loss = relax.compute_rebar_gradients_and_loss()
    relax.apply_gradients(grads)
    theta = tf.math.sigmoid(relax.log_alpha)
    eta = relax.eta.numpy()
    temp = tf.math.exp(relax.log_temp).numpy()
    if idx % check_every == 0:
        text = f'Loss {loss.numpy(): 1.2e} || '
        for i in range(theta.shape[3]):
            text += f't_{i} {theta.numpy()[0, 0, 0, i]: 1.2e} || '
        text += f'tau {temp:1.2e} || eta {eta:1.2e} || '
        text += f'i {idx: 4d}'
        print(text)
