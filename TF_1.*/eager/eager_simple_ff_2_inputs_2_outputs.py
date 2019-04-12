# eager_simple_feed_forward.py
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable Eager.
tfe.enable_eager_execution()


# ===== Global Variables =====

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000             # 1000 examples
training_inputs_x = tf.random_normal([NUM_EXAMPLES])     # The Tensor of 1000 examples
training_inputs_y = tf.random_normal([NUM_EXAMPLES])     # The Tensor of 1000 examples
noise_a = tf.random_normal([NUM_EXAMPLES])                 # Noise of 1000 examples.
noise_b = tf.random_normal([NUM_EXAMPLES])                 # Noise of 1000 examples.
weight_xa = 1
weight_ya = 5
weight_xb = 1
weight_yb = 9
bias_a = 1
bias_b = 1

# ===== Prediction, Loss and Gradient =====

# Prediction Function using input, weight and bias.
def prediction(input_x, input_y, weight_x, weight_y, bias, noise=0):
  return input_x * weight_x + input_y * weight_y + bias + noise

training_outputs_a = prediction(training_inputs_x, weight_xa, training_inputs_y, weight_ya, bias_a, noise_a)     # Output of 1000 examples.
training_outputs_b = prediction(training_inputs_x, weight_xb, training_inputs_y, weight_yb, bias_b, noise_b)     # Output of 1000 examples.

# A loss function using mean-squared error (MSE)
def loss(weights_x, weights_y, biases, training_outputs):
  error = prediction(training_inputs_x, training_inputs_y, weights_x, weights_y, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))  # MSE (Y0 - Y1)^2

# Return the derivative of loss with respect to weight and bias (gradient)
def grad(weights_x, weights_y, biases, training_outputs):
  with tf.GradientTape() as tape:           # records for automatic differentiation.
    loss_value = loss(weights_x, weights_y, biases, training_outputs)
  return tape.gradient(loss_value, [weights_x, weights_y, biases])



# ===== main = train =====
def main():

    # ===== Training Variables =====

    train_steps = 500
    learning_rate = 0.02

    # Start with arbitrary values for W and B on the same batch of data
    Wx_a = tfe.Variable(8.)
    Wy_a = tfe.Variable(2.)
    B_a = tfe.Variable(10.)
    Wx_b = tfe.Variable(5.)
    Wy_b = tfe.Variable(1.)
    B_b = tfe.Variable(1.)

    # ===== Train =====

    print("Initial A loss: {:.3f}".format(loss(Wx_a, Wy_a, B_a, training_outputs_a)))
    print("Initial B loss: {:.3f}".format(loss(Wx_b, Wy_b, B_b, training_outputs_b)))

    for i in range(train_steps):
      dWx_a, dWy_a, dB_a = grad(Wx_a, Wy_a, B_a, training_outputs_a)
      Wx_a.assign_sub(dWx_a * learning_rate)
      Wy_a.assign_sub(dWy_a * learning_rate)
      B_a.assign_sub(dB_a * learning_rate)
      if i % 20 == 0:
        print("A Loss at step {:03d}: {:.3f}".format(i, loss(Wx_a, Wy_a, B_a, training_outputs_a)))

      dWx_b, dWy_b, dB_b = grad(Wx_b, Wy_b, B_b, training_outputs_b)
      Wx_b.assign_sub(dWx_b * learning_rate)
      Wy_b.assign_sub(dWy_b * learning_rate)
      B_b.assign_sub(dB_b * learning_rate)
      if i % 20 == 0:
        print("B Loss at step {:03d}: {:.3f}".format(i, loss(Wx_b, Wy_b, B_b, training_outputs_b)))

    print("Final A loss: {:.3f}".format(loss(Wx_a, Wy_a, B_a, training_outputs_a)))
    print("Wx_a = {}, Wy_a = {}, B_a = {}".format(Wx_a.numpy(), Wy_a.numpy(), B_a.numpy()))

    print("Final B loss: {:.3f}".format(loss(Wx_b, Wy_b, B_b, training_outputs_b)))
    print("Wx_b = {}, W_b = {}, B_b = {}".format(Wx_b.numpy(), Wy_b.numpy(), B_b.numpy()))

    print("Predict A: 2 and 3")
    print(prediction(2, 3, Wx_a, Wy_a, B_a))
    print(prediction(2, 3, weight_xa, weight_ya, bias_a))

    print("Predict B: 2 and 3")
    print(prediction(2, 3, Wx_b, Wy_b, B_b))
    print(prediction(2, 3, weight_xb, weight_yb, bias_b))



if __name__ == "__main__":
    main()
