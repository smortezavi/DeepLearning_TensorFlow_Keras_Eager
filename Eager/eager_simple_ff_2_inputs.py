# eager_simple_feed_forward.py
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable Eager.
tfe.enable_eager_execution()


# ===== Global Variables =====

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000             # 1000 examples
training_inputs_x = tf.random_normal([NUM_EXAMPLES])     # The Tensor of 1000 examples
training_inputs_y = tf.random_normal([NUM_EXAMPLES])     # The Tensor of 1000 examples
noise = tf.random_normal([NUM_EXAMPLES])                 # Noise of 1000 examples.
weight_x = 4
weight_y = 7
bias = 9
training_outputs_z = training_inputs_x * weight_x + training_inputs_y * weight_y + bias + noise     # Output of 1000 examples.


# ===== Prediction, Loss and Gradient =====

# Prediction Function using input, weight and bias.
def prediction(input_x, input_y, weight_x, weight_y, bias):
  return input_x * weight_x + input_y * weight_y + bias

# A loss function using mean-squared error (MSE)
def loss(weights_x, weights_y, biases):
  error = prediction(training_inputs_x, training_inputs_y, weights_x, weights_y, biases) - training_outputs_z
  return tf.reduce_mean(tf.square(error))  # MSE (Y0 - Y1)^2

# Return the derivative of loss with respect to weight and bias (gradient)
def grad(weights_x, weights_y, biases):
  with tf.GradientTape() as tape:           # records for automatic differentiation.
    loss_value = loss(weights_x, weights_y, biases)
  return tape.gradient(loss_value, [weights_x, weights_y, biases])

def predict(x, y, w_x, w_y, b):
    return x * w_x + y * w_y + b


# ===== main = train =====
def main():

    # ===== Training Variables =====

    train_steps = 500
    learning_rate = 0.01

    # Start with arbitrary values for W and B on the same batch of data
    Wx = tfe.Variable(1.)
    Wy = tfe.Variable(1.)
    B = tfe.Variable(10.)

    # ===== Train =====

    print("Initial loss: {:.3f}".format(loss(Wx, Wy, B)))

    for i in range(train_steps):
      dWx, dWy, dB = grad(Wx, Wy, B)
      Wx.assign_sub(dWx * learning_rate)
      Wy.assign_sub(dWy * learning_rate)
      B.assign_sub(dB * learning_rate)
      if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(Wx, Wy, B)))

    print("Final loss: {:.3f}".format(loss(Wx, Wy, B)))
    print("Wx = {}, Wy = {}, B = {}".format(Wx.numpy(), Wy.numpy(), B.numpy()))

    print("Predict: 2 and 3")
    print(predict(2, 3, Wx, Wy, B))
    print(predict(2, 3, weight_x, weight_y, bias))



if __name__ == "__main__":
    main()
