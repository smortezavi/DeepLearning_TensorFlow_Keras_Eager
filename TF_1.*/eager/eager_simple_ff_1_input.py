# eager_simple_feed_forward.py
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable Eager.
tfe.enable_eager_execution()

# ===== Global Variables =====

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000             # 1000 examples
training_inputs = tf.random_normal([NUM_EXAMPLES])     # The Tensor of 1000 examples
noise = tf.random_normal([NUM_EXAMPLES])               # Noise of 1000 examples.
training_outputs = training_inputs * 3 + 2 + noise     # Output of 1000 examples.

# ===== Prediction, Loss and Gradient =====

# Prediction Function using input, weight and bias.
def prediction(input, weight, bias):
  return input * weight + bias

# A loss function using mean-squared error (MSE)
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))  # MSE (Y0 - Y1)^2

# Return the derivative of loss with respect to weight and bias (gradient)
def grad(weights, biases):
  with tf.GradientTape() as tape:           # records for automatic differentiation.
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])

def predict(x, w, b):
    return x * w + b


# ===== main = train =====
def main():

    # ===== Training Variables =====

    train_steps = 200
    learning_rate = 0.02
    # Start with arbitrary values for W and B on the same batch of data
    W = tfe.Variable(5.)
    B = tfe.Variable(10.)

    # ===== Train =====

    print("Initial loss: {:.3f}".format(loss(W, B)))

    for i in range(train_steps):
      dW, dB = grad(W, B)
      W.assign_sub(dW * learning_rate)
      B.assign_sub(dB * learning_rate)
      if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

    print("Final loss: {:.3f}".format(loss(W, B)))
    print("W = {}, B = {}".format(W.numpy(), B.numpy()))

    print("Predict: 2")
    print(predict(2, W, B))



if __name__ == "__main__":
    main()
