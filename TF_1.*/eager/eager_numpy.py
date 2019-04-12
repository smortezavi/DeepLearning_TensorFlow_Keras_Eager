# https://www.tensorflow.org/guide/eager

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()        # => True

print("\nSimple Tensor Math with Eager")
print("from https://www.tensorflow.org/guide/eager")

print("\n2 x 2 as a tensor")
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"


# ==== Eager and Numpy =====
print("\n2D numpy array as a tensor")
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
print("\n2D numpy array + 1 with tensors")
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print("\n2D tensor * 2D tensor")
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# Use NumPy values
import numpy as np

print("\n2D tensor * 2D tensor with numpy")
c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# Obtain numpy value from a tensor:
print("\n2D tensor with numpy")
print(a.numpy())
# => [[1 2]
#     [3 4]]
