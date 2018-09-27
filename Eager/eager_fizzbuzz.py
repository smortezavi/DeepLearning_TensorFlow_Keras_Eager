from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(max_num.numpy()):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter

print("\nFizzbuzz: 9")
fizzbuzz(9)

print("\nFizzbuzz: 15")
fizzbuzz(15)

print("\nFizzbuzz: 29")
fizzbuzz(29)
