#Since MNIST data is so frequently used for demonstration purposes, Tensorflow provides a way
#to automatically download it.
from tensorflow.examples.tutorials.mnist import input_data   # import input_data  error
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)