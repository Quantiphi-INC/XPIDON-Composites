from jax.nn import relu
from jax import random
import jax.numpy as np
"""
Multi-Layer Perceptron (MLP)** in **JAX** to serve as the backbone of our DeepONet model
"""
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      """Initialize weights with Xavier initialization and biases with zeros."""
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b

      key, *keys = random.split(rng_key, len(layers))
      """ map: is a function that executes a specified function for each item in an iterable.
      The reason map is used here is that we need to apply init_layer to each layer for generating its initial parameters."""
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params

  def apply(params, inputs):
      """Perform forward pass through the MLP."""
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply