from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf

class InstanceNormalization(Layer):
	def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
		super(InstanceNormalization, self).__init__(**kwargs)
		self.axis = axis
		self.epsilon = epsilon

	def build(self, input_shape):
		dim = input_shape[self.axis]
		if dim is None:
			raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
		shape = (dim,)

		self.gamma = self.add_weight(shape=shape, name='gamma', initializer='ones')
		self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
		self.built = True

	def call(self, inputs, training=None):
		mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
		return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)