import tensorflow as tf

from keras import backend as K
from tensorflow import keras
from keras.constraints import Constraint
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import conv_utils

@keras_export('keras.constraints.UnitSumNonNeg', 'keras.constraints.unit_sum_non_neg')
class UnitSumNonNeg(Constraint):
    """Limits weights to be non-negative and with sum equal to one

    Also available via the shortcut function `keras.constraints.unit_sum_non_neg`.
    """
    def __call__(self, w):
        aux =  w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

        return aux/(K.epsilon() + tf.reduce_sum(aux, axis=[1], keepdims=True))

class OWAPooling(tf.keras.layers.Layer):
    def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               name=None,
               sort=True,
               train=True, 
               seed=None,
               all_channels=False,
               **kwargs):
        super(OWAPooling, self).__init__(name=name, **kwargs)

        self.pool_size = pool_size
        self.strides = pool_size if strides == None else strides
        self.padding = padding
        self.data_format = conv_utils.normalize_data_format('channels_last')
        self.sort = sort
        self.train = train
        self.seed = seed if seed != None else 10
        self.all_channels = all_channels
        
    def build(self, input_shape):
      
      if self.all_channels:
        weights_shape = (input_shape[-1], self.pool_size[0] * self.pool_size[1])
      else:
        weights_shape = (1, self.pool_size[0] * self.pool_size[1])
      
      tf.random.set_seed(self.seed)
      kernel = tf.random.uniform(shape=weights_shape)
      kernel /= tf.reduce_sum(kernel, axis=[1], keepdims=True)
      
      self.kernel = tf.Variable(initial_value = kernel, trainable=self.train, dtype='float32', constraint=UnitSumNonNeg())

    def sort_p(self, inputs):

        _, pool_height, pool_width, channels, elems = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1, elems]) # Reshape tensor
        inputs = tf.sort(inputs, axis=-1, direction='DESCENDING', name=None)
        # Reshape
        inputs = tf.reshape(inputs, [-1, pool_height, pool_width, channels, elems]) # Reshape tensor
        return inputs

    def call(self, inputs):

        _, height, width, channels = inputs.get_shape().as_list()
        pad_bottom = self.pool_size[0] * height%self.pool_size[0]
        pad_right = self.pool_size[1] * width%self.pool_size[1]

        if(self.padding.upper()=='SAME'): # Complete size to pad 'SAME'
            paddings = tf.constant([[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]])
            inputs = tf.pad(inputs, paddings, "CONSTANT")

        # Extract pooling regions
        stride = [1, self.strides[0], self.strides[1], 1]
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]

        inputs = tf.image.extract_patches(inputs, sizes = ksize, strides = stride,
                            rates = [1, 1, 1, 1], padding='SAME')

        _, pool_height, pool_width, elems = inputs.get_shape().as_list()

        # Extract pooling regions for each channel
        elems =  int(elems / channels)
        inputs = tf.reshape(inputs, [-1, pool_height, pool_width, elems, channels]) # Reshape tensor
        inputs = tf.transpose(inputs,perm = [0, 1, 2, 4, 3])

        # Sort values for pooling
        if self.sort:
            inputs = self.sort_p(inputs)

        kernel = self.kernel

        kernel = tf.expand_dims(kernel, axis=0)
        kernel = tf.expand_dims(kernel, axis=0)
        kernel = tf.expand_dims(kernel, axis=0)

        if inputs.shape[0] != None:
          kernel = tf.repeat(kernel, axis=0, repeats = inputs.shape[0])
        kernel = tf.repeat(kernel, axis=1, repeats = inputs.shape[1])
        kernel = tf.repeat(kernel, axis=2, repeats = inputs.shape[2])

        if not self.all_channels:
            kernel = tf.repeat(kernel, axis=3, repeats = inputs.shape[3])

        outputs = tf.math.multiply(kernel, inputs)
        outputs = tf.reduce_sum(outputs,4)  #Reduce de 4th dimension
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        # if self.data_format == 'channels_first':
        #     rows = input_shape[2]
        #     cols = input_shape[3]
        # else:
        rows = input_shape[1]
        cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
        # if self.data_format == 'channels_first':
        #     return tf.TensorShape(
        #         [input_shape[0], input_shape[1], rows, cols])
        # else:
        return tf.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])
            
    def get_config(self):
        config = {
            'kernel':
                self.kernel_TENSOR,
            'kernel_size':
                self.kernel_TENSOR.shape,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format
        }
        base_config = super(OWAPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))