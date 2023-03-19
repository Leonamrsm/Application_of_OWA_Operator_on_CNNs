import tensorflow as tf

from keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import conv_utils
# import skimage.measure

@keras_export('keras.constraints.UnitSumNonNeg', 'keras.constraints.unit_sum_non_neg')
class UnitSumNonNeg(Constraint):
    """Limits weights to be non-negative and with sum equal to one

    Also available via the shortcut function `keras.constraints.unit_sum_non_neg`.
    """
    def __call__(self, w):
        aux =  w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

        return aux/(K.epsilon() + tf.reduce_sum(aux, axis=[0], keepdims=True))

class OWAPoolingNew(tf.keras.layers.Layer):
    def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding=(0,0),
               data_format=None,
               name=None,
               sort=True,
               train=True, 
               seed=None,
               all_channels=False,
               **kwargs):
        super(OWAPoolingNew, self).__init__(name=name, **kwargs)

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
        weights_shape = (self.pool_size[0] * self.pool_size[1], input_shape[-1])
      else:
        weights_shape = (self.pool_size[0] * self.pool_size[1], 1)
      
      tf.random.set_seed(self.seed)
      kernel = tf.random.uniform(shape=weights_shape)
      kernel /= tf.reduce_sum(kernel, axis=[0], keepdims=True)
      
      self.kernel = tf.Variable(initial_value = kernel, trainable=self.train, dtype='float32', constraint=UnitSumNonNeg())

    def call(self, inputs):

        padding = self.padding
        if padding != (0,0):
            paddings = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
            inputs = tf.pad(inputs, paddings, "CONSTANT")

        # Extract pooling regions
        n, h, w, c = inputs.shape.as_list()
        custom_stride = self.strides != self.pool_size
        pool_size = self.pool_size

        if custom_stride:
            grid = []
            for i in range(0, h+1-pool_size[0], self.strides[0]):
                grid.append([])
                for j in range(0, w+1-pool_size[1], self.strides[1]):
                    grid[-1].append(inputs[:,i:i+pool_size[0], j:j+pool_size[1],:])
            inputs = K.stack([K.stack(row, axis=2) for row in grid], axis=1) # NHkWkC
        else:
            inputs = K.reshape(inputs, [-1, h//pool_size[0], pool_size[0], w//pool_size[1], pool_size[1], c])

        inputs = tf.transpose(inputs,[0,1,3,2,4,5])
        inputs = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2], pool_size[0]*pool_size[1], c])

        # Sort values for pooling
        if self.sort:
            inputs = tf.sort(inputs, axis=-2, direction='DESCENDING', name=None)

        outputs = tf.reduce_sum(tf.math.multiply(self.kernel, inputs), axis=-2)

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
                self.kernel,
            'kernel_size':
                self.kernel.shape,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format
        }
        base_config = super(OWAPoolingNew, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))