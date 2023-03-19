import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import conv_utils

@keras_export('keras.constraints.UnitSumNonNeg', 'keras.constraints.unit_sum_non_neg')
class UnitSumNonNeg(Constraint):
    """Limits weights to be non-negative and with sum equal to one

    Also available via the shortcut function `keras.constraints.unit_sum_non_neg`.
    """
    def __call__(self, w):
        aux =  w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

        return aux/(K.epsilon() + tf.reduce_sum(aux, axis=[-1], keepdims=True))


class OWAConv_fm(tf.keras.layers.Layer):
    def __init__(self,
               filters, 
               strides=(1, 1),
               data_format=None,
               name=None,
               train=True, 
               seed=None,
               **kwargs):
        super(OWAConv_fm, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.strides = strides
        self.data_format = conv_utils.normalize_data_format('channels_last')
        self.train = train
        self.seed = seed if seed != None else 10
        
    def build(self, input_shape):
      

      weights_shape = (1, 1, input_shape[-1], self.filters)

      tf.random.set_seed(self.seed)
      kernel = tf.random.uniform(shape=weights_shape)
      kernel = kernel/tf.reduce_sum(kernel, axis=[-1], keepdims=True)
      
      self.kernel = tf.Variable(initial_value = kernel, trainable=self.train, dtype='float32', constraint=UnitSumNonNeg())

    def sort_p(self, inputs):

        if inputs.shape[0] == None:
            return inputs

        indexes = tf.argsort(tf.math.reduce_sum(inputs, axis=[1,2], keepdims=False), axis=-1)

        output_list = []

        for j in range(indexes.shape[1]):
            list_aux = []

            for i in range(inputs.shape[0]):
                indx = indexes[i,:]
                list_aux.append(inputs[i,:,:,indx==inputs.shape[-1]-1-j])

            output_list.append(list_aux)

        outputs = tf.reshape(tf.stack(output_list, axis=-1), inputs.shape)

        return outputs

    def call(self, inputs):

        # Sort values for pooling
        inputs = self.sort_p(inputs)

        outputs = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='VALID')

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        # if self.data_format == 'channels_first':
        #     rows = input_shape[2]
        #     cols = input_shape[3]
        # else:
        rows = input_shape[1]
        cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, 1, 'VALID',
                                         self.strides[0])
        cols = conv_utils.conv_output_length(cols, 1, 'VALID',
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
        base_config = super(OWAConv_fm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))