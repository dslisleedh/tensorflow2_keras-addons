import tensorflow as tf


class DropconnectDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(DropconnectDense, self).__init__(*args, **kwargs)
        self.prob = kwargs.pop('prob', .5)

        self.dropout = tf.keras.layers.Dropout(self.prob)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[int(input_shape[-1]),
                                             self.units],
                                      trainable=True)
        self.bias = self.add_weight('bias',
                                    shape=[self.units, ],
                                    trainable=True
                                    )

    def call(self, inputs, **kwargs):
        y = tf.matmul(inputs, self.dropout(self.kernel))
        if self.use_bias:
            y = tf.nn.bias_add(y, self.dropout(self.bias))
        return self.activation(y)
