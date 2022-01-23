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


class AdaIn(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaIn, self).__init__()

    @tf.function
    def get_mean_stddev(self, x):
        mean, var = tf.nn.moments(x,
                                  axis=[1, 2],
                                  keepdims=True
                                  )
        stddev = tf.sqrt(var + 1e-7)
        return mean, stddev

    def call(self, x, y, *args, **kwargs):
        mean_x, stddev_x = self.get_mean_stddev(x)
        mean_y, stddev_y = self.get_mean_stddev(y)
        return stddev_y * (x - mean_x) / stddev_x + mean_y