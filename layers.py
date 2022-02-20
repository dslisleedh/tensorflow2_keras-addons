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
    def __init__(self, eps=1e-8):
        super(AdaIn, self).__init__()
        self.eps = eps

    @tf.function
    def get_mean_stddev(self, x):
        mean, var = tf.nn.moments(x,
                                  axis=[1, 2],
                                  keepdims=True
                                  )
        stddev = tf.sqrt(var + self.eps)
        return mean, stddev

    def call(self, x, y, *args, **kwargs):
        mean_x, stddev_x = self.get_mean_stddev(x)
        mean_y, stddev_y = self.get_mean_stddev(y)
        return stddev_y * (x - mean_x) / stddev_x + mean_y


class ReflectPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectPadding2D, self).__init__()
        if len(padding) == 1:
            self.padding = (padding, padding)
        elif len(padding) > 2:
            raise ValueError('ReflectPadding2D only supports 2-dimensional pad.')
        else:
            self.padding = padding

    def call(self, inputs, *args, **kwargs):
        padded = tf.pad(input = inputs,
                        paddings=[[0, 0],
                                  [self.padding[0], self.padding[0]],
                                  [self.padding[1], self.padding[1]],
                                  [0, 0]
                                  ],
                        mode='REFLECT'
                        )
        return padded


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, eps=1e-8):
        super(PixelNormalization, self).__init__()
        self.eps = eps

    def call(self, inputs, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.eps)


class ReplicatePadding2D(tf.keras.layers.Layer):
    def __init__(self, n_pad):
        super(ReplicatePadding2D, self).__init__()
        self.n_pad = n_pad

    def call(self, inputs, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()
        top = tf.concat([inputs[:, :1, :, :] for _ in range(self.n_pad)],
                        axis=1
                        )
        bottom = tf.concat([inputs[:, h-1:, :, :] for _ in range(self.n_pad)],
                           axis=1
                           )
        inputs = tf.concat([top, inputs, bottom],
                           axis=1
                           )
        left = tf.concat([inputs[:, :, :1, :] for _ in range(self.n_pad)],
                         axis=2
                         )
        right = tf.concat([inputs[:, :, w-1:, :] for _ in range(self.n_pad)],
                          axis=2
                          )
        inputs = tf.concat([left, inputs, right],
                           axis=2
                           )
        return inputs
