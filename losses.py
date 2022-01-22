import tensorflow as tf


class GanLoss(tf.keras.losses.Loss):
    '''
    fake label : 0
    true label : 1
    '''
    def call(self, label, pred):
        loss = tf.reduce_mean(
            tf.losses.binary_crossentropy(label, pred)
        )
        return loss


class LsganLoss(tf.keras.losses.Loss):
    '''
    fake label : 0
    true label : 1
    '''
    def call(self, label, pred):
        loss = .5 * tf.reduce_mean(
            tf.losses.mean_squared_error(label, pred)
        )
        return loss


class WganLoss(tf.keras.losses.Loss):
    '''
    fake label : 1
    true label : -1
    '''
    def call(self, label, pred):
        loss = tf.reduce_mean(
            label * pred
        )
        return loss

