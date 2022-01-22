import tensorflow


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)
