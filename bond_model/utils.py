import random

import tensorflow as tf


class MaskCalculator(tf.keras.layers.Layer):
    def __init__(self, output_dim: int, pad_token_id: int, **kwargs):
        self.output_dim = output_dim
        self.pad_token_id = pad_token_id
        super(MaskCalculator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCalculator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.permute_dimensions(
            x=tf.keras.backend.repeat(
                x=tf.keras.backend.cast(
                    x=tf.math.not_equal(
                        x=inputs,
                        y=self.pad_token_id
                    ),
                    dtype='float32'
                ),
                n=self.output_dim
            ),
            pattern=(0, 2, 1)
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        shape = list(input_shape)
        shape.append(self.output_dim)
        return tuple(shape)

    def get_config(self):
        return {
            "output_dim": self.output_dim,
            "pad_token_id": self.pad_token_id
        }


class AttentionMaskLayer(tf.keras.layers.Layer):
    def __init__(self, pad_token_id: int, **kwargs):
        self.pad_token_id = pad_token_id
        super(AttentionMaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionMaskLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.cast(
            x=tf.math.not_equal(
                x=inputs,
                y=self.pad_token_id
            ),
            dtype='float32'
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        return input_shape

    def get_config(self):
        return {"pad_token_id": self.pad_token_id}


def generate_random_seed() -> int:
    return random.randint(0, 2147483646)
