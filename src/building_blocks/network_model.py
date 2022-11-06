import tensorflow as tf
from tensorflow.keras import layers


class NetworkModel:
    def __init__(
        self,
        network,
        optimizer,
        in_x=None,
        in_y=None,
        in_z=None,
        channels=1,
        dim3=False,
    ):
        if in_z is None:
            self.input = layers.Input(shape=(in_x, in_y, channels))
        else:
            self.input = layers.Input(shape=(in_x, in_y, in_z, channels))

        self.model = tf.keras.Model(
            inputs=self.input, outputs=network(self.input)
        )
        self.optimizer = optimizer

    def apply(self, input_tile):
        return self.model.predict(input_tile)
