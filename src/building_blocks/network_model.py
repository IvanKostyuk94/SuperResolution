from tensorflow.keras import layers
import tensorflow as tf


class NetworkModel:
    """
    Creates a trainable network model
    """

    def __init__(
        self, network_function, optimizer, input_shape=(None, None, None, 1)
    ):
        """
        Initialize the network model

        Args:
            network_function (function): The function of the network
            optimizer (tensorflow.python.keras.optimizer_v2): optimizer to be used for this network
            input_shape (tuple, optional): The shape of the network input, default is for a 3D generator.
                                        Critic requires the actual shape of the generator output.
                                        Defaults to (None, None, None, 1).
        """
        self.network_function = network_function
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.network_model = self._build_model()

    def _build_model(self):
        """
        Builds a keras Model out of the network funciton

        Returns:
            tensorflow.python.keras.engine.training.Model: Trainable keras model
        """
        input = layers.Input(shape=self.input_shape)
        model = tf.keras.Model(
            inputs=input, outputs=self.network_function(input)
        )
        return model

    def apply(self, data):
        return self.network_model(data)
