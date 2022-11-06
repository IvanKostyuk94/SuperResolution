from tensorflow import keras
from tensorflow.keras import layers


def cnn_3d(input_batch):
    """
    A simple 3D CNN with 3 layers which is meant to learn the changes between
    HR and LR with the LR being forwarded directly through a shortcut

        Arguments:
            batch {list}: A batch of 3D LR cubes

        Returns:
            X {list}: A batch of 3D HR cubes

    """
    X_short = input_batch

    X = layers.Conv3D(
        filters=4,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="layer1",
    )(input_batch)
    X = layers.LeakyReLU()(X)

    X = layers.Conv3D(
        filters=8,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="layer2",
    )(input_batch)
    X = layers.LeakyReLU()(X)

    X = layers.Conv3D(
        filters=1,
        kernel_size=(1, 1, 1),
        padding="same",
        use_bias=True,
        kernel_initializer=keras.initializers.RandomNormal(),
        name="final_layer",
    )(X)
    X = layers.ReLU()(X)
    X = X + X_short
    return X
