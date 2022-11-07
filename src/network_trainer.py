import numpy as np
import os
import tensorflow as tf


class NetworkTrainer:
    def __init__(self, lr_data, full_model, hr_data, epochs, batch_size):
        self.model = full_model
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        lr_dataset = tf.data.Dataset.from_tensor_slices(self.lr_data).batch(
            self.batch_size
        )
        hr_dataset = tf.data.Dataset.from_tensor_slices(self.hr_data).batch(
            self.batch_size
        )

    del self.lr_data
    del self.hr_data

    return
