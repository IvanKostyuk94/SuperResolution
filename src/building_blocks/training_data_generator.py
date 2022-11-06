import numpy as np
import os


class TrainingDataGenerator:
    def __init__(
        self, path_to_features, path_to_labels, run_dir, is_generated=False
    ):

        self.path_to_features = path_to_features
        self.path_to_labels = path_to_labels

        self.run_dir = run_dir
        self.training_data_dir = self._get_training_data_dir()
        self.feature_dir = self._get_feature_dir()
        self.label_dir = self._get_label_data_dir()

        self.is_generated = is_generated

    def _get_training_set_dir(self):
        return os.path.join(self.run_dir, "train")

    def _get_feature_dir(self):
        return os.path.join(self.training_data_dir, "features")

    def _get_label_dir(self):
        return os.path.join(self.training_data_dir, "labels")
