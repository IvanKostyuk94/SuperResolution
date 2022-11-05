import numpy as np
import tensorflow as tf
import importlib

class prepare_run:

    def __init__(self, parameter_file, generate_train_samples=False, train_sim=None, train_samples=None, scale=False, scale_func=None):
        self.parameter_file = parameter_file
        if (train_sim==None) and (train_samples==None):
            raise FileNotFoundError('Must provide either path to the train_sim or to the train_samples')
        self._import_parameter_file()



    def _import_parameter_file(self):
        self.p = importlib.import_module(self.parameter_file)
        return