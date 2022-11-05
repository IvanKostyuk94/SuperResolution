import tensorflow as tf 
import run_tracker
import full_model

# Function which takes 
class step:

    def __init__(self, full_model, run_tracker):
        self.model = full_model
        self.tracker = run_tracker

    def apply(self):
        if self.model.is_multipass:
            if self.model.is_gan:
                gen1_loss, crit1_loss, gen1_loss, crit1_gradients = 