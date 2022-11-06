import os
import csv


# Class which tracks the whole run i.e. it creates the folder to write
# the checkpoints to and writes loss, gradients etc. to the corresponding files
class RunTracker:
    def __init__(
        self,
        run_dir,
        track_loss=True,
        track_gradient=True,
        tracked_loss_prop=["epoch", "avg_loss_gen1"],
        tracked_grad_prop=["epoch", "avg_grad_gen1"],
    ):

        self.run_dir = run_dir
        self.check_dir = os.path.join(self.run_dir, "checkpoints")

        self.track_loss = track_loss
        self.track_gradient = track_gradient

        self.loss_file = None
        self.gradient_file = None

        self.tracked_loss_prop = tracked_loss_prop
        self.tracked_grad_prop = tracked_grad_prop

    # Create files for tracking the parameters of interest
    def initialize_run(self):
        if not os.path.isdir(self.check_dir):
            os.mkdir(self.check_dir)
            print(f"Created the checkpoint directory at {self.check_dir}")

        if self.track_loss:
            self.loss_file = os.path.join(self.run_dir, f"loss.csv")
            if not os.path.isfile(self.loss_file):
                with open(self.loss_file, "w", newline="") as loss_file:
                    writer = csv.writer(loss_file)
                    writer.writerow(self.tracked_loss_prop)(
                        f"Created the rundirectory at {self.run_dir}"
                    )
                print(f"Created loss file")

        if self.track_gradient:
            self.grad_file = os.path.join(self.run_dir, f"grad.csv")
            if not os.path.isfile(self.grad_file):
                with open(self.grad_file, "w", newline="") as grad_file:
                    writer = csv.writer(grad_file)
                    writer.writerow(self.tracked_grad_prop)
                print(f"Created grad file")

    # Updates loss file by writing the line with additional info
    def update_loss(self, data):
        if len(data) == len(self.tracked_loss_prop):
            with open(self.loss_file, "a", newline="") as loss_file:
                writer = csv.writer(loss_file)
                writer.writerow(data)
            print("Updated loss")
            return
        else:
            raise ValueError(
                f"Length of data ({len(data)}) must be equal to the number of properties being tracked ({len(self.tracked_loss_prop)})"
            )

    # Updates grad file by writing the line with additional info
    def update_grad(self, data):
        if len(data) == len(self.tracked_grad_prop):
            with open(self.grad_file, "a", newline="") as grad_file:
                writer = csv.writer(grad_file)
                writer.writerow(data)
            print("Updated gradient")
            return
        else:
            raise ValueError(
                f"Length of data ({len(data)}) must be equal to the number of properties being tracked ({len(self.tracked_loss_prop)})"
            )

    def get_tracked_properties(self, prop_type):
        if prop_type == "loss":
            return self.tracked_loss_prop
        elif prop_type == "grad":
            return self.tracked_grad_prop
        else:
            raise ValueError('Property must be "loss" of "grad"')
