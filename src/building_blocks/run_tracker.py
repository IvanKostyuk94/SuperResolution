import os
import csv


# Class which tracks the whole run i.e. it creates the folder to write
# the checkpoints to and writes loss, gradients etc. to the corresponding files
class RunTracker:
    """
    Class which tracks the whole run i.e. it creates the folder to write
    the checkpoints to and writes loss, gradients etc. to the corresponding files
    """

    def __init__(
        self,
        run_dir,
        track_loss=True,
        track_gradient=True,
        tracked_loss_prop=["epoch", "avg_loss_gen1"],
        tracked_grad_prop=["epoch", "avg_grad_gen1"],
    ):
        """
        Initialize the the run tracker


        Args:
            run_dir (str): Directory of the run
            track_loss (bool, optional): Should the loss be tracked? Defaults to True.
            track_gradient (bool, optional): Should the gradients be tracked? Defaults to True.
            tracked_loss_prop (list, optional): Which losses and connected properties should be tracked.
                                                Defaults to ["epoch", "avg_loss_gen1"].
            tracked_grad_prop (list, optional): Which gradients and connected properties should be tracked.
                                                Defaults to ["epoch", "avg_grad_gen1"].
        """

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
                print("Created loss file")

        if self.track_gradient:
            self.grad_file = os.path.join(self.run_dir, f"grad.csv")
            if not os.path.isfile(self.grad_file):
                with open(self.grad_file, "w", newline="") as grad_file:
                    writer = csv.writer(grad_file)
                    writer.writerow(self.tracked_grad_prop)
                print("Created grad file")

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
        """
        Prints which properties are being tracked

        Args:
            prop_type (str): loss or grad (gradient)

        Returns:
            list: List of properties being tracked
        """
        if prop_type == "loss":
            return self.tracked_loss_prop
        elif prop_type == "grad":
            return self.tracked_grad_prop
        else:
            raise ValueError('Property must be "loss" of "grad"')
