# Code used for generating training tiles out of gridded simulations
import numpy as np


class DatasetGenerator:
    """
    A class to generate a training dataset.

    Attributes
    ----------
    lr_simulation_path(str):
        Path to the low resolution simulation box
    hr_simulation_path(str):
        Path to the low resolution simulation box
    lr_dir(str):
        Path to the low resolution simulation training data
    hr_dir(str):
        Path to the high resolution simulation training data
    lr_name(str):
        Name for the low resolution training boxes
    hr_name(str):
        Name for the high resolution training boxes
    scaling_functio(function):
        Function with which to scale the data before training. Defaults to None.
    grid_size(int):
        Size of the training boxes. Defaults to 64.
    lr_simulation(numpy.ndarray):
        Low resolution simulation box should have the same size as the hr box
    hr_simulation(numpy.ndarray):
        High resolution simulation box
    boxes_per_dim(int):
        How many training boxes per sides of simulation box
    n_transformations(int):
        Number of transformations for data augmentation including unity

    Methods
    -------
    _load_simulation(simulation_path):
        Load gridded simulation from path

    _create_dataset(simulation, output_dir, output_name):
        Creates a dataset from a gridded simulation cube saved in a set of npy files
    create_lr_dataset():
        Create the low resolution dataset for training
    create_hr_dataset():
        Create the high resolution dataset for training
    """

    def __init__(
        self,
        lr_simulation_path,
        hr_simulation_path,
        lr_dir,
        hr_dir,
        lr_name,
        hr_name,
        scaling_function=None,
        grid_size=64,
    ):
        """Constructs all necessary attributes for the dataset generator object

        Args:
            lr_simulation_path(str):
                Path to the low resolution simulation box
            hr_simulation_path(str):
                Path to the low resolution simulation box
            lr_dir(str):
                Path to the low resolution simulation training data
            hr_dir(str):
                Path to the high resolution simulation training data
            lr_name(str):
                Name for the low resolution training boxes
            hr_name(str):
                Name for the high resolution training boxes
            scaling_functio(function):
                Function with which to scale the data before training. Defaults to None.
            grid_size(int):
                Size of the training boxes. Defaults to 64.
        """
        self.lr_simulation_path = lr_simulation_path
        self.hr_simulation_path = hr_simulation_path
        self.lr_simulation = self._load_simulation(self.lr_simulation_path)
        self.hr_simulation = self._load_simulation(self.hr_simulation_path)
        if scaling_function is not None:
            self.lr_simulation = scaling_function(self.lr_simulation)
            self.hr_simulation = scaling_function(self.hr_simulation)

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_name = lr_name
        self.hr_name = hr_name
        self.grid_size = grid_size
        self.boxes_per_dim = self.hr_simulation.shape[0] // self.grid_size
        self.n_transformations = 10

        self.random_array = np.random.permutation(
            np.arange(
                self.boxes_per_dim
                * self.boxes_per_dim
                * self.boxes_per_dim
                * self.n_transformations
            )
        )

    def _load_simulation(self, simulation_path):
        """Load gridded simulation from path

        Args:
            simulation_path (str): path to the gridded simulation

        Returns:
            numpy.ndarray: simulation box
        """
        return np.load(simulation_path)

    def _create_dataset(self, simulation, output_dir, output_name):
        """Creates a dataset from a gridded simulation cube saved in a set of npy files

        Args:
            simulation (numpy.ndarray): Simulation box should be cube shaped
            output_dir (str): Path to the directory where to store the training data
            output_name (str): Name of the training data boxes
            grid_size (int, optional): Size of the training boxes. Defaults to 64.
        """

        counter = 0
        for i in range(self.boxes_per_dim):
            for j in range(self.boxes_per_dim):
                for k in range(self.boxes_per_dim):
                    cube = simulation[
                        i * self.grid_size : (i + 1) * self.grid_size,
                        j * self.grid_size : (j + 1) * self.grid_size,
                        k * self.grid_size : (k + 1) * self.grid_size,
                    ]

                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        cube,
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.flip(cube, 0),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.flip(cube, 1),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.flip(cube, 2),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.rot90(cube, 1, axes=(0, 1)),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.rot90(cube, 2, axes=(0, 1)),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.rot90(cube, 3, axes=(0, 1)),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.rot90(cube, 1, axes=(1, 2)),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.rot90(cube, 2, axes=(1, 2)),
                    )
                    k += 1
                    np.save(
                        output_dir
                        + output_name
                        + str(self.random_array[counter]),
                        np.rot90(cube, 3, axes=(1, 2)),
                    )
                    counter += 1
        return

    def create_lr_dataset(self):
        """Create the low resolution dataset for training"""
        self._create_dataset(
            simulation=self.lr_dir,
            output_dir=self.lr_dir,
            output_name=self.lr_name,
        )
        return

    def create_hr_dataset(self):
        """Create the high resolution dataset for training"""
        self._create_dataset(
            simulation=self.hr_dir,
            output_dir=self.hr_dir,
            output_name=self.hr_name,
        )
        return
