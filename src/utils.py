import os


# Function to write the batch skript for training the network
def write_batch_skript(training_skript, path_to_run, gpu_type):
    """Write the batch skript to be submitted in order to train the network

    Args:
        training_skript (str): Python file to run
        path_to_run (str): Path in which to safe the training run
        gpu_type (str): Which GPU type should be used
                    Options: gtx5000 or v100
    """
    job_output_dir = os.path.join(path_to_run, "batch_output")
    out_path = os.path.join(job_output_dir, "tjob.out.%j")
    err_path = os.path.join(job_output_dir, "tjob.err.%j")
    exe_path = os.path.join(path_to_run, training_skript)
    job_path = os.path.join(path_to_run, "submit_training.sh")

    if gpu_type == "gtx5000":
        memory = 15000
    elif gpu_type == "v100":
        memory = 32000

    else:
        raise NotImplementedError(f"{gpu_type} not supported")

    with open(job_path, "w") as job:
        job.write(
            """#!/bin/bash -l
# Standard output and error: 
"""
        )

        job.write(f"#SBATCH -o {out_path} \n")
        job.write(f"#SBATCH -e {err_path}")

        job.write(
            """
# Initial working directory:
#SBATCH -D ./
#
#SBATCH -J test_slurm
#
# Node feature: 
#SBATCH --constraint="gpu" 
# Specify type and number of GPUs to use: 
#   GPU type can be v100 or rtx5000
"""
        )

        job.write(f"#SBATCH --gres=gpu:{gpu_type}:1 \n")
        job.write(f"#SBATCH --mem={memory}")

        job.write(
            """
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
# #SBATCH --ntasks-per-node=40     
#SBATCH --ntasks-per-node=1  
#
#SBATCH --mail-type=all
#SBATCH --mail-user=ivkos@mpa-garching.mpg.de
#
# wall clock limit:
#SBATCH --time=24:00:00
module load cuda
module load cudnn
module load anaconda
# #module load tensorflow

# Run the program:
"""
        )

        job.write(f"srun python {exe_path}")

    return


# Funktion to create the run_dir and all its subdirs
def init_rundir(path_to_run):
    """Function to initialize all directories for the training run

    Args:
        path_to_run (str): Path where the results of the run should be stored
    """
    os.mkdir(path_to_run)

    job_output_dir = os.path.join(path_to_run, "batch_output")
    os.mkdir(job_output_dir)

    networks_dir = os.path.join(path_to_run, "networks")
    os.mkdir(networks_dir)

    code_dir = os.path.join(path_to_run, "code")
    os.mkdir(code_dir)

    loss_dir = os.path.join(path_to_run, "loss")
    os.mkdir(loss_dir)

    checkpoints_dir = os.path.join(path_to_run, "checkpoints")
    os.mkdir(checkpoints_dir)

    data_dir = os.path.join(path_to_run, "training_data")
    os.mkdir(data_dir)

    features_dir = os.path.join(data_dir, "features")
    os.mkdir(features_dir)
    labels_dir = os.path.join(data_dir, "labels")
    os.mkdir(labels_dir)

    return


def move_files(run_dir, training_skript="run.py", configfile="config.py"):
    """Move the config file and training skript to the run dir before training the network

    Args:
        run_dir (str): Path to the directory where the network should be trained
        configfile (str): Name of the configuration file. Defaults to config.py
        training_skript (str): Name of the network training skript: Defaults to run.py
    """
    base_dir = os.getcwd()

    origin_training_skript = os.path.join(base_dir, training_skript)
    dest_training_skript = run_dir
    os.system(f"cp {origin_training_skript} {dest_training_skript}")

    origin_configfile = os.path.join(base_dir, configfile)
    dest_configfile = run_dir
    os.system(f"cp {origin_configfile} {dest_configfile}")
    return
