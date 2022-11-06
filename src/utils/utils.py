import os

# Function to write the batch skript for training the network
def write_batch_skript(training_skript, path_to_run, gpu_type):
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


def move_files(training_skript, run_dir, configfile, networks, losses):
    base_dir = os.getcwd()
    config_dir = os.path.join(base_dir, "configs")
    # code_dir = os.path.join(base_dir, 'building_blocks')
    network_dir = os.path.join(base_dir, "networks")
    loss_dir = os.path.join(base_dir, "loss")

    origin_training_skript = os.path.join(base_dir, training_skript)
    dest_training_skript = run_dir
    os.system(f"cp {origin_training_skript} {dest_training_skript}")

    origin_configfile = os.path.join(config_dir, configfile)
    dest_configfile = run_dir
    os.system(f"cp {origin_configfile} {dest_configfile}")

    # dest_code = os.path.join(run_dir, code)
    # for skript in code:
    #     origin_skript = os.path.join(code_dir, skript)
    #     os.system(f'cp {origin_skript} {dest_code}')

    dest_networks = os.path.join(run_dir, "networks")
    for network in networks:
        origin_network = os.path.join(network_dir, network)
        os.system(f"cp {origin_configfile} {dest_networks}")

    dest_loss = os.path.join(run_dir, "loss")
    for loss in losses:
        origin_loss = os.path.join(loss_dir, loss)
        os.system(f"cp {origin_loss} {dest_loss}")

    return


if __name__ == "__main__":
    path = "/u/ivkos/new_sr/test"
    init_rundir(path)
