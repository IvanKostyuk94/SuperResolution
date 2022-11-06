import sys
import importlib
import utils
import os
config = sys.argv[1][:-3].replace('/','.')
configfile = sys.argv[1].split('/')[1]
p = importlib.import_module(config)

cwd = os.getcwd()
path_to_runs = os.path.join(cwd[:-4], 'runs')
run_path = os.path.join(path_to_runs, p.run_name)
print(run_path)
if not os.path.exists(run_path):
    utils.init_rundir(run_path)
    utils.write_batch_skript(p.training_skript, run_path, p.gpu_type)
    utils.move_files(p.training_skript, run_dir=run_path, configfile=configfile, networks=p.networks, losses=p.losses)
else: 
    raise ValueError(f'The run folder {p.run_name} already exists.')
