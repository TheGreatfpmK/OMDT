#!/home/may/phd/storm/venv/bin/python

import os
import subprocess
import resource
import shutil
import sys
import re
import click
import math
import concurrent.futures

date="01-21"
experiment_group_name = f"{date}-omdt"

new_models = False

##### benchmarks evaluation ######

def set_memory_limit(maxmem_mb):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxmem_mb*1024*1024, hard))

@click.command()
@click.option('--omdt-dir', type=str, default="/home/may/synthesis", show_default=True, help='Path to the Paynt root folder.')
@click.option('--models-dir', type=str, default="/home/may/synthesis/models/cav", show_default=True, help='Path to the models folder.')
# @click.option('--paynt-dir', type=str, default="/opt/paynt", show_default=True, help='Path to the Paynt root folder.')
@click.option('--workers', type=int, default=4, show_default=True, help='Number of parallel tests.')
@click.option('--timeout', type=int, default=1200, show_default=True, help='Time limit for abstraction refinement (per model), seconds.')
@click.option('--maxmem', type=int, default=16, show_default=True, help='Memory limit, GB.')
@click.option('--output', type=str, default="results/logs", show_default=True, help='Name for the output logs folder.')
@click.option('--experiment-name', type=str, default=None, show_default=True, help='Name of the experiments.')
@click.option('--depth-min', type=int, default=1, show_default=True, help='Minimal depth for the exeperiments.')
@click.option('--depth-max', type=int, default=8, show_default=True, help='Maximal depth for the exeperiments.')
@click.option('--show-only', is_flag=True, default=False, show_default=True, help='Show results only.')
@click.option('--generate-csv', is_flag=True, default=False, show_default=True, help='Generate CSV file with results.')
@click.option('--restart', is_flag=True, help='Re-run all benchmarks.')
def main(omdt_dir, models_dir, workers, timeout, maxmem, output, experiment_name, depth_min, depth_max, show_only, generate_csv, restart):


    profiling = ""
    # profiling = " --profiling"
    # tree_enumeration = ""
    tree_enumeration = " --tree-enumeration"
    if experiment_name is not None:
        experiment_group_name = experiment_name

    models = [ f.path.split('/')[-1] for f in os.scandir(models_dir) if f.is_dir() ]

    different_gamma = {"consensus-3-32" : 0.9999, "philosophers-4": 0.99, "rabin-4": 0.99}
    qcomp_models = ["consensus-3-32", "csma-2-4", "firewire-3", "ij-10", "pnueli-zuck-3", "philosophers-4", "rabin-4", "resource-gathering-5", "wlan-1-2"]

    all_log_paths = []

    model_count = 1
    for model in models:
        model_tasks = []
        for d in range(depth_min,depth_max+1):
            task = (f"python3 run-experiment.py omdt {model} --seed 0 --gamma {0.99 if model not in list(different_gamma.keys()) else different_gamma[model]} --max_depth {d} --time_limit {timeout} --output_dir /opt/cav25-experiments/results/logs/{experiment_group_name}/ --verbose 1 --model-file-name {"model-random-enabled.drn" if model in qcomp_models else "model-random.drn"}", f"/opt/cav25-experiments/results/logs/{experiment_group_name}/{model}/log-depth-{d}.log", f"model {model_count}/{len(models)} depth {d}/{depth_max} - {model} -")
            model_tasks.append(task)
            all_log_paths.append(f"/opt/cav25-experiments/results/logs/{experiment_group_name}/{model}/log-depth-{d}.log")

        if not show_only:
            preexec_fn = lambda: set_memory_limit(maxmem*1024)

            if workers == 1:
                for task in model_tasks:
                    command, log_file, model_str = task
                    if os.path.exists(log_file) and not restart:
                        print(f"{model_str} Log file already exists. Skipping task.")
                        continue
                    print(f"{model_str} started")
                    try:
                        result = subprocess.run(command.split(), preexec_fn=preexec_fn, timeout=(timeout+120)*2, capture_output=True)
                        with open(log_file, 'w') as f:
                            f.write(result.stdout.decode())
                            f.write(result.stderr.decode())

                            if result.returncode != 0:
                                print(f"Error running task {model_str} for model {model} see {log_file} for details")

                    except Exception as e:
                        print(f"Error running task {model_str} for model {model}: {e}")
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    for task in model_tasks:
                        command, log_file, model_str = task
                        if os.path.exists(log_file) and not restart:
                            print(f"{model_str} Log file already exists. Skipping task.")
                            continue
                        print(f"{model_str} started")
                        try:
                            result = subprocess.run(command.split(), preexec_fn=preexec_fn, timeout=(timeout+120)*2, capture_output=True)
                            with open(log_file, 'w') as f:
                                f.write(result.stdout.decode())
                                f.write(result.stderr.decode())

                                if result.returncode != 0:
                                    print(f"Error running task {model_str} for model {model} see {log_file} for details")
                        except Exception as e:
                            print(f"Error running task {model_str} for model {model}: {e}")

            print(f"Finished running tasks for model {model}")
        model_count += 1


    if generate_csv:
        csv_file = os.path.join(f"/opt/cav25-experiments/results/logs/{experiment_group_name}/", "results-generated.csv")
        with open(csv_file, 'w') as f:
            f.write("model,max_depth,omdt time,omdt best,omdt bound,omdt depth\n")
            for log_path in all_log_paths:
                if os.path.exists(log_path):
                    model_name = log_path.split('/')[-2]
                    depth = log_path.split('-')[-1].split('.')[0]
                    with open(log_path, 'r') as log_file:
                        log_lines = log_file.readlines()
                        for i, line in enumerate(log_lines):
                            if line.startswith("Explored"):
                                data = line.split(' ')
                                time = data[7]
                            elif line.startswith("Best objective"):
                                data = line.split(' ')
                                best = data[2][:-1]
                                if best == "-":
                                    best = "-10000"
                                bound = data[5][:-1]
                        
                    f.write(f"{model_name},{depth},{time},{best},{bound},{depth}\n")
                else:
                    print(f"Log file {log_path} does not exist.")

if __name__ == '__main__':
    main()
