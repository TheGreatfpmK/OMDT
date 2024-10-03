import argparse
import importlib
import os
import time
import warnings
from pathlib import Path

import numpy as np

from omdt.mdp import MarkovDecisionProcess

import json

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run on a given environment")
parser.add_argument("algorithm", type=str, help="the algorithm to train with")
parser.add_argument("env_name", type=str, help="the environment (MDP) to train on")
parser.add_argument(
    "--max_depth",
    default=None,
    type=int,
    help="maximum decision tree depth, trees will be trained with depth 1...max_depth. Ignored by dtcontrol",
)
parser.add_argument(
    "--time_limit",
    default=None,
    type=int,
    help="time limit for solving one tree in seconds, by default train until optimality",
)
parser.add_argument(
    "--n_cpus", default=1, type=int, help="number of CPU cores to train on"
)
parser.add_argument(
    "--gamma",
    default=0.99,
    type=float,
    help="discount factor for future states (prevents infinite MDP problems)",
)
parser.add_argument(
    "--verbose",
    default=0,
    type=int,
    help="verbosity level, 0 (no prints), 1 (solver logs), 2 (all logs)",
)
parser.add_argument(
    "--output_dir",
    default="experiments/out-philosophers-4/",
    type=str,
    help="base directory for outputting files, files are created under out/environment/",
)
parser.add_argument(
    "--max_iter",
    default=1000,
    type=int,
    help="maximum number of value iteration iterations",
)
parser.add_argument(
    "--delta",
    default=1e-10,
    type=float,
    help="stop value iteration after value updates get this small",
)
parser.add_argument("--seed", default=0, type=int, help="random seed for the solver")
parser.add_argument(
    "--record_progress",
    default=False,
    type=bool,
    help="record solver objective and bounds over time",
)
parser.add_argument(
    "--export_graphviz",
    default=True,
    type=bool,
    help="visualize the learned trees with graphviz, needs dot command in path",
)
parser.add_argument(
    "--input-path",
    default='models/',
    type=str,
    help="path to directory that includes the inputs",
)
parser.add_argument(
    "--export-valuations",
    default=False,
    type=bool,
    help="if set state valuations json is exported",
)
parser.add_argument(
    "--model-file-name",
    default="sketch.drn",
    type=str,
    help="name of the model file",
)
parser.add_argument(
    "--scheduler-name",
    default="scheduler",
    type=str,
    help="name of the scheduler",
)


args = parser.parse_args()

# Create an output directory if it does not yet exist
mdp_output_dir = f"{args.output_dir}{args.env_name}/"
Path(mdp_output_dir).mkdir(parents=True, exist_ok=True)

if False:
    # Generate the mdp, possibly with extra arguments
    print(f"Generating MDP for {args.env_name}...")
    environment = importlib.import_module(f"environments.{args.env_name}")
    mdp = environment.generate_mdp()
elif True:
    mdp_file = f"{args.input_path}/{args.env_name}/{args.model_file_name}"
    with open(mdp_file, "r") as f:
        mdp_lines = f.readlines()
        nr_states = 0
        nr_actions = 0
        feature_names = []
        action_names = []
        for i, line in enumerate(mdp_lines):
            if line.startswith("@nr_states"):
                nr_states = int(mdp_lines[i+1])
            if line.startswith("@nr_choices"):
                nr_actions = int(mdp_lines[i+1])//nr_states
            if line.startswith("state 0"):
                features = mdp_lines[i+1].replace('[', '').replace(']','').replace('/','').replace('&','').strip().split()
                for feature in features:
                    if feature.startswith('!'):
                        feature = feature.replace('!', '')
                    elif feature.count('=') > 0:
                        feature = feature.split('=')[0]
                    feature_names.append(feature)

                j = i+2
                while (not mdp_lines[j].startswith("state")) and (j < len(mdp_lines)):
                    processed_line = mdp_lines[j].strip()
                    if processed_line.startswith("action"):
                        action_info = processed_line.replace('[', '').replace(']','').split()
                        action_names.append(action_info[1])
                    j += 1
                    if j == len(mdp_lines):
                        break

                break

        assert nr_actions == len(action_names)
        trans_probs = np.zeros((nr_states, nr_states, nr_actions))
        rewards = np.zeros((nr_states, nr_states, nr_actions))  
        initial_state = np.zeros(nr_states)
        observations = np.empty((nr_states, len(feature_names)))

        for i, line in enumerate(mdp_lines):
            if line.startswith("state"):
                state_info = line.split()
                state_index = int(state_info[1])
                if "init" in state_info:
                    initial_state[state_index] = 1
                if "discount_sink" in state_info:
                    for feature_index, feature_name in enumerate(feature_names):
                        observations[state_index, feature_index] = 0
                else:
                    feature_info = mdp_lines[i+1].replace('[', '').replace(']','').replace('/','').replace('&','').strip().split()
                    for feature_index, feature_name in enumerate(feature_names):
                        for state_feature in feature_info:
                            if state_feature.startswith(feature_name):
                                if state_feature.count('=') > 0:
                                    _, value = state_feature.split('=')
                                    observations[state_index, feature_index] = int(value)
                                    break
                                else:
                                    observations[state_index, feature_index] = 1
                                    break
                            elif state_feature.startswith(f"!{feature_name}"):
                                observations[state_index, feature_index] = 0
                                break
                        else:
                            assert False, f"Feature {feature_name} not found in state {state_index}"

                j = i+2
                processing_action = 0
                while (not mdp_lines[j].startswith("state")) and (j < len(mdp_lines)):
                    processed_line = mdp_lines[j].strip()
                    if processed_line.startswith("action"):
                        action_info = processed_line.replace('[', '').replace(']','').split()
                        processing_action = action_names.index(action_info[1])
                        action_reward = action_info[-1]
                    else:
                        transition_line = processed_line.split(' : ')
                        trans_probs[state_index, int(transition_line[0]), processing_action] = float(transition_line[1])
                        rewards[state_index, int(transition_line[0]), processing_action] = float(action_reward)
                    j += 1
                    if j == len(mdp_lines):
                        break
                
        mdp = MarkovDecisionProcess(trans_probs=trans_probs, rewards=rewards, initial_state_p=initial_state, observations=observations, feature_names=feature_names,action_names=action_names, name=args.env_name, path=args.input_path)

        if args.export_valuations:
            features_json = []
            for state in range(mdp.n_states_):
                state_features = []
                for i, feature in enumerate(mdp.feature_names):
                    state_features.append([feature, int(mdp.observations[state][i])])
                features_json.append(state_features)

            json_file = open(f"{args.input_path}{args.env_name}/state_valuations.json", "w")
            json.dump(features_json, json_file)
            json_file.close()

            exit()
else:
    mdp_file = f"{args.input_path}/{args.env_name}/sketch-random-sink.drn"
    with open(mdp_file, "r") as f:
        mdp_lines = f.readlines()
        nr_states = 0
        sink_state = 0
        nr_actions = 0
        feature_names = []
        action_names = []
        pmax = True
        if args.env_name in ["firewire-3", "resource-gathering-5"]:
            pmax = False
        for i, line in enumerate(mdp_lines):
            if line.startswith("@nr_states"):
                sink_state = int(mdp_lines[i+1])
                nr_states = sink_state+1
                continue
            if line.startswith('@reward_models'):
                pmax = mdp_lines[i+1].isspace()
                continue
            if line.startswith("@nr_choices"):
                nr_actions = int(mdp_lines[i+1])//(nr_states-1)
            if line.startswith("state 0"):
                features = mdp_lines[i+1].replace('[', '').replace(']','').replace('/','').replace('&','').split()
                for feature in features:
                    if feature.startswith('!'):
                        feature = feature.replace('!', '')
                    elif feature.count('=') > 0:
                        feature = feature.split('=')[0]
                    feature_names.append(feature)
                
                j = i+2
                while (not mdp_lines[j].startswith("state")) and (j < len(mdp_lines)):
                    processed_line = mdp_lines[j].strip()
                    if processed_line.startswith("action"):
                        action_info = processed_line.replace('[', '').replace(']','').split()
                        action_names.append(action_info[1])
                    j += 1
                    if j == len(mdp_lines):
                        break

                break
            # line_w = line.strip()
            # if line_w.startswith('action'):
            #     action_info = line_w.replace('[', '').replace(']','').split()
            #     if action_info[1] not in action_names:
            #         action_names.append(action_info[1])

        # nr_actions = len(action_names)

        # action_names = ["u", "r", "d", "l"]
        # feature_names = ["picked0", "picked1", "picked2", "picked3", "picked4", "x", "y"]
        assert nr_actions == len(action_names)
        trans_probs = np.zeros((nr_states, nr_states, nr_actions))
        rewards = np.zeros((nr_states, nr_states, nr_actions))  
        initial_state = np.zeros(nr_states)
        observations = np.empty((nr_states, len(feature_names)))

        for i, line in enumerate(mdp_lines):
            if line.startswith("state"):
                target = False
                state_info = line.split()
                state_index = int(state_info[1])
                if "init" in state_info:
                    initial_state[state_index] = 1
                if "goal" in state_info:
                    target = True
                feature_info = mdp_lines[i+1].replace('[', '').replace(']','').replace('/','').split()
                for feature_index, feature_name in enumerate(feature_names):
                    for state_feature in feature_info:
                        if state_feature.startswith(feature_name):
                            if state_feature.count('=') > 0:
                                _, value = state_feature.split('=')
                                observations[state_index, feature_index] = int(value)
                            else:
                                observations[state_index, feature_index] = 1
                        elif state_feature.startswith(f"!{feature_name}"):
                            observations[state_index, feature_index] = 0
                j = i+2
                processing_action = 0
                processed_actions = []
                while (not mdp_lines[j].startswith("state")) and (j < len(mdp_lines)):
                    processed_line = mdp_lines[j].strip()
                    if processed_line.startswith("action"):
                        action_info = processed_line.replace('[', '').replace(']','').split()
                        processing_action = action_names.index(action_info[1])
                        processed_actions.append(processing_action)
                        if pmax and target:
                            action_reward = 1
                        elif not pmax and not target:
                            action_reward = 0 - float(action_info[-1])
                        else:
                            action_reward = 0
                    else:
                        if target:
                            trans_probs[state_index, sink_state, processing_action] = 1
                            rewards[state_index, sink_state, processing_action] = action_reward
                        else:
                            transition_line = processed_line.split(' : ')
                            trans_probs[state_index, int(transition_line[0]), processing_action] = float(transition_line[1])
                            rewards[state_index, int(transition_line[0]), processing_action] = action_reward
                    j += 1
                    if j == len(mdp_lines):
                        break

        for ind, action in enumerate(action_names):
            trans_probs[sink_state, sink_state, ind] = 1
            rewards[sink_state, sink_state, ind] = 0

        for feature_index, feature_name in enumerate(feature_names):
            observations[sink_state, feature_index] = 0


                
        mdp = MarkovDecisionProcess(trans_probs=trans_probs, rewards=rewards, initial_state_p=initial_state, observations=observations, feature_names=feature_names,action_names=action_names, name=args.env_name, path=args.input_path)

        # features_json = []
        # for state in range(mdp.n_states_):
        #     state_features = []
        #     for i, feature in enumerate(mdp.feature_names):
        #         state_features.append([feature, mdp.observations[state][i]])
        #     features_json.append(state_features)

        # json_file = open(f"environments/qcomp-mdp/{args.env_name}/state_valuations.json", "w")
        # json.dump(features_json, json_file)
        # json_file.close()

        
        # exit()

print("Solving...")

if args.algorithm.lower() == "omdt":
    if args.max_depth is None:
        raise ValueError("OMDT is only supposed to be run with max_depth")
    else:
        # If max_depth is given we want to fit a tree of that depth with maximal return
        from omdt.solver import OmdtSolver

        solver = OmdtSolver(
            depth=args.max_depth,
            gamma=args.gamma,
            max_iter=args.max_iter,
            delta=args.delta,
            n_cpus=args.n_cpus,
            verbose=args.verbose,
            time_limit=args.time_limit,
            output_dir=mdp_output_dir,
            seed=args.seed,
        )
        method_name = (
            f"omdt_depth_{args.max_depth}_seed_{args.seed}_timelimit_{args.time_limit}"
        )
elif args.algorithm.lower() == "dtcontrol":
    if args.max_depth is not None:
        raise ValueError("dtcontrol is only supposed to be run with max_depth")

    from dtcontrol.solver import DtControlSolver

    solver = DtControlSolver(
        output_dir=mdp_output_dir,
        verbose=args.verbose,
    )
    method_name = "dtcontrol"
elif args.algorithm.lower() == "dtcontrol-parser":

    from dtcontrol_parse.solver import DtControlSolverParser

    solver = DtControlSolverParser(
        output_dir=mdp_output_dir,
        verbose=args.verbose,
        scheduler_name=args.scheduler_name,
    )
    method_name = "dtcontrol-parser"
elif args.algorithm.lower() == "viper":
    if args.max_depth is None:
        raise ValueError("viper is only supposed to be run without max_depth")

    from viper.solver import ViperSolver

    solver = ViperSolver(
        max_depth=args.max_depth,
        output_dir=mdp_output_dir,
        verbose=args.verbose,
        random_seed=args.seed,
    )
    method_name = f"viper_depth_{args.max_depth}_seed_{args.seed}"
else:
    raise ValueError(f"Algorithm {args.algorithm} not known")

start_time = time.time()

solver.solve(mdp)

if method_name == "dtcontrol-parser":
    runtime = solver.runtime
    objective = None
else:
    runtime = time.time() - start_time
    objective = mdp.evaluate_policy(solver.act, args.gamma, 10000)

optimal = solver.optimal_
bound = solver.bound_

n_nodes = solver.tree_policy_.count_nodes()
depth = solver.tree_policy_.count_depth()

print("Writing result files...")

# Write a .dot file to visualize the learned decision tree and also
# export to PNG and PDF.
if args.export_graphviz:
    import pydot

    integer_features = np.all(
        np.isclose(mdp.observations % np.round(mdp.observations), 0), axis=0
    )

    dot_string = solver.tree_policy_.to_graphviz(
        mdp.feature_names, mdp.action_names, integer_features
    )
    graph = pydot.graph_from_dot_data(dot_string)[0]

    filename = f"{mdp_output_dir}{method_name}_visualized_policy"
    graph.write_png(f"{filename}.png")
    graph.write_pdf(f"{filename}.pdf")
    graph.write_dot(f"{filename}.dot")

result_filename = f"{args.output_dir}results.csv"

if os.path.exists(result_filename):
    write_header = False
else:
    write_header = True

# Append a new line to the result file with the results of this run.
# Optionally write a header first.
with open(result_filename, "a") as file:
    if method_name == "dtcontrol-parser":
        if write_header:
            file.write(
                "method,mdp,runtime,depth,n_nodes\n"
            )

        depth_str = args.max_depth if args.max_depth else ""
        file.write(
            f"{args.algorithm},{args.env_name},{runtime},{depth},{n_nodes}\n"
        )
    else:
        if write_header:
            file.write(
                "method,mdp,seed,time_limit,max_depth,runtime,objective,bound,depth,n_nodes,optimal\n"
            )

        depth_str = args.max_depth if args.max_depth else ""
        file.write(
            f"{args.algorithm},{args.env_name},{args.seed},{args.time_limit},{depth_str},{runtime},{objective},{bound},{depth},{n_nodes},{optimal}\n"
        )
