#!/usr/bin/env bash
mkdir -p experiments/out-20min
for x in models/* ; do 
    folder_name=$(basename $x)
    mkdir -p experiments/out-20min/$folder_name
done
(python run-experiment.py omdt 3d_navigation --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/3d_navigation/log-depth-1.log
(python run-experiment.py omdt 3d_navigation --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/3d_navigation/log-depth-2.log
(python run-experiment.py omdt 3d_navigation --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/3d_navigation/log-depth-3.log
(python run-experiment.py omdt 3d_navigation --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/3d_navigation/log-depth-4.log
(python run-experiment.py omdt 3d_navigation --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/3d_navigation/log-depth-5.log
(python run-experiment.py omdt 3d_navigation --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/3d_navigation/log-depth-6.log
(python run-experiment.py omdt blackjack --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/blackjack/log-depth-1.log
(python run-experiment.py omdt blackjack --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/blackjack/log-depth-2.log
(python run-experiment.py omdt blackjack --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/blackjack/log-depth-3.log
(python run-experiment.py omdt blackjack --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/blackjack/log-depth-4.log
(python run-experiment.py omdt blackjack --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/blackjack/log-depth-5.log
(python run-experiment.py omdt blackjack --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/blackjack/log-depth-6.log
(python run-experiment.py omdt frozenlake_4x4 --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_4x4/log-depth-1.log
(python run-experiment.py omdt frozenlake_4x4 --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_4x4/log-depth-2.log
(python run-experiment.py omdt frozenlake_4x4 --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_4x4/log-depth-3.log
(python run-experiment.py omdt frozenlake_4x4 --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_4x4/log-depth-4.log
(python run-experiment.py omdt frozenlake_4x4 --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_4x4/log-depth-5.log
(python run-experiment.py omdt frozenlake_4x4 --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_4x4/log-depth-6.log
(python run-experiment.py omdt frozenlake_8x8 --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_8x8/log-depth-1.log
(python run-experiment.py omdt frozenlake_8x8 --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_8x8/log-depth-2.log
(python run-experiment.py omdt frozenlake_8x8 --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_8x8/log-depth-3.log
(python run-experiment.py omdt frozenlake_8x8 --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_8x8/log-depth-4.log
(python run-experiment.py omdt frozenlake_8x8 --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_8x8/log-depth-5.log
(python run-experiment.py omdt frozenlake_8x8 --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_8x8/log-depth-6.log
(python run-experiment.py omdt frozenlake_12x12 --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_12x12/log-depth-1.log
(python run-experiment.py omdt frozenlake_12x12 --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_12x12/log-depth-2.log
(python run-experiment.py omdt frozenlake_12x12 --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_12x12/log-depth-3.log
(python run-experiment.py omdt frozenlake_12x12 --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_12x12/log-depth-4.log
(python run-experiment.py omdt frozenlake_12x12 --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_12x12/log-depth-5.log
(python run-experiment.py omdt frozenlake_12x12 --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/frozenlake_12x12/log-depth-6.log
(python run-experiment.py omdt maze-7 --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/maze-7/log-depth-1.log
(python run-experiment.py omdt maze-7 --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/maze-7/log-depth-2.log
(python run-experiment.py omdt maze-7 --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/maze-7/log-depth-3.log
(python run-experiment.py omdt maze-7 --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/maze-7/log-depth-4.log
(python run-experiment.py omdt maze-7 --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/maze-7/log-depth-5.log
(python run-experiment.py omdt maze-7 --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/maze-7/log-depth-6.log
(python run-experiment.py omdt inventory_management --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/inventory_management/log-depth-1.log
(python run-experiment.py omdt inventory_management --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/inventory_management/log-depth-2.log
(python run-experiment.py omdt inventory_management --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/inventory_management/log-depth-3.log
(python run-experiment.py omdt inventory_management --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/inventory_management/log-depth-4.log
(python run-experiment.py omdt inventory_management --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/inventory_management/log-depth-5.log
(python run-experiment.py omdt inventory_management --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/inventory_management/log-depth-6.log
(python run-experiment.py omdt system_administrator_1 --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_1/log-depth-1.log
(python run-experiment.py omdt system_administrator_1 --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_1/log-depth-2.log
(python run-experiment.py omdt system_administrator_1 --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_1/log-depth-3.log
(python run-experiment.py omdt system_administrator_1 --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_1/log-depth-4.log
(python run-experiment.py omdt system_administrator_1 --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_1/log-depth-5.log
(python run-experiment.py omdt system_administrator_1 --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_1/log-depth-6.log
(python run-experiment.py omdt system_administrator_2 --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_2/log-depth-1.log
(python run-experiment.py omdt system_administrator_2 --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_2/log-depth-2.log
(python run-experiment.py omdt system_administrator_2 --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_2/log-depth-3.log
(python run-experiment.py omdt system_administrator_2 --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_2/log-depth-4.log
(python run-experiment.py omdt system_administrator_2 --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_2/log-depth-5.log
(python run-experiment.py omdt system_administrator_2 --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_2/log-depth-6.log
(python run-experiment.py omdt system_administrator_tree --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_tree/log-depth-1.log
(python run-experiment.py omdt system_administrator_tree --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_tree/log-depth-2.log
(python run-experiment.py omdt system_administrator_tree --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_tree/log-depth-3.log
(python run-experiment.py omdt system_administrator_tree --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_tree/log-depth-4.log
(python run-experiment.py omdt system_administrator_tree --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_tree/log-depth-5.log
(python run-experiment.py omdt system_administrator_tree --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/system_administrator_tree/log-depth-6.log
(python run-experiment.py omdt tictactoe_vs_random --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/tictactoe_vs_random/log-depth-1.log
(python run-experiment.py omdt tictactoe_vs_random --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/tictactoe_vs_random/log-depth-2.log
(python run-experiment.py omdt tictactoe_vs_random --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/tictactoe_vs_random/log-depth-3.log
(python run-experiment.py omdt tictactoe_vs_random --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20minout-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/tictactoe_vs_random/log-depth-4.log
(python run-experiment.py omdt tictactoe_vs_random --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/tictactoe_vs_random/log-depth-5.log
(python run-experiment.py omdt tictactoe_vs_random --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/tictactoe_vs_random/log-depth-6.log
(python run-experiment.py omdt traffic_intersection --seed 0 --max_depth 1 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/traffic_intersection/log-depth-1.log
(python run-experiment.py omdt traffic_intersection --seed 0 --max_depth 2 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/traffic_intersection/log-depth-2.log
(python run-experiment.py omdt traffic_intersection --seed 0 --max_depth 3 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/traffic_intersection/log-depth-3.log
(python run-experiment.py omdt traffic_intersection --seed 0 --max_depth 4 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/traffic_intersection/log-depth-4.log
(python run-experiment.py omdt traffic_intersection --seed 0 --max_depth 5 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/traffic_intersection/log-depth-5.log
(python run-experiment.py omdt traffic_intersection --seed 0 --max_depth 6 --time_limit 1200 --output_dir experiments/out-20min/ --verbose 1 --model-file-name model-random.drn) > experiments/out-20min/traffic_intersection/log-depth-6.log