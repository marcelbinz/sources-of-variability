import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

import os


# Define the fixed parts of the dictionary
base_dict = {
    "rnd_seed": 1,
    "python_file": "train.py",
}

# , "id_nohist", "shared_hist", "shared_nohist"]
l_condition_names = ["original"]
l_d_model = [32]
l_d_ff = [256]
l_is_testcase = ["False"]
l_num_layers = [2]
l_masktype = ["causal", "windowed_causal"]
l_windowsize = [1, 5]

# Generate all combinations
combinations = list(
    itertools.product(
        l_condition_names, l_d_model, l_d_ff, l_is_testcase, l_num_layers, l_masktype, l_windowsize
    )
)

# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    condition_name, d_model, d_ff, is_testcase, num_layers, masktype, windowsize
) in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "condition_name": condition_name,
            "d_model": d_model,
            "d_ff": d_ff,
            "is_testcase": is_testcase,
            "num_layers": num_layers,
            "masktype": masktype,
            "windowsize": windowsize,
        }
    )
    arg_combinations.append(temp_dict)


# Function to run the command
def run_command(args):
    command = f" python {args['python_file']} --rnd_seed {args['rnd_seed']} \
        --d_model {args['d_model']} \
        --d_ff {args['d_ff']} \
        --is_testcase {args['is_testcase']} \
        --num_layers {args['num_layers']} \
        --masktype {args['masktype']} \
        --windowsize {args['windowsize']} \
        --condition_name {args['condition_name']}"
    subprocess.run(command, shell=True)


# serial
# for args in arg_combinations:
#     run_command(args)

# parallel
with ThreadPoolExecutor(max_workers=18) as executor:
    executor.map(run_command, arg_combinations)
