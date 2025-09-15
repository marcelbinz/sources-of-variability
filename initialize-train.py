import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

import os


# Define the fixed parts of the dictionary
base_dict = {
    "rnd_seed": 1,
    "python_file": "train.py",
}


l_d_model = [32, 64]
l_d_ff = [128, 256]
# , "random_weights_random_scaling"
l_is_testcase = [True]  # ,"testcase"

# Generate all combinations
combinations = list(
    itertools.product(
        l_d_model, l_d_ff, l_is_testcase
    )
)

# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    d_model, d_ff, is_testcase,
) in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "d_model": d_model,
            "d_ff": d_ff,
            "is_testcase": is_testcase,
        }
    )
    arg_combinations.append(temp_dict)


# Function to run the command


def run_command(args):
    command = f" python {args['python_file']} --rnd_seed {args['rnd_seed']} \
        --d_model {args['d_model']} \
        --d_ff {args['d_ff']} \
        --is_testcase {args['is_testcase']}"
    subprocess.run(command, shell=True)


# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_command, arg_combinations)
