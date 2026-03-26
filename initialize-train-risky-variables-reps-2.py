import subprocess
from concurrent.futures import ThreadPoolExecutor

from itertools import product, chain


import os


# Define the fixed parts of the dictionary
base_dict = {
    "rnd_seed": 1,
    "python_file": "train.py",
    "dataset_name": "risky", # or "itc"
    "dataset_select": "repetitions", # is ignored for itc data
}

l_condition_names = ["shared_hist", "id_nohist"] # [,]"original", "shared_nohist", 


# swap_colnames and shuffle_single_colnames are only needed when variables within a condition are shuffled
# i.e., besides individual differences and trial history
# as not all fully crossed, get combinations in two steps

# for itc data:
# for risky data:
# bit ugly for risky data, because many columns to be swapped for one variable (e.g., value)
swap_colnames = [
    '[[\"right_1_val\", \"left_1_val\"], [\"right_2_val\", \"left_2_val\"], \
    [\"right_3_val\", \"left_3_val\"], [\"right_4_val\", \"left_4_val\"], \
    [\"right_5_val\", \"left_5_val\"], [\"right_6_val\", \"left_6_val\"], \
    [\"right_7_val\", \"left_7_val\"], [\"right_8_val\", \"left_8_val\"], \
    [\"right_9_val\", \"left_9_val\"], [\"right_1_prob\", \"left_1_prob\"], \
    [\"right_2_prob\", \"left_2_prob\"], [\"right_3_prob\", \"left_3_prob\"], \
    [\"right_4_prob\", \"left_4_prob\"], [\"right_5_prob\", \"left_5_prob\"], \
    [\"right_6_prob\", \"left_6_prob\"], [\"right_7_prob\", \"left_7_prob\"], \
    [\"right_8_prob\", \"left_8_prob\"], [\"right_9_prob\", \"left_9_prob\"]]'
]
shuffle_single_colnames = ['[\"right_picked_prev\"]', '[]']

swap_colnames_2 = [
    '[[]]',

    '[[\"right_1_val\", \"left_1_val\"], [\"right_2_val\", \"left_2_val\"], \
    [\"right_3_val\", \"left_3_val\"], [\"right_4_val\", \"left_4_val\"], \
    [\"right_5_val\", \"left_5_val\"], [\"right_6_val\", \"left_6_val\"], \
    [\"right_7_val\", \"left_7_val\"], [\"right_8_val\", \"left_8_val\"], \
    [\"right_9_val\", \"left_9_val\"]]',

    '[[\"right_1_prob\", \"left_1_prob\"], [\"right_2_prob\", \"left_2_prob\"], \
    [\"right_3_prob\", \"left_3_prob\"], [\"right_4_prob\", \"left_4_prob\"], \
    [\"right_5_prob\", \"left_5_prob\"], [\"right_6_prob\", \"left_6_prob\"], \
    [\"right_7_prob\", \"left_7_prob\"], [\"right_8_prob\", \"left_8_prob\"], \
    [\"right_9_prob\", \"left_9_prob\"]]'
]
shuffle_single_colnames_2 = ['[]']

l_d_model = [16]
l_d_ff = [64]
l_is_testcase = ["False"]
l_num_layers = [2]
l_masktype = ["causal"] # "windowed_causal"
l_windowsize = [2]  # only relevant for windowed_causal, ignored when "causal" # 1, 2, 3, 5, 10


# second part can be dropped when only generated data sets (i.e., four conditions) are used
# Generate all combinations
combinations = list(chain(
    product(
        l_condition_names, swap_colnames, shuffle_single_colnames,
        l_d_model, l_d_ff, l_is_testcase, l_num_layers, l_masktype, l_windowsize
    ),
    product(
        l_condition_names, swap_colnames_2, shuffle_single_colnames_2,
        l_d_model, l_d_ff, l_is_testcase, l_num_layers, l_masktype, l_windowsize
    )
))

# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    condition_name, swap_colnames, shuffle_single_colnames, d_model, d_ff, is_testcase, num_layers, masktype, windowsize
) in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "condition_name": condition_name,
            "swap_colnames":swap_colnames, 
            "shuffle_single_colnames":shuffle_single_colnames,
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
    command = [
        "python",
        args["python_file"],
        "--dataset_name", args["dataset_name"],
        "--dataset_select", args["dataset_select"],
        "--rnd_seed", str(args["rnd_seed"]),
        "--d_model", str(args["d_model"]),
        "--d_ff", str(args["d_ff"]),
        "--is_testcase", str(args["is_testcase"]),
        "--num_layers", str(args["num_layers"]),
        "--masktype", str(args["masktype"]),
        "--windowsize", str(args["windowsize"]),
        "--condition_name", args["condition_name"],
        "--swap_colnames", args["swap_colnames"],
        "--shuffle_single_colnames", args["shuffle_single_colnames"],
    ]
    subprocess.run(command)



# serial
# for args in arg_combinations:
#     run_command(args)

# parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(run_command, arg_combinations)



# best hyperparameters for itc data



# best hyperparameters for risky no repetitions data
# l_d_model = [32]
# l_d_ff = [128]
# l_is_testcase = ["False"]
# l_num_layers = [1]
# l_masktype = ["causal"] # "windowed_causal"
# l_windowsize = [2] 

# best hyperparameters for risky peterson selected data
# l_d_model = [32]
# l_d_ff = [64]
# l_is_testcase = ["False"]
# l_num_layers = [2]
# l_masktype = ["causal"] # "windowed_causal"
# l_windowsize = [2]

# best hyperparameters for risky peterson selected data with windowed causal mask
# l_d_model = [32]
# l_d_ff = [128]
# l_is_testcase = ["False"]
# l_num_layers = [1]
# l_masktype = ["windowed_causal"] # "causal"
# l_windowsize = [1, 2, 5, 10]  # only relevant for windowed_causal, ignored when "causal" # [7, 10]
