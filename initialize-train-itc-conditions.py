import subprocess
from concurrent.futures import ThreadPoolExecutor

from itertools import product, chain


import os


# Define the fixed parts of the dictionary
base_dict = {
    "rnd_seed": 2,
    "python_file": "train.py",
    "dataset_name": "itc",  # or "risky"
    "tf": "log_values",
    "num_epochs": 250,
}

l_condition_names = ["original", "shared_nohist", "shared_hist", "id_nohist"]  # [,]


# swap_colnames and shuffle_single_colnames are only needed when variables within a condition are shuffled
# i.e., besides individual differences and trial history
# as not all fully crossed, get combinations in two steps

# for itc data:
# swap_colnames = ['[[\"right_val\", \"left_val\"], [\"right_time\", \"left_time\"]]']
# shuffle_single_colnames = ['[\"right_picked_prev\"]', '[]']
swap_colnames_2 = [
    "[[]]"
]  # ,'[[\"right_val\", \"left_val\"]]', '[[\"right_time\", \"left_time\"]]'
shuffle_single_colnames_2 = ["[]"]

# for 10 train trials: d_model: 16, d_ff: 128
# for 38 train trials: d_model: 32, d_ff: 256
# for 130 train trials: d_model: 32, d_ff: 256


# larger model dims for med and full select
l_dataset_select = ["full"]
l_d_model = [32]
l_d_ff = [256]
l_is_testcase = ["False"]
l_num_layers = [2]
l_masktype = ["causal"]  # "windowed_causal"
l_windowsize = [2]  # only relevant for windowed_causal, ignored when "causal" # [7, 10]

combinations = list(
    chain(
        product(
            l_condition_names,
            swap_colnames_2,
            shuffle_single_colnames_2,
            l_dataset_select,
            l_d_model,
            l_d_ff,
            l_is_testcase,
            l_num_layers,
            l_masktype,
            l_windowsize,
        )
    )
)


# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    condition_name,
    swap_colnames,
    shuffle_single_colnames,
    dataset_select,
    d_model,
    d_ff,
    is_testcase,
    num_layers,
    masktype,
    windowsize,
) in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "condition_name": condition_name,
            "swap_colnames": swap_colnames,
            "shuffle_single_colnames": shuffle_single_colnames,
            "dataset_select": dataset_select,
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
        "--tf",
        args["tf"],
        "--dataset_name",
        args["dataset_name"],
        "--dataset_select",
        args["dataset_select"],
        "--rnd_seed",
        str(args["rnd_seed"]),
        "--d_model",
        str(args["d_model"]),
        "--d_ff",
        str(args["d_ff"]),
        "--is_testcase",
        str(args["is_testcase"]),
        "--num_layers",
        str(args["num_layers"]),
        "--masktype",
        str(args["masktype"]),
        "--windowsize",
        str(args["windowsize"]),
        "--condition_name",
        args["condition_name"],
        "--swap_colnames",
        args["swap_colnames"],
        "--shuffle_single_colnames",
        args["shuffle_single_colnames"],
        "--num_epochs",
        str(args["num_epochs"]),
    ]
    subprocess.run(command)


# serial
for args in arg_combinations:
    run_command(args)

# parallel
# with ThreadPoolExecutor(max_workers=20) as executor:
#     executor.map(run_command, arg_combinations)
