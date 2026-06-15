import subprocess
from concurrent.futures import ThreadPoolExecutor

from itertools import product, chain


# Define the fixed parts of the dictionary
base_dict = {
    "rnd_seed": 1,
    "python_file": "train.py",
    "dataset_name": "mm",  # or "itc"
    "num_epochs": 250
}
l_dataset_select = [
    "med_seq",
    #"culture_seq_1000",
]  # NB short_seq includes most data bc more participants who completed fewer trials
l_indep_vars = ["few_culture"] # , "all_culture""all" uses all indep vars, "few" only to test for age, culture and age x culture effect
l_condition_names = [
    "id_nohist",
    "shared_nohist",
]  # [,], no proper history available for culture sequences, because they are merged from different people

# swap_colnames and shuffle_single_colnames are only needed when variables within a condition are shuffled
# i.e., besides individual differences and trial history
# as not all fully crossed, get combinations in two steps

# for itc data:
swap_colnames = [
    '[[\"left_oldman\", \"right_oldman\"], [\"left_oldwoman\", \"right_oldwoman\"], [\"left_man\", \"right_man\"], [\"left_woman\", \"right_woman\"], [\"left_boy\", \"right_boy\"], [\"left_girl\", \"right_girl\"]]',
    "[[]]"
    ]
shuffle_single_colnames = ['[\"Eastern\", \"Southern\"]', '[]']


l_d_model = [4] # 8 #
l_d_ff = [8] # 16 #
l_is_testcase = ["False"]
l_num_layers = [2]
l_masktype = ["causal"]  # "windowed_causal"
l_windowsize = [2]  # only relevant for windowed_causal, ignored when "causal" # [7, 10]


# second part can be dropped when only generated data sets (i.e., four conditions) are used
# Generate all combinations
combinations = list(
    chain(
        product(
            l_dataset_select,
            l_condition_names,
            l_indep_vars,
            swap_colnames,
            shuffle_single_colnames,
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
    dataset_select,
    condition_name,
    indep_vars,
    swap_colnames,
    shuffle_single_colnames,
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
            "dataset_select": dataset_select,
            "condition_name": condition_name,
            "indep_vars": indep_vars,
            "swap_colnames": swap_colnames,
            "shuffle_single_colnames": shuffle_single_colnames,
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
        "--dataset_name",
        args["dataset_name"],
        "--dataset_select",
        args["dataset_select"],
        "--indep_vars",
        args["indep_vars"],
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
# for args in arg_combinations:
#     run_command(args)

# parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(run_command, arg_combinations)
