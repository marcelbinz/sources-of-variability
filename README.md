# Understanding Behavior Through Permutation-Based Predictive Modeling

- Download the whole osf repo and unzip

## Instructions to reproduce the python results

### General Setup

_Conda and Datasets_

- install miniconda on your machine (https://www.anaconda.com/docs/getting-started/miniconda/main), or if you prefer, anaconda
- open a command line window and activate the "base" environment: `conda activate base`
- on the command line, run: `python setup.py`. This will unzip the moral machine data and create the SoV conda environment
- activate the created SoV conda environment on the command line: `conda activate SoV`

  _Wandb_

- create a wandb account and create a new API key (https://docs.wandb.ai/models/quickstart)
- on the command line, set the WANDB_API_KEY environment variable: `export WANDB_API_KEY=<your_api_key>`

### Computational Modeling: python scripts

- initialize all models by running `python run_initializers.py` on the command line. This will sequentially execute all python scripts and save the modeling results on your wandb account
- if you prefer running an individual model, just run `python initialize-train-...py`. Note that **itc** stands for the money-earlier-or-later (MEL) task, and **mm** stands for the ethical choice task in the moral-machine experiment. **conditions** refers to Level 1 shuffling, **variables** to Level 2 shuffling, **masklength** to the causal mask analysis, **conditions-cultureseq** to the additional analysis merging sequences from participants from the same culture in the ethical choice task, and **conditions-vary-n** for the additional analysis varying the data set size in the MEL task.

### Additional Analyses: jupyter notebooks

- open jupyter notebook (`jupyter notebook` on the commandline when the SoV conda environment is activated)
- to load all results from wandb, navigate to the notebook named ~/load-wandb-runs.ipynb and run all cells from beginning to end
- to analyze the proportion of participants who change their preference when delivery time is changed and who change their preferences when money is changed, navigate to the notebook named ~/itc-prop-switches.ipynb. Running all cells will create two plots saved under ~/figures/: proportion of switches against time differences and proportion of switches against value differences for selected participants. The file will additionally save a csv with path ~/data/itc-processed.csv
- to run the by-participant logistic regressions in the ethical choice task, and plot the culture cluster group means with confidence intervals, run the jupyter notebook ~/mm-group-differences-stats.ipynb. This notebook will save the plot ~/figures/variability-age-means.pdf

## Instrutions to reproduce the R results

- to generate the overview result plots for each task, run ~/R/plot-model-results.R. Note that the model results have first to be loaded by running the jupyter notebook under ~/load-wandb-runs.ipynb (see above).
- to run the heuristic cognitive models for the MEL task, run ~/R/hierarchical-logit.R. Note that this script requires ~/data/itc-processed.csv, which is written in the jupyter notebook ~/itc-prop-switches.ipynb (see above)
