import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Put the UCR datasets in .ts format here: data/UCR/<dataset>/<dataset>_TRAIN.ts and data/UCR/<dataset>/<dataset>_TEST.ts
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "UCR")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DATASETS = ["Computers"]
SEEDS = list(range(1, 31))

POPULATION = 100
GENERATION = 50
CX_PROB = 0.8
MUT_PROB = 0.19
ELITISM_PROB = 0.01
INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
MAX_DEPTH = 6
TOURN_RATIO = 0.2

N_FOLDS = 5
FITNESS_TREES = 10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for dataset in DATASETS:
        for seed in SEEDS:
            subprocess.run([
                sys.executable, "EvoTSC.py",
                "--dataset", dataset,
                "--seed", str(seed),
                "--population", str(POPULATION),
                "--generation", str(GENERATION),
                "--cx_prob", str(CX_PROB),
                "--mut_prob", str(MUT_PROB),
                "--elitism_prob", str(ELITISM_PROB),
                "--init_min_depth", str(INIT_MIN_DEPTH),
                "--init_max_depth", str(INIT_MAX_DEPTH),
                "--max_depth", str(MAX_DEPTH),
                "--tourn_ratio", str(TOURN_RATIO),
                "--n_folds", str(N_FOLDS),
                "--fitness_trees", str(FITNESS_TREES),
                "--data_dir", DATA_DIR,
                "--results_dir", RESULTS_DIR,
            ], cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
