import os

import evaluate_evolved_model as E

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Put the UCR datasets in .ts format here: data/UCR/<dataset>/<dataset>_TRAIN.ts and data/UCR/<dataset>/<dataset>_TEST.ts
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "UCR")
MODELS_FILE = os.path.join(SCRIPT_DIR, "trained_models.csv")

DATASET = "Computers"
EXPRESSION = None
SEED = 21


def main():
    expression = EXPRESSION if EXPRESSION else E.load_models(MODELS_FILE)[DATASET]
    acc, n_features = E.evaluate(DATASET, expression, DATA_DIR, SEED)
    print(f"dataset: {DATASET}")
    print(f"expression: {expression}")
    print(f"features: {n_features}")
    print(f"test accuracy (ET300, seed {SEED}): {acc:.4f}%")


if __name__ == "__main__":
    main()
