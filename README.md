# EvoTSC

This repository contains the code of the paper "EvoTSC: Evolving Feature Learning Models for Time Series Classification via Genetic Programming".

- `run_EvoTSC.py`: runs EvoTSC to evolve a feature learning model on a dataset.
- `run_evolved_model.py`: applies an evolved model to a dataset and reports its test accuracy.
- `trained_models.csv`: the evolved feature learning model (best individual) of each dataset.

The required packages are listed in `requirements.txt`.

## Datasets

The datasets are from the UCR Time Series Classification Archive and are used with their official train/test splits. They are placed in `data/UCR/<dataset>/<dataset>_TRAIN.ts` and `data/UCR/<dataset>/<dataset>_TEST.ts`. The Computers dataset is included as an example.