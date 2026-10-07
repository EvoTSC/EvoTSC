import os
import sys
import argparse
import random
import evalGP_fgps_paretoft as evalGP
import gp_restrict as gp_restrict
from deap import base, creator, tools, gp
from strongGPDataType_TS import VectorC,VectorCC,Series,Segment,Segmentdif,SegmentFreq,SegStart,SegSize,VL,DepthRG,KernelType,StatisRatio,BoundRatio
from sklearn.model_selection import StratifiedKFold
from aeon.datasets import load_from_ts_file
import function_EvoTSC as F
import logging
from datetime import datetime
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
import torch

import copy

population = 100
generation = 50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 6
randomSeeds = 1
dataname = "Computers"
paretoTournFrac = 0.2
nFolds = 5
fitnessTrees = 10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(SCRIPT_DIR, "data", "UCR")
resultsDir = os.path.join(SCRIPT_DIR, "results")


def _make_logger(dataname, seed):
    os.makedirs(resultsDir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logname = f"{dataname}_{seed}_{stamp}.log"
    logfile = os.path.join(resultsDir, logname)

    logger = logging.getLogger("runlog")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.FileHandler(logfile, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger, logfile

pset = gp.PrimitiveSetTyped('MAIN', [Series], VectorCC, prefix='Series')


pset.addPrimitive(F.FeaCon, [VectorC, VectorCC], VectorCC, name='FeaConH')
pset.addPrimitive(F.FeaCon, [VectorC, VectorC], VectorCC, name='FeaCon2')
pset.addPrimitive(F.FeaCon, [VectorC, VectorC, VectorC], VectorCC, name='FeaCon3')
pset.addPrimitive(F.FeaCon, [VectorC, VectorC, VectorC, VectorC], VectorCC, name='FeaCon4')


pset.addPrimitive(F.StatisDist, [VL, StatisRatio], VectorC, name="StatisDist")
pset.addPrimitive(F.Craft_kernel_feature, [VL, KernelType, BoundRatio], VectorC, name="ShapeFea")


for src_type, prefix in [
    (Series,      "A"),
    (Segment, "B"),
    (Segmentdif, "C"),
    (SegmentFreq, "D"),
]:
    pset.addPrimitive(F.AdaPatch, [src_type, DepthRG], VL, name=f"AdaPatch_{prefix}")


pset.addPrimitive(F.DomDiff, [Series],  Segmentdif, name='DomDiff_A')
pset.addPrimitive(F.DomDiff, [Segment], Segmentdif, name='DomDiff_B')
pset.addPrimitive(F.DomFreq, [Series],      SegmentFreq, name='DomFreq_A')
pset.addPrimitive(F.DomFreq, [Segment],     SegmentFreq, name='DomFreq_B')


pset.addPrimitive(F.SegDect, [Series, SegStart, SegSize], Segment, name='SegDect')


pset.renameArguments(Series0='Series')

def _SegStart():
    return random.random()
def _SegSize():
    return random.random()
pset.addEphemeralConstant('SegStart', _SegStart, SegStart)
pset.addEphemeralConstant('SegSize',  _SegSize,  SegSize)

def _DepthRG():
    return random.randint(1, 7)
pset.addEphemeralConstant('DepthRG', _DepthRG, DepthRG)

def _KernelType():
    return random.randint(1, 3)
pset.addEphemeralConstant('KernelType', _KernelType, KernelType)

def _StatisRatio():
    return random.choice([0.25, 0.5, 0.75])
pset.addEphemeralConstant('StatisRatio', _StatisRatio, StatisRatio)

def _BoundRatio():
    return random.choice([0.25, 0.5, 0.75])
pset.addEphemeralConstant('BoundRatio', _BoundRatio, BoundRatio)


creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        feats = []
        N = X_train.shape[0]
        for i in range(N):
            x_i = X_train[i, 0, :]
            f_i = np.asarray(func(x_i), dtype=float).ravel()
            feats.append(f_i)
        train_tf = np.vstack(feats)
        feat_dim = int(train_tf.shape[1])
        classes_all, counts = np.unique(y_train, return_counts=True)
        if counts.size == 0:
            raise ValueError("Empty training labels.")
        min_count = int(counts.min())
        n_splits = min(nFolds, min_count)
        if n_splits < 2:
            raise ValueError(f"Not enough samples per class for CV: min_count={min_count}")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=randomSeeds)
        accs = []
        losses = []
        fold_accs = []
        eps = 1e-15
        for tr_idx, va_idx in cv.split(train_tf, y_train):
            Xtr, Xva = train_tf[tr_idx], train_tf[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            et = ExtraTreesClassifier(
                n_estimators=fitnessTrees,
                max_features="sqrt",
                random_state=randomSeeds,
                n_jobs=1
            )
            et.fit(Xtr, ytr)
            y_pred = et.predict(Xva)
            _acc = float(np.mean(y_pred == yva))
            accs.append(_acc)
            fold_accs.append(100.0 * _acc)
            proba = et.predict_proba(Xva)
            seen_classes = et.classes_
            full_proba = np.zeros((len(yva), len(classes_all)), dtype=float)
            col_map = {c: j for j, c in enumerate(classes_all)}
            for j, c in enumerate(seen_classes):
                full_proba[:, col_map[c]] = proba[:, j]
            full_proba = np.clip(full_proba, eps, 1.0 - eps)
            full_proba /= full_proba.sum(axis=1, keepdims=True)
            losses.append(float(log_loss(yva, full_proba, labels=classes_all)))
        accuracy = 100.0 * float(np.mean(accs))
        cv_logloss = float(np.mean(losses))
        individual.fold_accs = tuple(fold_accs)
    except Exception:
        accuracy = 0.0
        cv_logloss = 1e9
        feat_dim = 10 ** 9
        individual.fold_accs = tuple()
    return accuracy, cv_logloss, feat_dim,


def _build_toolbox():
    global toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset,
                     min_=initialMinDepth, max_=initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("select", evalGP.selParetoTournament, tourn_frac=paretoTournFrac)
    toolbox.register("mate", evalGP.safe_cx_one_point, max_depth=maxDepth)
    toolbox.register("mutate", evalGP.safe_mut_uniform, pset=pset, max_depth=maxDepth, min_=0)


def _transform(func, X):
    feats = []
    for i in range(X.shape[0]):
        feats.append(np.ravel(np.asarray(func(X[i, 0, :]))))
    return np.asarray(feats, dtype=float)


def _test_accuracy(ind):
    test_path = os.path.join(dataDir, dataname, f"{dataname}_TEST.ts")
    X_test, y_test = load_from_ts_file(test_path, return_type="numpy3D")
    func = toolbox.compile(expr=ind)
    et = ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                              random_state=randomSeeds, n_jobs=1)
    et.fit(_transform(func, X_train), y_train)
    return 100.0 * et.score(_transform(func, X_test), y_test)


def GPMain(randomSeeds):
    random.seed(randomSeeds)
    np.random.seed(randomSeeds)
    torch.manual_seed(randomSeeds)
    global X_train, y_train
    data_root = dataDir
    data_dir  = os.path.join(data_root, dataname)
    train_path = os.path.join(data_dir, f"{dataname}_TRAIN.ts")
    X_train, y_train = load_from_ts_file(train_path, return_type="numpy3D")
    _build_toolbox()
    toolbox.register("evaluate", evalTrain)
    logger, logfile = _make_logger(dataname, randomSeeds)
    hparams = {
        "dataname": dataname,
        "population": population,
        "generation": generation,
        "cxProb": cxProb,
        "mutProb": mutProb,
        "elitismProb": elitismProb,
        "initialMinDepth": initialMinDepth,
        "initialMaxDepth": initialMaxDepth,
        "maxDepth": maxDepth,
        "randomSeeds": randomSeeds,
        "paretoTournFrac": paretoTournFrac,
    }
    logger.info("# === Parameters ===")
    for k, v in hparams.items():
        logger.info(f"{k}: {v}")
    logger.info("")
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)

    evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, halloffame=hof)

    best = hof[0]
    test_acc = _test_accuracy(best)
    logger.info("# === Best Individual ===")
    logger.info(f"expression: {best}")
    logger.info(f"nodes: {len(best)}")
    logger.info(f"height: {best.height}")
    logger.info(f"test accuracy: {test_acc:.4f}%")
    return best

def _parse_args():
    p = argparse.ArgumentParser(description="Run EvoTSC on one UCR dataset with one random seed.")
    p.add_argument("--dataset", default=dataname)
    p.add_argument("--seed", type=int, default=randomSeeds)
    p.add_argument("--population", type=int, default=population)
    p.add_argument("--generation", type=int, default=generation)
    p.add_argument("--cx_prob", type=float, default=cxProb)
    p.add_argument("--mut_prob", type=float, default=mutProb)
    p.add_argument("--elitism_prob", type=float, default=elitismProb)
    p.add_argument("--init_min_depth", type=int, default=initialMinDepth)
    p.add_argument("--init_max_depth", type=int, default=initialMaxDepth)
    p.add_argument("--max_depth", type=int, default=maxDepth)
    p.add_argument("--tourn_ratio", type=float, default=paretoTournFrac,
                   help="tournament ratio mu of the Pareto tournament selection")
    p.add_argument("--n_folds", type=int, default=nFolds)
    p.add_argument("--fitness_trees", type=int, default=fitnessTrees)
    p.add_argument("--data_dir", default=dataDir)
    p.add_argument("--results_dir", default=resultsDir)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataname = args.dataset
    randomSeeds = args.seed
    population = args.population
    generation = args.generation
    cxProb = args.cx_prob
    mutProb = args.mut_prob
    elitismProb = args.elitism_prob
    initialMinDepth = args.init_min_depth
    initialMaxDepth = args.init_max_depth
    maxDepth = args.max_depth
    paretoTournFrac = args.tourn_ratio
    nFolds = args.n_folds
    fitnessTrees = args.fitness_trees
    dataDir = args.data_dir
    resultsDir = args.results_dir
    GPMain(randomSeeds)
