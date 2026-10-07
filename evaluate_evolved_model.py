import csv
import os

import numpy as np
from aeon.datasets import load_from_ts_file
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

import function_EvoTSC as F


_FUNC_NAMESPACE = {
    "FeaConH": F.FeaCon,
    "FeaCon2": F.FeaCon,
    "FeaCon3": F.FeaCon,
    "FeaCon4": F.FeaCon,
    "DomDiff_A": F.DomDiff,
    "DomDiff_B": F.DomDiff,
    "DomFreq_A": F.DomFreq,
    "DomFreq_B": F.DomFreq,
    "AdaPatch_A": F.AdaPatch,
    "AdaPatch_B": F.AdaPatch,
    "AdaPatch_C": F.AdaPatch,
    "AdaPatch_D": F.AdaPatch,
    "StatisDist": F.StatisDist,
    "ShapeFea": F.Craft_kernel_feature,
    "SegDect": F.SegDect,
}


def _tokenize(expr_str):
    tokens = []
    i = 0
    s = expr_str.strip()
    while i < len(s):
        ch = s[i]
        if ch in (" ", "\t", "\n", "\r"):
            i += 1
            continue
        if ch in ("(", ")", ","):
            tokens.append(ch)
            i += 1
            continue
        j = i
        while j < len(s) and s[j] not in ("(", ")", ",", " ", "\t", "\n", "\r"):
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def _parse_expr(tokens, pos):
    token = tokens[pos]

    if token in _FUNC_NAMESPACE:
        func = _FUNC_NAMESPACE[token]
        pos += 1
        if tokens[pos] != "(":
            raise ValueError(f"Expected '(' after {token}, got {tokens[pos]}")
        pos += 1
        args = []
        while tokens[pos] != ")":
            arg, pos = _parse_expr(tokens, pos)
            args.append(arg)
            if tokens[pos] == ",":
                pos += 1
        pos += 1

        def make_node(fn, arg_fns):
            def node(series):
                return fn(*[a(series) if callable(a) else a for a in arg_fns])
            return node

        return make_node(func, args), pos

    if token == "Series":
        return (lambda series: series), pos + 1

    try:
        return int(token), pos + 1
    except ValueError:
        pass
    try:
        return float(token), pos + 1
    except ValueError:
        pass

    raise ValueError(f"Unknown token at position {pos}: '{token}'")


def parse_expression(expr_str):
    tokens = _tokenize(expr_str)
    model, pos = _parse_expr(tokens, 0)
    if pos != len(tokens) or not callable(model):
        raise ValueError(f"Invalid expression: {expr_str}")
    return model


def extract_features(model, X):
    feats = []
    for i in range(X.shape[0]):
        feats.append(np.ravel(np.asarray(model(X[i, 0, :]))))
    return np.asarray(feats, dtype=float)


def load_models(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["dataset"]: row["expression"] for row in csv.DictReader(fh)}


def evaluate(dataset, expr_str, data_dir, seed):
    model = parse_expression(expr_str)
    data_path = os.path.join(data_dir, dataset)
    X_train, y_train = load_from_ts_file(
        os.path.join(data_path, f"{dataset}_TRAIN.ts"), return_type="numpy3D")
    X_test, y_test = load_from_ts_file(
        os.path.join(data_path, f"{dataset}_TEST.ts"), return_type="numpy3D")

    Xtr = extract_features(model, X_train)
    Xte = extract_features(model, X_test)

    et = ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                              random_state=seed, n_jobs=1)
    pipe = make_pipeline(et)
    pipe.fit(Xtr, y_train)
    return 100.0 * pipe.score(Xte, y_test), Xtr.shape[1]
