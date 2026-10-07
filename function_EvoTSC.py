from __future__ import annotations
import numpy as np
import torch


def FeaCon(*args) -> np.ndarray:
    arrs = []
    for a in args:
        a = np.asarray(a)
        if a.ndim == 0:
            a = a[None]
        arrs.append(a.ravel())
    return np.concatenate(arrs, axis=0)


def SegDect(x: np.ndarray, start_r: float, size_r: float) -> np.ndarray:

    x = np.ravel(x)
    L = x.size
    if L < 8:
        raise ValueError(f"Sequence too short for segment >=9 and <L (L={L}).")
    size_min, size_max = 9, L - 1
    size = size_min + int(np.floor(float(size_r) * (size_max - size_min + 1)))
    size = min(max(size, size_min), size_max)

    start_max = L - size
    start = int(np.floor(float(start_r) * (start_max + 1)))
    start = min(max(start, 0), start_max)

    return x[start : start + size]


def DomDiff(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return x
    return np.diff(x, n=1)


def DomFreq(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    with torch.no_grad():
        t = torch.from_numpy(x)
        X = torch.fft.rfft(t)
        mag = torch.abs(X)
    return mag.numpy()


def AdaPatch(x: np.ndarray, depth: int) -> list[np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    L = x.size

    if L < 2:
        return [x]

    max_depth = int(np.floor(np.log2(L)))
    depth = min(int(depth), max_depth)

    if depth <= 1:
        return [x]

    k = 2 ** (depth - 1)
    P = L // k
    S = P // 2

    if P < 2 or S < 1:
        return [x]

    n_patches = (L - P) // S + 1
    level_seqs = []

    for i in range(n_patches):
        start = i * S
        end = start + P
        if end <= L:
            level_seqs.append(x[start:end])

    return level_seqs if level_seqs else [x]


def _Stastis_1d(x: np.ndarray, ratio: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 1:
        return x.copy()

    r = float(ratio)
    if r <= 0.0:
        num_points = 1
    else:
        r = min(r, 1.0)
        num_points = int(np.ceil(n * r))
        num_points = max(1, min(num_points, n))

    if num_points == 1:
        return np.array([np.median(x)], dtype=float)

    qs = np.linspace(0.0, 1.0, num_points)
    qv = np.quantile(x, qs)
    mean_x = x.mean()
    qv[1::2] = qv[1::2] - mean_x
    return qv.astype(float)


def StatisDist(x_segments, ratio: float = 0.5) -> np.ndarray:
    segments: list[np.ndarray] = []
    if isinstance(x_segments, (list, tuple)):
        for seg in x_segments:
            seg_arr = np.asarray(seg, dtype=float).ravel()
            if seg_arr.size > 0:
                segments.append(seg_arr)
    else:
        seg_arr = np.asarray(x_segments, dtype=float).ravel()
        if seg_arr.size > 0:
            segments.append(seg_arr)

    if not segments:
        return np.zeros(1, dtype=float)

    feats = [_Stastis_1d(seg_arr, ratio=ratio) for seg_arr in segments]
    return np.concatenate(feats, axis=0)


_HANDCRAFT_KERNEL_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _build_increasing_kernel(k: int) -> np.ndarray:
    if k % 2 != 0:
        raise ValueError(f"Increasing kernel length must be even, got {k}")
    w = np.ones(k, dtype=float)
    idx = np.arange(k)
    w[idx % 2 == 0] *= -1.0
    return w


def _build_decreasing_kernel(k: int) -> np.ndarray:
    if k % 2 != 0:
        raise ValueError(f"Decreasing kernel length must be even, got {k}")
    w = np.ones(k, dtype=float)
    idx = np.arange(k)
    w[idx % 2 == 1] *= -1.0
    return w


def _build_peak_kernel(f: int) -> np.ndarray:
    if f % 2 != 0:
        raise ValueError(f"Peak base length must be even, got {f}")
    if f == 2:
        return np.array([-1.0, 2.0, -1.0], dtype=float)

    length = f + f // 2
    w = np.zeros(length, dtype=float)
    q = f // 4
    if q == 0:
        return np.array([-1.0, 2.0, -1.0], dtype=float)

    xmesh = np.linspace(0.0, 1.0, q + 1, dtype=float)[1:]
    left = xmesh**2
    right = left[::-1]

    w[0:q] = -left
    w[q : f // 2] = -right
    w[f // 2 : 3 * f // 4] = 2.0 * left
    w[3 * f // 4 : f] = 2.0 * right
    w[f : f + q] = -left
    w[f + q : f + 2 * q] = -right
    return w


def _get_kernel(kernel_type: str, base_len: int) -> np.ndarray:
    key = (kernel_type, base_len)
    if key in _HANDCRAFT_KERNEL_CACHE:
        return _HANDCRAFT_KERNEL_CACHE[key]

    if kernel_type == "inc":
        w = _build_increasing_kernel(base_len)
    elif kernel_type == "dec":
        w = _build_decreasing_kernel(base_len)
    elif kernel_type == "peak":
        w = _build_peak_kernel(base_len)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    w = w.astype(float)
    _HANDCRAFT_KERNEL_CACHE[key] = w
    return w


def _conv1d_valid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if x.size < w.size:
        return np.array([], dtype=float)
    return np.correlate(x, w, mode="valid").astype(float)


def _compute_pools(conv_out: np.ndarray) -> tuple[float, float, float]:
    conv_out = np.asarray(conv_out, dtype=float).ravel()
    N = conv_out.size
    if N == 0:
        return 0.0, 0.0, 0.0
    ppv = float((conv_out > 0.0).sum()) / float(N)
    max_pool = float(conv_out.max())
    mean_pool = float(conv_out.mean())
    return ppv, max_pool, mean_pool


def Craft_kernel_feature(x_segments, kernel_id: int = 1, bound_ratio: float = 0.5) -> np.ndarray:
    segments: list[np.ndarray] = []
    if isinstance(x_segments, (list, tuple)):
        for s in x_segments:
            arr = np.asarray(s, dtype=float).ravel()
            if arr.size > 0:
                segments.append(arr)
    else:
        arr = np.asarray(x_segments, dtype=float).ravel()
        if arr.size > 0:
            segments = [arr]

    if not segments:
        return np.zeros(1, dtype=float)

    id2type = {1: "inc", 2: "dec", 3: "peak"}
    if kernel_id not in id2type:
        raise ValueError(f"kernel_id must be in {list(id2type.keys())}, got {kernel_id}")
    k_type = id2type[kernel_id]

    all_features: list[np.ndarray] = []
    for seg in segments:
        seg = np.asarray(seg, dtype=float).ravel()
        L = seg.size

        mean_v = seg.mean()
        std_v = seg.std()
        if std_v > 1e-8:
            seg_norm = (seg - mean_v) / std_v
        else:
            seg_norm = np.zeros_like(seg)

        base_lengths = []
        b = 2
        while b < L * float(bound_ratio):
            base_lengths.append(b)
            b *= 2

        feat_list: list[float] = []
        for base_len in base_lengths:
            w = _get_kernel(k_type, base_len)
            K = w.size
            if 2 * K >= L:
                continue
            conv_out = _conv1d_valid(seg_norm, w)
            if conv_out.size == 0:
                continue
            ppv, max_pool, mean_pool = _compute_pools(conv_out)
            feat_list.extend([ppv, max_pool, mean_pool])

        all_features.append(
            np.asarray(feat_list, dtype=float) if feat_list else np.zeros(0, dtype=float)
        )

    return all_features[0] if len(all_features) == 1 else np.concatenate(all_features, axis=0)