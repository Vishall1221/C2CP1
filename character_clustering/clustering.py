"""
clustering.py — constraint-aware clustering primitives (numpy only, deterministic).

* UnionFind                 path-compressed disjoint sets
* merge_tracklets()         join within-shot tracklets of the same face when they
                            never appear in the same sampled frame
* constrained_hac()         average-linkage agglomerative clustering with
                            cannot-link constraints (people cannot appear twice
                            in one frame), O(n^2) memory, order independent
"""
from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if rb < ra:
            ra, rb = rb, ra          # smaller index stays root -> stable ids
        self.parent[rb] = ra
        return ra


def merge_tracklets(vectors: list[np.ndarray | None], frames: list[set[int]],
                    min_cos: float = 0.55) -> list[int]:
    """
    Group tracklets that look like the same face and never co-occur.

    vectors[i]  unit face embedding of tracklet i (None if it has no usable face)
    frames[i]   set of sample indices where tracklet i is present
    Returns root[i] — equal roots mean "same person".  Greedy by similarity,
    ties broken by index, so the result does not depend on input order beyond ids.
    """
    n = len(vectors)
    uf = UnionFind(n)
    have = [i for i in range(n) if vectors[i] is not None]
    if len(have) < 2:
        return [uf.find(i) for i in range(n)]
    mat = np.stack([vectors[i] for i in have]).astype(np.float32)
    sim = mat @ mat.T
    iu, ju = np.triu_indices(len(have), k=1)
    keep = sim[iu, ju] >= min_cos
    pairs = sorted(zip(sim[iu, ju][keep].tolist(), iu[keep].tolist(), ju[keep].tolist()),
                   key=lambda p: (-p[0], p[1], p[2]))
    group_frames = {i: set(frames[i]) for i in range(n)}
    for _, a, b in pairs:
        ra, rb = uf.find(have[a]), uf.find(have[b])
        if ra == rb or group_frames[ra] & group_frames[rb]:
            continue
        root = uf.union(ra, rb)
        other = rb if root == ra else ra
        group_frames[root] |= group_frames.pop(other)
    return [uf.find(i) for i in range(n)]


def constrained_hac(sim: np.ndarray, cannot: np.ndarray, tau: float) -> np.ndarray:
    """
    Average-linkage HAC on a similarity matrix.

    sim     (n, n) symmetric similarities (higher = more alike)
    cannot  (n, n) bool, True where two items must never share a cluster
    tau     stop merging when the best average similarity falls below tau
    Returns int labels 0..k-1 numbered by first appearance.
    A cannot-link survives merges: the merged row averages in -inf, so a
    cluster that contains one forbidden pair can never absorb the other side.
    """
    n = len(sim)
    if n == 0:
        return np.zeros(0, dtype=int)
    S = np.array(sim, dtype=np.float32, copy=True)
    S[np.asarray(cannot, dtype=bool)] = -np.inf
    np.fill_diagonal(S, -np.inf)
    size = np.ones(n, dtype=np.float32)
    label = np.arange(n)
    while True:
        flat = int(np.argmax(S))
        i, j = divmod(flat, n)
        best = S[i, j]
        if not np.isfinite(best) or best < tau:
            break
        if j < i:
            i, j = j, i
        row = (size[i] * S[i] + size[j] * S[j]) / (size[i] + size[j])
        S[i, :] = row
        S[:, i] = row
        S[i, i] = -np.inf
        S[j, :] = -np.inf
        S[:, j] = -np.inf
        size[i] += size[j]
        label[label == j] = i
    _, first = np.unique(label, return_index=True)
    order = {lab: k for k, lab in enumerate(label[np.sort(first)])}
    return np.array([order[x] for x in label], dtype=int)
