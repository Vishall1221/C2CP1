"""
grouping.py — cross-shot outfit grouping per cast member (vectorised).

Same scoring and join rules as the original implementation (DINOv2
region vectors + colour histograms + changed-layer veto), but:
  * the full pairwise score table is built once with blocked BLAS calls
  * the expensive pixel-level `changed_layer` veto runs lazily, only for pairs
    that are actually candidates, and is cached
  * group joins are numpy sub-matrix operations, not Python double loops
Output is deterministic (no dependence on dict/set order or caches).
"""
from __future__ import annotations

from base64 import b64decode
from collections import defaultdict

import cv2
import numpy as np

from outfit_analysis.clothing import wardrobe_view

JOIN = 0.66              # minimum group-join strength
TIME_BONUS = 0.10        # extra strength for views close in time
TIME_SCALE_MS = 12000.0
BLOCK = 128              # rows per BLAS block when building the score table
# Two established groups (>= 3 views each) are only joined if at least one cross pair agrees on
# colour. On the test clip correct joins of such groups had cross-colour max >= 0.826; the one
# wrong join (grey track jacket + orange tee vs navy/maroon jacket + white tee, 23 + 6 views)
# was bridged by a single DINOv2 pair at strength 0.690 with colour max 0.633.
COLOUR_VETO = 0.70
COLOUR_VETO_MIN_VIEWS = 3


# ── Pairwise fabric similarity ────────────────────────────────────────────────

def fabric_scores(
    left: np.ndarray,
    right: np.ndarray,
    vectors: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, w = vectors[left], vectors[right]
    c, d = colors[left], colors[right]
    appearance = np.einsum("ird,jrd->ijr", v, w, optimize=True)
    colour = np.einsum("ird,jrd->ijr", c, d, optimize=True)
    visible = np.minimum(sizes[left, None, 1:], sizes[right][None, :, 1:]) > 0.23
    parts = 0.70 * appearance[:, :, 2:6] + 0.30 * colour[:, :, 1:]
    agreements = (parts * visible).sum(2) / np.maximum(visible.sum(2), 1)
    score = (0.19 * appearance[:, :, 0] + 0.21 * appearance[:, :, 1]
             + 0.14 * agreements + 0.24 * colour[:, :, 0]
             + 0.22 * appearance[:, :, 6])
    mismatch = np.any(
        visible & (appearance[:, :, 2:6] < 0.34) & (colour[:, :, 1:] < 0.25), axis=2
    )
    return score, colour[:, :, 0], mismatch


def changed_layer(a: np.ndarray, b: np.ndarray) -> bool:
    """Veto localized clothing changes when pose and background align."""
    outside = (a[:, :, 3] < 72) & (b[:, :, 3] < 72)
    cloth = (a[:, :, 3] > 160) & (b[:, :, 3] > 160)
    if outside.sum() < 90 or cloth.sum() < 40:
        return False
    delta = a[:, :, :3] - b[:, :, :3]
    correction = np.median(delta[outside], axis=0)
    diff = np.abs(delta - correction)
    close = (diff[:, :, 0] < 18) & (diff[:, :, 1:].max(axis=2) < 11)
    if close[outside].mean() < 0.77:
        return False
    changed = cloth & ((diff[:, :, 1:].max(axis=2) > 19) | (diff[:, :, 0] > 53))
    islands = cv2.connectedComponentsWithStats(changed.astype(np.uint8))[2][1:]
    return bool(len(islands) and islands[:, cv2.CC_STAT_AREA].max()
                >= max(10, 0.14 * cloth.sum()))


# ── Score / clash tables ──────────────────────────────────────────────────────

class Matches:
    """Dense symmetric tables over all views of one character."""

    def __init__(self, rows, vectors, colors, sizes):
        n = len(rows)
        self.rows = rows
        self.maps = [
            np.frombuffer(b64decode(r["alignment_map"]), np.uint8).reshape(24, 24, 4).astype(np.int16)
            for r in rows
        ]
        idx = np.arange(n)
        self.score = np.empty((n, n), np.float32)
        self.color = np.empty((n, n), np.float32)
        self.mismatch = np.empty((n, n), bool)
        for r0 in range(0, n, BLOCK):
            s, c, m = fabric_scores(idx[r0:r0 + BLOCK], idx, vectors, colors, sizes)
            self.score[r0:r0 + BLOCK], self.color[r0:r0 + BLOCK], self.mismatch[r0:r0 + BLOCK] = s, c, m
        ms = np.array([r["ms"] for r in rows], dtype=np.float64)
        self.strength = (self.score + TIME_BONUS
                         * np.exp(-np.abs(ms[:, None] - ms[None, :]) / TIME_SCALE_MS)).astype(np.float32)
        self.coverage = np.array([r["coverage"] for r in rows], dtype=np.float64)
        self._clash: dict[tuple[int, int], bool] = {}

    def clash(self, i: int, j: int) -> bool:
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))
        key = (i, j)
        hit = self._clash.get(key)
        if hit is None:
            hit = bool(
                (self.mismatch[i, j] and abs(self.coverage[i] - self.coverage[j]) < 0.20
                 and self.score[i, j] > 0.52)
                or changed_layer(self.maps[i], self.maps[j])
            )
            self._clash[key] = hit
        return hit

    def pair(self, i: int, j: int) -> tuple[float, float, bool]:
        return float(self.score[i, j]), float(self.color[i, j]), self.clash(i, j)

    def any_clash(self, left, right) -> bool:
        return any(self.clash(i, j) for i in left for j in right)


# ── Cluster helpers ───────────────────────────────────────────────────────────

def _examples(group: list, rows: list, limit: int = 5) -> list:
    if len(group) <= limit:
        return group
    ranked = sorted(group, key=lambda i: -rows[i]["quality"])
    chosen = [ranked[0],
              min(group, key=lambda i: rows[i]["coverage"]),
              max(group, key=lambda i: rows[i]["coverage"])]
    return list(dict.fromkeys(chosen + ranked))[:limit]


def _join_strength(left, right, matches: Matches) -> float:
    if matches.any_clash(left, right):
        return 0.0
    sel = np.ix_(np.asarray(left), np.asarray(right))
    if (min(len(left), len(right)) >= COLOUR_VETO_MIN_VIEWS
            and float(matches.color[sel].max()) < COLOUR_VETO):
        return 0.0
    weights = matches.strength[sel]
    support = min(np.quantile(weights.max(0), 0.65), np.quantile(weights.max(1), 0.65))
    return float(min(weights.max(), support + 0.14))


def _cluster_anchors(indices: list, matches: Matches) -> list[list]:
    idx = np.asarray(indices)
    groups = {int(i): [int(i)] for i in idx}
    owner = {int(i): int(i) for i in idx}
    strength = matches.strength[np.ix_(idx, idx)]
    iu, ju = np.triu_indices(len(idx), 1)
    candidate = strength[iu, ju] >= JOIN
    edges = []
    for a, b in zip(iu[candidate], ju[candidate]):
        i, j = int(idx[a]), int(idx[b])
        if not matches.clash(i, j):                       # pixel-level veto only for candidates
            edges.append((float(strength[a, b]), i, j))
    for _, i, j in sorted(edges, reverse=True):
        a, b = owner[i], owner[j]
        if a == b:
            continue
        if _join_strength(groups[a], groups[b], matches) >= JOIN:
            groups[a].extend(groups.pop(b))
            for k in groups[a]:
                owner[k] = a
    return list(groups.values())


# ── Short-return attachment ───────────────────────────────────────────────────

def _attach_short_returns(groups: list[list], rows: list, matches: Matches):
    for group in sorted(groups.copy(), key=len):
        if group not in groups or len(group) > 6 or len(groups) < 2:
            continue
        start = min(rows[i]["ms"] for i in group)
        end = max(rows[i]["ms"] for i in group)
        others = [(rows[i]["ms"], g) for g in groups if g is not group for i in g]
        before_items = [(t, g) for t, g in others if t < start]
        after_items = [(t, g) for t, g in others if t > end]
        before = max(before_items, key=lambda p: p[0], default=(None, None))
        after = min(after_items, key=lambda p: p[0], default=(None, None))

        neighbors = []
        if before[1] is not None and start - before[0] <= 14000:
            neighbors.append(before[1])
        if after[1] is not None and after[0] - end <= 14000 and after[1] is not before[1]:
            neighbors.append(after[1])

        eligible = []
        for other in neighbors:
            pairs = [matches.pair(min(i, j), max(i, j))
                     for i in group for j in _examples(other, rows)]
            if any(v for _, _, v in pairs):
                continue
            nearby = sorted(other, key=lambda j: min(abs(rows[j]["ms"] - rows[i]["ms"])
                                                     for i in group))[:4]
            checks = [matches.pair(min(i, j), max(i, j)) for i in group for j in nearby]
            if any(v for _, _, v in checks):
                continue
            same_both = other is before[1] and other is after[1] and end - start <= 12000
            viable = [s for s, c, _ in checks + pairs if c >= (0.52 if same_both else 0.65)]
            agreement = max(viable, default=0.0)
            if agreement >= (0.50 if same_both else 0.565):
                eligible.append((agreement, other))

        eligible.sort(key=lambda it: -it[0])
        if eligible and (len(eligible) == 1 or eligible[0][0] - eligible[1][0] >= 0.045):
            eligible[0][1].extend(group)
            groups.remove(group)


# ── Entry point ───────────────────────────────────────────────────────────────

def group_views(
    rows: list,
    vectors: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
) -> tuple[list[list], list]:
    """
    Cluster clothing appearances into distinct outfit groups.
    Returns (groups of row indices, unresolved row indices).
    """
    if not rows:
        return [], []

    anchors = [i for i, r in enumerate(rows) if wardrobe_view(r)]
    matches = Matches(rows, vectors, colors, sizes)
    groups: list[list] = []

    for off in range(0, len(anchors), 192):             # bound the quadratic edge scan
        new = _cluster_anchors(anchors[off:off + 192], matches)
        if off == 0:
            groups = new
            continue
        for grp in new:
            left = _examples(grp, rows)
            scores = [_join_strength(left, _examples(g, rows), matches) for g in groups]
            if scores and max(scores) >= JOIN:
                groups[int(np.argmax(scores))].extend(grp)
            else:
                groups.append(grp)

    _attach_short_returns(groups, rows, matches)

    anchor_set = set(anchors)
    banks = [_examples(g, rows) for g in groups]
    unresolved = []
    for i, r in enumerate(rows):
        if i in anchor_set:
            continue
        options = []
        for n, grp in enumerate(groups):
            nearest = min((j for j in grp if j in anchor_set),
                          key=lambda j: abs(rows[j]["ms"] - r["ms"]), default=None)
            if nearest is None:
                continue
            checks = [matches.pair(min(i, j), max(i, j)) for j in set(banks[n] + [nearest])]
            if any(v for _, _, v in checks):
                continue
            score = max(s for s, _, _ in checks)
            score += 0.09 * np.exp(-abs(r["ms"] - rows[nearest]["ms"]) / 8000)
            options.append((score, n))
        options.sort(reverse=True)
        if options and options[0][0] >= 0.59 and (
            len(options) == 1 or options[0][0] - options[1][0] >= 0.045
        ):
            groups[options[0][1]].append(i)
        else:
            unresolved.append(i)

    return groups, unresolved
