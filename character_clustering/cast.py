"""
cast.py - main-character clustering: tracks are clustered by face embedding
(constrained average-link: two tracks visible at the same time cannot be one person),
then ranked by screen time. Writes up to 3 face photos per character.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from character_clustering.clustering import constrained_hac


def cluster_tracks(found: list, tau: float) -> list[list[int]]:
    """Cluster tracks (with a face vector) by face; cannot-link = same shot, overlapping frames."""
    items = [i for i, (_, vec, _) in enumerate(found) if vec is not None]
    if not items:
        return []
    mat = np.stack([found[i][1] for i in items]).astype(np.float32)
    sim = mat @ mat.T
    n = len(items)
    cannot = np.zeros((n, n), dtype=bool)
    by_shot: dict[str, list[int]] = defaultdict(list)
    for k, i in enumerate(items):
        by_shot[found[i][0]["shot_id"]].append(k)
    for ks in by_shot.values():
        for a in ks:
            ta = found[items[a]][0]
            for b in ks:
                if a < b:
                    tb = found[items[b]][0]
                    if min(ta["end_ms"], tb["end_ms"]) >= max(ta["start_ms"], tb["start_ms"]):
                        cannot[a, b] = cannot[b, a] = True
    labels = constrained_hac(sim, cannot, tau)
    groups: dict[int, list[int]] = defaultdict(list)
    for k, lab in enumerate(labels):
        groups[int(lab)].append(items[k])
    return list(groups.values())

def group_cast(found: list, root: Path, *, tau: float) -> tuple[list[dict], list[dict]]:
    clusters = cluster_tracks(found, tau)

    def rank(members):
        tracks = [found[i][0] for i in members]
        return (-sum(t["visible_ms"] for t in tracks),
                -len({t["shot_id"] for t in tracks}),
                min(t["start_ms"] for t in tracks))

    clusters.sort(key=rank)
    candidates = []
    for n, members in enumerate(clusters, 1):
        cid = f"cast-{n:03}"
        tracks = [found[i][0] for i in members]
        options = sorted((f for i in members for f in found[i][2]), key=lambda f: (-f.quality, f.time_ms))
        used: set[int] = set()
        portraits: list[str] = []
        for face in options:
            if face.time_ms // 1000 in used:
                continue
            used.add(face.time_ms // 1000)
            rel = Path("characters") / cid / f"photo-{len(portraits) + 1}.jpg"
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(face.portrait)
            portraits.append(rel.as_posix())
            if len(portraits) == 3:
                break
        candidates.append({
            "id": cid, "track_ids": sorted(t["id"] for t in tracks),
            "shots": len({t["shot_id"] for t in tracks}),
            "visible_ms": sum(t["visible_ms"] for t in tracks), "portraits": portraits,
        })
    return candidates, [t for t, _, _ in found]
