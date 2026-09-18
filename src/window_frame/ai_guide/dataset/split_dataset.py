"""Keep exact duplicates and optional scene groups in the same split."""
import random


def split_dataset(samples, seed=42, validation_fraction=0.15, test_fraction=0.15, scene_groups=None):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    parent = {s["id"]: s["id"] for s in samples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for s in samples:
        keys = [("hash", s["input_hash"])]
        if scene_groups is not None:
            if s["id"] not in scene_groups:
                raise ValueError(f"Scene mapping missing sample {s['id']}")
            keys.append(("scene", str(scene_groups[s["id"]])))
        for key in keys:
            if key in seen:
                parent[find(s["id"])] = find(seen[key])
            else:
                seen[key] = s["id"]
    groups = {}
    for s in samples:
        groups.setdefault(find(s["id"]), []).append(s["id"])
    groups = list(groups.values())
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    nv, nt = max(1, round(n * validation_fraction)), max(1, round(n * test_fraction))
    if nv + nt >= n:
        nv = nt = 1
    flatten = lambda groups: sorted(x for group in groups for x in group)
    return dict(train=flatten(groups[nv + nt:]), validation=flatten(groups[:nv]),
                test=flatten(groups[nv:nv + nt]), seed=seed,
                grouping="scene+exact_pixels" if scene_groups is not None else "exact_pixels_only",
                warning="Without scene IDs, near-duplicate leakage remains possible.")
