"""Immutable train/val/test frame split, created BEFORE any map data is
built so no val/test pixel or depth can leak into initialization.

val = every `holdout_every`-th frame of the SAME traversal -- used for
metrics and A/B decisions, i.e. honestly a validation set (near-view
interpolation, not an independent test).
test = reserved for an independent traversal (reverse pass / adjacent row /
later date); never frames from the same drive-by.
"""

import json
from pathlib import Path


def load_or_create_manifest(seg: Path, holdout_every: int) -> dict:
    mpath = seg / "manifest.json"
    if mpath.exists():
        return json.loads(mpath.read_text())
    meta = json.loads((seg / "segment_meta.json").read_text())
    n = meta["n_frames"]
    manifest = {"train": [i for i in range(n) if i % holdout_every != 0],
                "val": [i for i in range(n) if i % holdout_every == 0],
                "test": [],
                "policy": f"val = every {holdout_every}th frame of the same "
                          "traversal (near-view interpolation, NOT an "
                          "independent test); test = reserved for an "
                          "independent traversal"}
    mpath.write_text(json.dumps(manifest, indent=2))
    return manifest
