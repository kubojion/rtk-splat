"""Read the immutable train/validation/test split from contract v2.

val = every `holdout_every`-th frame of the SAME traversal -- used for
metrics and A/B decisions, i.e. honestly a validation set (near-view
interpolation, not an independent test).
test = reserved for an independent traversal (reverse pass / adjacent row /
later date); never frames from the same drive-by.
"""

from pathlib import Path

from .segment import SegmentReader


def load_manifest(segment: str | Path) -> dict:
    """Return the adapter-authored split; core stages never invent or rewrite it."""
    return SegmentReader(segment).manifest
