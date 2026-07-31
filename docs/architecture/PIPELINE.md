# Pipeline and Boundaries

## Stable core

The reusable mapping path is:

```text
canonical images + calibration
          │
          ├── depth artifacts
          │
          └── named metric poses
                    │
                    ↓
          pose-matched initialization cloud
                    ↓
             GS train and evaluate
```

The core does not need to know whether poses came from dual-antenna RTK,
course-over-ground GNSS, a trajectory file, COLMAP, or a future local
stereo–RTK estimator.

## Adapters

Adapters own dataset-specific work:

- ROS version and storage format.
- Topic and message types.
- Image decompression and synchronization.
- Sensor calibration extraction.
- GNSS status/covariance decoding.
- Published trajectory or ground-truth parsing.

The current `bagio.py` and `ingest_agrigs.py` are adapter code even though they
remain in the flat Python package during Phase 1. Moving modules without an
end-to-end test would add churn without improving scientific behavior. A future
adapter package can be introduced after the contract is exercised by a second
real dataset.

## Pose backends

Current:

- Raw dual-antenna RTK pose construction.
- GNSS course-derived heading.
- External TUM camera trajectory.
- Calibrated stereo incremental COLMAP plus geographic alignment.
- Calibrated stereo COLMAP Global Mapper using a copied cached front end,
  fixed-scale geographic alignment, and strict publication gates.
- A resource-bounded Global Mapper profile that retains the complete pair graph
  but caps longest-track global positioning and audits per-image support.

Diagnostic only:

- Bounded RTK–stereo extrinsic and clock integrity audit.

Planned:

- RTK-constrained global BA rather than post-hoc anchoring alone.
- RTK-anchored chunks with overlap constraints.
- Bounded sliding-window stereo–RTK estimation after simpler backends are
  measured.

Every backend writes a new named artifact and quality report. There is no silent
fallback from visual poses to raw RTK for missing frames.

## Stage behavior

`all` intentionally means the baseline:

```text
select → extract → depth → cloud → train
```

It does not mean “run every experimental sidecar.” COLMAP and calibration are
explicit because they are long, optional, and should never surprise a user.

The headland reproduction script starts from an already extracted canonical
segment. A clean-from-dataset reproduction remains a sequence of explicit CLI
stages, not a hidden behavior of that long-run script.
