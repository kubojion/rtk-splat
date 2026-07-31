# Pipeline and Boundaries

## Status and evidence

The contract-v2 architecture for Phases 0, 1, 3, and 4 is implemented and
covered by unit and synthetic tests, and the real headland input has been
published under the normalized contract. The mapper-neutral COLMAP frontend
and adaptive keyframe arms are prepared for a controlled real-data A/B, but
that A/B has not yet been run. Historical headland results in `PROGRESS.md`
were produced before this frontend refactor and remain the measured quality
and runtime reference.

Phase 2 was deliberately skipped. Phases 5--7 are outside the current
implementation scope.

## Artifact graph

```text
dataset-specific adapter
        |
        v
immutable contract-v2 segment
        |
        v
sealed mapper-neutral frontend
  images + rig + features + priors + verified pairs
        |
        +-------------------------+
        |                         |
        v                         v
private Global snapshot     private incremental snapshot
        |                         |
        +-----------+-------------+
                    v
          named metric pose artifact
                    |
                    v
        pose-matched cloud -> GS run
```

Each expensive or scientifically meaningful boundary is a named artifact.
Stages fail rather than silently overwrite an existing artifact. A backend
works from its own verified database snapshot, so Global and incremental
experiments cannot contaminate the sealed frontend or one another.

## Core and adapter boundary

`rtk_splat/core/` is the small, dataset-independent core. It defines the canonical
segment reader/writer, geometry, generic configuration, pose/cloud artifact
primitives, and golden verification. Source ingestion, visual frontends,
mapper/training backends, diagnostics, and orchestration live in sibling
subpackages of the single installed `rtk_splat` namespace. Import-isolation
tests enforce that boundary.

`rtk_splat/adapters/` owns all source-specific behavior:

- ROS version, bag storage, topics, and message types;
- image decoding and timestamp synchronization;
- calibration extraction;
- GNSS status and covariance decoding;
- optional dual-antenna heading data; and
- external trajectory or ground-truth parsing.

The registry currently implements `ros2_zed_ublox` and `agrigs`. CitrusFarm,
ROS 1 split-bag input, and arbitrary topic layouts are not yet implemented.
They cannot be enabled through YAML alone.

## Immutable contract-v2 segment

A published segment retains:

- exact integer nanosecond timestamps;
- both rectified RGB streams and both camera calibrations;
- the metric stereo transform and explicit transform semantics;
- RTK position, complete covariance, and receiver status;
- the complete dual-antenna N/E/D baseline when available; and
- explicit capabilities for optional data rather than fabricated defaults.

Single-antenna data is valid: it simply lacks the dual-antenna-heading
capability. Downstream code must react to that declared capability.

When computed SGBM depth is needed, `depth --derived-segment NEW_PATH`
publishes another immutable v2 segment. It symlinks the source images and adds
the computed depth products and capability metadata; it never edits the
ingested source segment.

The one-time migration has also been exercised on the real 1,344-frame
headland input. The normalized output is
`~/agromap4d_work/field_turn_contract_v2_normalized/segment`; its compact
hashes, exact association residuals, capabilities, and resource use are
recorded in `docs/experiments/migrations/headland_contract_v2.json`. Neither
the source v1 segment nor the earlier v2 migration was modified. No new COLMAP
or GS result is claimed from this migration yet.

## Sealed mapper-neutral frontend

The frontend is built once per experiment arm:

1. `frontend-build` symlinks canonical images and writes the immutable frame
   manifest, stereo rig, keyframe plan, match-pair plan, quality summaries,
   resolved configuration, and provenance.
2. `frontend-features` extracts features into the frontend database.
3. `frontend-rig` applies the calibrated stereo rig.
4. `frontend-priors` filters usable RTK evidence and inserts pose priors.
5. `frontend-match` verifies the planned correspondence graph and seals the
   transaction-consistent database, every finalized JSON/pair input, and the
   contents and resolved targets of all image symlinks.

The frontend does not run a mapper. Both backends receive an independently
verified copy of the same sealed evidence.

Every canonical timestamp retains both its exact left and right image names.
Keyframes reduce the expensive visual solve; they do not discard non-keyframe
images. Mandatory same-timestamp stereo pairs and attachment edges allow the
registration stage to restore every frame.

## Adaptive keyframes and match graph

The implemented presets are:

| Preset | Translation threshold | Rotation threshold |
|---|---:|---:|
| `all` | every frame | every frame |
| `dense` | 0.08 m | 1 deg |
| `balanced` | 0.10 m | 2 deg |
| `sparse` | 0.15 m | 3 deg |

All adaptive presets also impose a default 1.5 s maximum elapsed interval.
Planning can use translation, rotation, time, image quality, turn behavior,
and revisits. The match graph combines temporal and metric neighborhoods,
view direction, revisit links, mandatory stereo edges, and non-keyframe
attachment edges. Before optional edge pruning, it reserves a deterministic
physically admissible bounded-degree spanning graph across solve frames; an
unbridgeable or degree-infeasible solve fails with component diagnostics.

These policies are implemented and tested. Their registration, speed, and GS
quality on the real headland segment remain unmeasured.

## Mapper backends

Global is the default candidate; incremental is the optional fallback and
control. Both backends use the same lifecycle:

1. `backend-prepare` verifies the sealed frontend and creates a private
   transaction-consistent SQLite snapshot, including committed WAL state.
2. `backend-solve` solves only the selected keyframe images.
3. `backend-register` uses COLMAP's image registrator to recover all
   non-keyframe left and right images.
4. `backend-quality` checks registration, rig completeness, geometry, and
   provenance.
5. `backend-export` publishes a new pose artifact only after the gates pass.

The corresponding Python API is `MapperConfig`,
`prepare_mapper_backend()`, `run_mapper_solve()`,
`run_image_registration()`, `run_quality_summary()`, and
`export_pose_artifact()` in `rtk_splat.backends.mapper`.

Resume markers are stage-specific and content-verified. A marker is not trusted
when its command, input hashes, calibration, or output inventory differs.

## Exactly how RTK is used

| Stage | RTK role |
|---|---|
| Frontend planning | Translation, heading when available, revisits, and pair planning |
| COLMAP database | Filtered position priors retained as input evidence |
| Global/incremental visual solve | Visual feature geometry and rig constraints |
| Metric export | Fixed-scale SE(3) fit on calibration time blocks; RTK gates on untouched blocks |
| Diagnostic | A calibration-block Sim(3) reports scale drift but is never applied |

The integrated Global Mapper does **not** optimize covariance-weighted RTK
factors in its bundle adjustment. Preserving database priors is useful
provenance and initialization evidence, but it must not be described as an
RTK-constrained Global BA method.

Fixed scale is essential because the calibrated stereo baseline and depth are
metric. Applying a free Sim(3) scale to the trajectory while leaving stereo
depth unchanged would make the two geometry sources inconsistent.

## Current command surface

The installed interface is one explicit command per stage:

```text
validate
ingest
depth
frontend-build
frontend-features
frontend-rig
frontend-priors
frontend-match
backend-prepare
backend-solve
backend-register
backend-quality
backend-export
cloud
train
```

Every command requires `--config`. Artifact names, expected frame counts,
frontend profile, keyframe preset, backend, and run name can be overridden
explicitly. There is no `all` command and no normal training path silently
launches COLMAP, calibration, or an experimental sidecar.

## Required real-data evaluation

The next controlled A/B begins from one validated contract-v2 segment:

1. compare `gpu` with `cpu_reference` features using the `all` preset;
2. run pose-only `all`, `dense`, `balanced`, and `sparse` arms with identical
   feature evidence and mapper settings;
3. require 100% left/right registration and no georeferencing regression;
4. use cheap pose and rendering proxies to choose candidates; and
5. run GS only for the baseline and selected candidate.

The acceptance target is a material runtime reduction with no more than
0.2--0.3 dB masked-PSNR loss. Until that experiment is completed, Phase 4 is
an implemented hypothesis, not a measured improvement.
