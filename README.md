# RTK-Splat

RTK-Splat is an offline research pipeline for georeferenced 3D Gaussian
Splatting from calibrated stereo imagery and an external metric position
source. The mapping core is dataset-neutral: adapters publish one immutable
contract-v2 segment, a sealed visual frontend is reused by independent pose
backends, and Gaussian training consumes an explicitly named pose artifact.

LiDAR and IMU are not required. IMU evidence and dual-antenna heading are
optional contract capabilities; the current headland reference uses stereo
images and RTK-GNSS.

## What is measured and what is new

Two July 2026 headland results are frozen as golden history.

| Metric | Raw RTK poses | Incremental stereo BA |
|---|---:|---:|
| Masked PSNR | 20.6097 | **24.4449** |
| Corrected masked PSNR | 21.1519 | **25.8480** |
| SSIM | 0.3209 | **0.5944** |
| Corrected LPIPS | 0.5333 | **0.3187** |

The reduced Global Mapper control then retained every frame and matched the
incremental GS quality while reducing the historical mapper stage from about
581 minutes to 18.1 minutes:

| Metric | Incremental stereo BA | Reduced Global Mapper | Difference |
|---|---:|---:|---:|
| Masked PSNR | 24.4449 | 24.4583 | +0.0134 dB |
| Corrected masked PSNR | 25.8480 | 25.8418 | -0.0062 dB |
| SSIM | 0.5944 | 0.5904 | -0.0040 |
| LPIPS | 0.3232 | 0.3227 | -0.0005 |

These are results on one 1,344-frame headland sequence, not evidence of
cross-dataset generalization or state of the art. Exact hashes, unrounded
metrics, environment records, and provenance are in
[the golden manifests](docs/experiments/golden/).

The current repository has since implemented and unit-tested a cleaner
contract-v2 path:

- Phase 0: the two measured golden results are frozen and regression checked.
- Phase 1: contract v2, isolated adapters, adapter conformance tests, two-level
  robot/sequence configuration, and the one-time v1 migration utility are
  implemented.
- Phase 2: independent target-based stereo calibration was deliberately
  skipped.
- Phase 3: a sealed mapper-neutral COLMAP frontend and isolated Global or
  incremental backends are implemented.
- Phase 4: adaptive solve keyframes and RTK-guided bounded matching are
  implemented.

The normalized 1,344-frame headland contract-v2 migration is complete and
hash-recorded. Phases 3 and 4 are prepared and covered by synthetic/unit tests,
but the GPU-versus-CPU feature A/B and the
all/dense/balanced/sparse keyframe A/B have not yet been run through COLMAP and
GS from that segment. Do not confuse the historical Global result above with
validation of the new frontend.
RTK-anchored submaps, CitrusFarm support, and rendering-quality experiments
(Phases 5--7) remain out of scope for this implementation.

## Architecture

```text
dataset-specific source
        ↓  adapter
immutable contract-v2 segment
        ↓
sealed frontend: symlinked images + rig + features + verified matches + priors
        ↓                         ↓
Global Mapper (primary)      incremental mapper (fallback/reference)
        └──────────────┬──────────┘
                       ↓  image_registrator adds every non-keyframe
fixed-scale ENU pose artifact
        ↓
pose-matched cloud → GS train → evaluate
```

The core package under `rtk_splat/core/` imports no ROS, bag, dataset, backend,
or workflow module. ROS 2 ZED/u-blox and AgriGS are adapters under
`rtk_splat/adapters/`.
CitrusFarm is not supported until a ROS 1/split-bag adapter passes the same
contract tests.

RTK use is intentionally explicit. Covariance, status, timestamps, and
optional full dual-antenna baselines are preserved in contract v2. Trusted
Cartesian camera-centre priors are inserted into the reusable frontend
database and retained as evidence. The current `global_mapper` solve itself is
visual: it does **not** optimize RTK residual factors in Global Mapper bundle
adjustment. After visual solving and all-frame registration, the exporter fits
the fixed-scale SE(3) alignment on alternating contiguous calibration blocks
and gates georeferencing on untouched temporal blocks. A Sim(3) scale is fit
on calibration blocks only, reported as a diagnostic, and never applied.

See [PIPELINE.md](docs/architecture/PIPELINE.md) and
[DATA_CONTRACT.md](docs/architecture/DATA_CONTRACT.md).

## Install

Install the CPU-side pipeline and tests:

```bash
python -m pip install -e '.[test,diagnostics]'
```

ROS 2 bag ingestion is optional:

```bash
python -m pip install -e '.[ros2]'
```

Training additionally requires a compatible CUDA, PyTorch, and gsplat setup:

```bash
python -m pip install -e '.[gpu]'
```

The observed golden environment is recorded separately. A generic `pip`
command cannot guarantee the same CUDA/COLMAP binaries.

## Workflow commands

There is no `all` command. Long stages are explicit, independently rerunnable,
and write separate artifacts. Every invocation requires a configuration:

```bash
rtk-splat -h
rtk-splat validate --config configs/sequences/headland.example.yaml
```

From a source checkout, `python -m rtk_splat.workflows.cli` is equivalent to the
installed `rtk-splat` entry point.

For a new source recording, publish and validate a new immutable segment:

```bash
rtk-splat ingest \
  --config configs/sequences/headland.example.yaml \
  --expected-frames 1344

rtk-splat validate \
  --config configs/sequences/headland.example.yaml \
  --expected-frames 1344
```

If the source segment has no usable depth, derive SGBM depth into a **new**
immutable contract-v2 segment. The source images are symlinked and the source
segment is never modified:

```bash
rtk-splat depth \
  --config configs/sequences/headland.example.yaml \
  --derived-segment /new/experiment/segments/headland-sgbm
```

Point the following experiment configuration (or `--segment`) at that derived
segment. If a validated v2 segment with suitable depth already exists, this
stage is unnecessary. Then build one frontend foundation and run each COLMAP
stage:

```bash
rtk-splat frontend-build \
  --config configs/sequences/headland.example.yaml \
  --frontend-name balanced-gpu \
  --keyframe-preset balanced

rtk-splat frontend-features \
  --config configs/sequences/headland.example.yaml \
  --frontend-name balanced-gpu \
  --feature-profile gpu

rtk-splat frontend-rig \
  --config configs/sequences/headland.example.yaml \
  --frontend-name balanced-gpu

rtk-splat frontend-priors \
  --config configs/sequences/headland.example.yaml \
  --frontend-name balanced-gpu

rtk-splat frontend-match \
  --config configs/sequences/headland.example.yaml \
  --frontend-name balanced-gpu
```

The same sealed frontend can feed separate backend artifacts without
recomputing features or matches:

```bash
rtk-splat backend-prepare \
  --config configs/sequences/headland.example.yaml \
  --frontend-name balanced-gpu \
  --backend global \
  --backend-name balanced-gpu-global

rtk-splat backend-solve \
  --config configs/sequences/headland.example.yaml \
  --backend-name balanced-gpu-global

rtk-splat backend-register \
  --config configs/sequences/headland.example.yaml \
  --backend-name balanced-gpu-global

rtk-splat backend-quality \
  --config configs/sequences/headland.example.yaml \
  --backend-name balanced-gpu-global

rtk-splat backend-export \
  --config configs/sequences/headland.example.yaml \
  --backend-name balanced-gpu-global \
  --pose-name balanced-gpu-global
```

Use `--backend incremental` and a different backend name for the fallback or
reference arm. Backend export refuses an existing destination and publishes
all-frame OpenCV view matrices, ENU camera centres, exact frame IDs and
timestamps, alignment diagnostics, quality gates, and provenance under
`<workdir>/pose_artifacts/<pose-name>/`.

Cloud construction and training remain explicit:

```bash
rtk-splat cloud \
  --config configs/sequences/headland.example.yaml \
  --pose-name balanced-gpu-global

rtk-splat train \
  --config configs/sequences/headland.example.yaml \
  --pose-name balanced-gpu-global \
  --run-name balanced-gpu-global-15k \
  --train-iters 15000
```

Do pose-only comparisons first. A short 15,000-iteration GS proxy should be run
only after an arm passes registration, reprojection, RTK, and scale gates; one
full 65,000-iteration run is reserved for the selected arm.

Verify the historical golden artifacts without rerunning COLMAP or GS:

```bash
rtk-splat-verify
```

## Migration status

`rtk_splat.adapters.migrate_v1_to_v2` is a non-destructive, fail-closed
one-time converter for the validated legacy headland layout. It preserves full
timestamps, GNSS covariance/status, and dual-antenna evidence while symlinking
bulk image/depth data. There is no v1 reader in the mapping core.

The strict normalized migration was validated at
`~/agromap4d_work/field_turn_contract_v2_normalized/segment`: 1,344/1,344
position and heading rows are valid RTK-fixed observations, raw GNSS and
heading streams each retain 2,273 samples, and the 1,176/168/0 split is
unchanged. Images and computed depth are directory symlinks to the untouched
source segment. Exact output hashes, association residuals, capabilities,
runtime, and memory are recorded in
[the migration receipt](docs/experiments/migrations/headland_contract_v2.json).
This proves the conversion/contract boundary, not the unrun COLMAP or GS A/B.

## Repository layout

```text
rtk_splat/                  the only installed Python package
  core/                     small dataset-neutral contract and primitives
  adapters/                 ROS/dataset ingestion and one-time migration
  frontends/                keyframes, pair graph, sealed COLMAP frontend
  backends/                 Global/incremental mapping and gsplat
  workflows/                explicit command-line orchestration
  diagnostics/              optional ROS-free analysis
tests/                      contract, isolation, geometry, and dry-run tests
configs/robots/             stable platform facts
configs/sequences/          per-recording facts
configs/reproductions/      frozen historical experiment records
docs/experiments/golden/    accepted metrics and hashes
```

## Current limitations

- Validation views participated in the historical SfM pose estimation.
- The headland sequence has no independent survey-grade camera trajectory.
- The new sealed frontend and adaptive-keyframe path still need controlled
  real-data runtime, pose, and GS A/B results.
- Global Mapper is visual; RTK factors inside local/global BA are not
  implemented.
- Full-field bounded submaps and cross-session merging are not implemented.
- CitrusFarm/ROS 1 split-bag ingestion is not implemented.
- Right-camera GS photometric supervision and learned stereo depth are not
  validated.

See [PROGRESS.md](PROGRESS.md) for the detailed evidence and next experiment.
