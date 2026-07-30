# RTK-Splat

RTK-Splat is an offline research pipeline for georeferenced 3D Gaussian
Splatting from calibrated stereo imagery and an external metric position
source.

The current reference method uses stereo visual bundle adjustment for locally
consistent camera poses and RTK-GNSS for the global geographic anchor. It uses
neither LiDAR nor an IMU. IMU tilt and dual-antenna heading are optional pose
inputs, not assumptions of the Gaussian mapper.

## Current result

The July 2026 headland experiment registered all 1,344 stereo frames and
improved masked PSNR from 20.610 dB with raw RTK poses to 24.445 dB with
stereo-BA poses.

| Metric | Raw RTK poses | Stereo-BA poses |
|---|---:|---:|
| Masked PSNR | 20.610 | **24.445** |
| Color-corrected masked PSNR | 21.152 | **25.848** |
| SSIM | 0.321 | **0.594** |
| Color-corrected LPIPS | 0.533 | **0.319** |

This establishes a large reconstruction-quality improvement on one difficult
headland sequence. It does not yet establish survey-grade absolute accuracy,
cross-dataset generalization, or state of the art.

See [PROGRESS.md](PROGRESS.md) for the exact status and
[the golden manifest](docs/experiments/golden/headland_stereo_ba.json) for
machine-readable provenance.

## Architecture

```text
dataset adapter
    ↓
canonical stereo segment
    ↓
explicit pose artifact
    ↓
depth + metric initialization cloud
    ↓
Gaussian training + evaluation
```

The boundary is deliberate:

- Adapters translate ROS bags, folders, or published datasets into one
  canonical segment.
- Pose backends create named, immutable pose artifacts.
- The mapping core consumes images, calibration, poses, depth, and manifests.
  It does not need ROS topic names, u-blox messages, or one robot's TF.
- Audits diagnose calibration and metric integrity but never silently modify
  normal training.

The existing ROS 2/u-blox reader and AgriGS importer are adapters. CitrusFarm
support is not claimed until its adapter and contract tests are implemented.
The exact interface is in
[DATA_CONTRACT.md](docs/architecture/DATA_CONTRACT.md).

## Install

Install the CPU-side pipeline and tests:

```bash
python -m pip install -e '.[test,diagnostics]'
```

Training additionally requires a CUDA-compatible PyTorch and gsplat build:

```bash
python -m pip install -e '.[gpu]'
```

CUDA/PyTorch/gsplat compatibility is platform-specific. The observed golden
environment is recorded in the golden manifest; the repository does not
pretend that one generic `pip` command reproduces those binaries exactly.

## Commands

Every command requires an explicit configuration:

```bash
python -m rtk_splat.cli -h
python -m rtk_splat.cli all \
  --config configs/reproductions/field_row_rtk.yaml
```

`all` runs selection, extraction, SGBM depth, cloud construction, and GS
training. It does not invoke COLMAP or the metric-integrity audit.

Check the headland stereo-BA reproduction without writing anything:

```bash
scripts/reproduce/headland_stereo_ba.sh --check \
  --python /path/to/rtk-splat/python \
  --colmap /path/to/colmap
```

The full script chooses new artifact names by default and refuses to overwrite
an existing pose artifact or GS run. It expects selection, extraction, and
depth to be complete already.

Verify the existing golden artifacts without rerunning COLMAP or GS:

```bash
python -m rtk_splat.verify_golden
```

For a new nondeterministic reproduction, apply the quality gates:

```bash
python -m rtk_splat.verify_golden \
  --mode acceptance \
  --workdir /path/to/workdir \
  --pose-artifact colmap_stereo_repro \
  --run headland_stereo_ba_repro
```

## Repository layout

```text
rtk_splat/                 Python package
tests/                     lightweight and synthetic tests
configs/reproductions/     machine-specific experiment records
configs/benchmarks/        external-method/dataset configurations
configs/environments/      environment descriptions
scripts/reproduce/         guarded long-run entry points
docs/architecture/         current contracts and design
docs/methods/              current method notes
docs/experiments/          result records and golden manifests
docs/archive/              superseded research snapshots
```

Only the reproduction configs contain local paths. They are experiment records,
not hidden defaults. The CLI has no machine-specific default config.

## Safety and reproducibility

- Source datasets are read-only.
- Long and optional stages are explicit.
- Refined poses and their initialization clouds are isolated by artifact name.
- The calibration audit is excluded from `all`, cloud building, and training.
- Existing long-run outputs are not overwritten by the canonical script.
- Generated bags, databases, arrays, checkpoints, logs, and splats are ignored
  by Git.
- Published results must record code state, config, split, pose fingerprint,
  metrics, and environment.

## Known limitations

- The reference evaluation images participated in SfM pose estimation. Their
  RGB and depth were excluded from GS training and cloud initialization, but
  this is reconstruction evaluation rather than frozen-map localization.
- There is no independent survey-grade camera trajectory for the headland run.
- Incremental COLMAP took about 581 minutes for mapping and is not suitable for
  the complete 77-minute recording without chunking or a faster backend.
- GS supervision is currently primarily left-camera RGB. Both cameras constrain
  stereo geometry and COLMAP, but dual-camera GS supervision is not yet a
  validated improvement.
- SGBM is the only built-in depth backend. Other depth methods must first be
  implemented and benchmarked rather than selected by an unused config label.
- Current ROS ingestion still has dataset-specific synchronization assumptions.

Historical plans under `docs/archive/` are provenance, not current guidance.
