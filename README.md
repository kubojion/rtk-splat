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
- Phase 1: contract v2, isolated adapters, adapter conformance tests,
  profile/robot/sequence configuration plus auditable runtime derivation, and
  the one-time v1 migration utility are implemented.
- Phase 2: independent target-based stereo calibration was deliberately
  skipped.
- Phase 3: a sealed mapper-neutral COLMAP frontend and isolated Global or
  incremental backends are implemented.
- Phase 4: adaptive solve keyframes and RTK-guided bounded matching are
  implemented.
- A ROS 1 ordered-bag adapter, staged reproduction launcher, and optional
  factor-held-out RTK position-refinement backend are implemented for the CitrusFarm
  sequence-05 window at 543--735 s.

The normalized 1,344-frame headland contract-v2 migration is complete and
hash-recorded. Phases 3 and 4 are prepared and covered by synthetic/unit tests,
but the GPU-versus-CPU feature A/B and the
all/dense/balanced/sparse keyframe A/B have not yet been run through COLMAP and
GS from that segment. Do not confuse the historical Global result above with
validation of the new frontend. The full Citrus window has completed ingest,
SGBM, a sealed frontend, and a 2,990-image visual reconstruction. Both its
visual-only export and a separate 897-prior/598-holdout position-refinement A/B
failed the refinement-factor holdout RTK gates, so no Citrus pose, GS model, or rendering
score is accepted. Bounded custom RTK-factor submaps and rendering-quality
experiments remain future work.

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
        optional RTK position refinement (calibration blocks only)
                       ↓  blocks excluded from refinement-factor gates
fixed-scale ENU pose artifact
        ↓
pose-matched cloud → GS train → evaluate
```

The core package under `src/rtk_splat/core/` imports no ROS, bag, dataset, backend,
or workflow module. ROS 2 ZED/u-blox, ROS 1 CitrusFarm, and AgriGS are adapters
under `src/rtk_splat/adapters/`. The CitrusFarm implementation handles an ordered
chain of ROS 1 bags, raw rectified stereo, Piksi `NavSatFix`, single-antenna
course heading, clock auditing, and travelled-distance sampling without adding
dataset conditionals to the mapping core.

RTK use is intentionally explicit. Covariance, status, timestamps, and
optional full dual-antenna baselines are preserved in contract v2. Trusted
Cartesian camera-centre priors are inserted into the reusable frontend
database and retained as evidence. The primary Global Mapper solve itself is
visual: it does **not** optimize RTK residual factors in its bundle
adjustment. An explicit optional sidecar can refine positions using
calibration-block priors; it removes holdouts from a private database, freezes
metric stereo calibration, and rejects scale, baseline, or visual regressions.
It is position-only and does not add a dual-antenna heading factor. After pose
solving and all-frame registration, the exporter fits
the fixed-scale SE(3) alignment on alternating contiguous calibration blocks
and gates georeferencing on blocks untouched by that fit/refinement. These
blocks are not end-to-end GNSS-independent because upstream frame and pair
planning may use the full RTK track. A Sim(3) scale is fit
on calibration blocks only, reported as a diagnostic, and never applied.

See [PIPELINE.md](docs/architecture/PIPELINE.md) and
[DATA_CONTRACT.md](docs/architecture/DATA_CONTRACT.md).

## Configuration model

Configuration has three authored layers and one measured layer:

- `profiles/` owns versioned method and quality policy;
- `robots/` owns stable topics, sensor geometry, and pose-source facts;
- `sequences/` owns recording paths, windows, artifact names, and a narrow set
  of documented scene overrides; and
- runtime resolution derives data-dependent controls only after their inputs
  exist.

The runtime-derived controls are frame stride or metric spacing from measured
motion, stereo range from focal-length-times-baseline and reliable disparity,
training iterations from the actual number of training views, and Gaussian
capacity from initial-cloud size and available VRAM. `auto` opts into a named
formula; a numeric value is an explicit override. Formula inputs, bounds,
source-file hashes, CLI overrides, and chosen values are written to
`<workdir>/config_artifacts/resolved_config.json` and copied into immutable
stage evidence. Reusing a work directory after a configuration source changes
fails closed.

`quality_v1` is an empirical candidate policy, not a cross-dataset optimum or
a quality guarantee. RTK acceptance limits remain independent, authored metre
caps; receiver covariance is evidence for weighting and diagnostics, not
permission to silently relax a failed georeferencing result. See
[the configuration guide](configs/README.md).

## Install

Install the CPU-side pipeline and tests:

```bash
python -m pip install -e '.[test,diagnostics]'
```

ROS 2 bag ingestion is optional:

```bash
python -m pip install -e '.[ros2]'
```

ROS 1 bag ingestion, including CitrusFarm, uses the separate optional extra:

```bash
python -m pip install -e '.[ros1]'
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

# Optional and experimental: publishes a separate refinement sidecar.
rtk-splat backend-refine-rtk \
  --config configs/sequences/headland.example.yaml \
  --backend-name balanced-gpu-global \
  --refinement-name balanced-gpu-global-rtk-v1

rtk-splat backend-export \
  --config configs/sequences/headland.example.yaml \
  --backend-name balanced-gpu-global \
  --refinement-name balanced-gpu-global-rtk-v1 \
  --pose-name balanced-gpu-global
```

`backend-refine-rtk` uses covariance-weighted Cauchy position residuals by
default. `--prior-position-loss trivial` is an explicit diagnostic that keeps
the same covariance whitening but removes Cauchy downweighting. Use a new
refinement name for every arm. A missing loss field in a sealed legacy plan is
interpreted as the historical Cauchy default without modifying that plan.
`--initialization-mode fresh` omits the finished input model and lets the same
calibration-block priors constrain COLMAP while it builds the reconstruction.
The historical/default `continuation` mode remains byte-compatible. Fresh mode
uses absolute graph-quality gates because its tracks are newly constructed.

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

That no-argument shortcut is available only in a source checkout. An installed
wheel is dataset-neutral and therefore requires an explicit experiment record:

```bash
rtk-splat-verify --manifest /path/to/experiment-manifest.json
```

## Prepared CitrusFarm reproduction

The checked-in CitrusFarm candidate uses the 192 s window at 543--735 s,
0.15 m travelled-distance sampling, stereo RGB, computed SGBM depth, and
single-antenna Piksi RTK. Its launcher calls the same explicit workflow stages,
writes only to an internal-disk work directory, does not use `tmux`, and
resumes only stages whose code/config/command evidence still matches:

```bash
bash scripts/runs/citrusfarm_05_13d_uturn.sh plan
bash scripts/runs/citrusfarm_05_13d_uturn.sh preflight
bash scripts/runs/citrusfarm_05_13d_uturn.sh run \
  --workdir /home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_v2
```

The real read-only adapter preflight measured a 228.726 m GNSS path, estimated
about 1,525 stereo pairs, recovered a 0.119885 m recorded rig baseline, and
estimated a +72.548749 ms camera-to-GNSS clock correction. That audited value
is now frozen in the sequence profile (zero configured residual); estimated
window drift was -2.581 ms, within the 15 ms gate. Recorded ZED depth and
confidence are inventoried but are not used by this primary arm; SGBM
publishes depth in a new immutable derived segment.

A 4 s real-data smoke published 10 stereo pairs, computed SGBM depth, and
sealed a 20-image GPU frontend with all 52 requested pairs verified. The first
full-window attempt later published 1,495 stereo pairs and registered all 2,990
images at 0.973 px mean reprojection error, but it failed the untouched-block
RTK export gate (0.266 m median and 59.2% support). A separate 31.7-minute
position-prior refinement preserved the rig/intrinsics and improved those
figures to 0.249 m and 67.2%, but still failed the 0.15 m/80% gates. It also
reduced reprojection error to 0.674 px. Neither arm published an accepted pose,
cloud, GS model, or PSNR. The adapter now also requires and
preserves the receiver's own fixed/float state; the recorded NavSatFix
covariance is explicitly static/approximated rather than live accuracy. See
`PROGRESS.md` for the exact audit and measured stage timings. The 24.8 dB
headland result remains a historical regression reference, not a promised or
directly comparable CitrusFarm score.
The rejected refinement A/B is frozen in
`docs/experiments/citrusfarm_rtk_refinement_v1.json` so it cannot silently be
relabelled as an accepted result later.

The controlled cached loss experiment completed. Its quadratic arm improved
held-out median RTK residual to 0.2284 m and support to 95.32%, but still failed
the 0.15 m median and 0.30 m inlier-p95 gates, so it remains rejected.

The fresh-initialization arm has also completed. It registered all 2,990 images
at 0.684 px and improved the held-out median to 0.196 m with 99.16% support,
but failed the unchanged 0.15 m median, 0.30 m p95 and 0.5% scale gates. It
therefore remains a rejected experimental baseline and produced no pose,
cloud, GS model or PLY. Its sealed paired receipt is
`docs/experiments/citrusfarm_pose_prior_initialization_ab_v1.json`.

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
src/rtk_splat/              the only installed Python package
  core/                     small dataset-neutral contract and primitives
  adapters/                 ROS/dataset ingestion and one-time migration
  frontends/                keyframes, pair graph, sealed COLMAP frontend
  backends/                 Global/incremental mapping and gsplat
  workflows/                explicit command-line orchestration
  diagnostics/              optional ROS-free analysis
tests/                      contract, isolation, geometry, and dry-run tests
configs/robots/             stable platform facts
configs/sequences/          per-recording facts
configs/profiles/           reusable quality policy
configs/reproductions/      frozen historical experiment records
docs/experiments/golden/    accepted metrics and hashes
```

The `src/` boundary ensures that tests and scripts exercise the installed
package layout instead of importing a same-named directory merely because the
repository is the current working directory. Configurations and experiment
records remain repository data; they are not hidden inside the wheel.

## Current limitations

- Validation views participated in the historical SfM pose estimation.
- The headland sequence has no independent survey-grade camera trajectory.
- The new sealed frontend and adaptive-keyframe path still need controlled
  real-data runtime, pose, and GS A/B results.
- Global Mapper is visual. The optional whole-model sidecar consumes RTK
  position priors but failed its first real held-out A/B; custom bounded local
  RTK factors and a submap graph are not implemented.
- Full-field bounded submaps and cross-session merging are not implemented.
- CitrusFarm has completed one full-window visual reconstruction and one
  rejected RTK-refinement control; the first accepted georeferenced pose and
  GS result remain future work.
- Right-camera GS photometric supervision and learned stereo depth are not
  validated.

See [PROGRESS.md](PROGRESS.md) for the detailed evidence and next experiment.
