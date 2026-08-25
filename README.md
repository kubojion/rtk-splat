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
- A separate ROS 1 Rosario v2 adapter publishes rectified IR stereo, recorded
  aligned depth, complete dual-position PPK evidence, and an acquisition-bound
  RGB sidecar without exposing Rosario topics or calibration quirks to the
  mapping core.

The normalized 1,344-frame headland contract-v2 migration is complete and
hash-recorded. The modern all-frame GPU frontend registered all 2,688 stereo
images, and its Global pose plus 65k GS control reproduced the accepted result
within 0.081 dB masked and 0.115 dB corrected masked PSNR. The matched CPU arm
also registered every image but took about 52.6 times longer for feature
extraction. The dense/balanced/sparse adaptive-keyframe GS A/B is the remaining
unrun efficiency experiment; it is not a prerequisite if a full-field build
keeps all selected frames.

The Citrus and Rosario work is retained as transfer evidence, not as a new
production default. Citrus exposed a fixed-scale georeferencing failure and a
monolithic GS capacity failure. Rosario produced an accepted IR/depth pose and
GS, while its colour arms remain diagnostic because rolling-shutter RGB timing
and RGB-to-stereo registration are not independently validated.

The first bounded-map component is now implemented: a sealed, data-driven
TilePlan derives spatial cores and visibility context from metric depth and one
global pose artifact. It does not hard-code row count or time intervals. The
cached headland produced one automatic tile, while a forced two-tile seam plan
retained 919/879 training frames with 99.34%/99.36% measured core-support
coverage. Tile-aware cloud/training, exact core ownership, controlled scene
publication, and a metric-depth seam gate are implemented and have completed a
real GPU validation. The core-owned merged scene passed all 13 declared checks
and improved over a same-code monolithic control by 0.305 dB masked and
0.380 dB corrected masked PSNR over the same 168 held-out views. Its exact 1 m
seam band also improved by 0.158/0.092 dB. This validates bounded GS training
and deterministic stitching, not equal-compute superiority: the two tiles use
two independent 65k/2.5M-cap optimizations. The cached pose is
`legacy_unassessed`, so the scene is correctly provisional and creates no new
metric-georeferencing claim. See [TILED_SCENE.md](docs/methods/TILED_SCENE.md).

The 77-minute field path has now completed as a diagnostic milestone. It
processed 10,227 selected stereo frames, automatically planned and trained 32
tiles, and published one sealed layered scene. Every structural, visual,
ownership, fixed-scale/baseline, inventory, and seam check passed. The result
is visually useful, but its synchronized trajectory failed unchanged
independent absolute-RTK gates (366.483 mm held-out median versus 120 mm
allowed), so it remains `diagnostic_render_only` and makes no production
georeferencing claim. Current work separates the geodetic root-cause study
from a frozen-model distant-ground/horizon rendering study. See the
[full-field milestone](docs/milestones/FULL_FIELD_DIAGNOSTIC_V1.md) for exact
metrics and [SERVER_RUN.md](SERVER_RUN.md) for the historical execution
record.

A sealed layered scene can be exported to one portable Gaussian PLY without
loading all tile tensors into memory:

```bash
rtk-splat-scene-export \
  --source-scene /path/to/sealed/layered-scene \
  --destination /new/immutable/export-bundle
```

The PLY is the opacity-pruned, uniquely core-owned union for general 3DGS
viewers. It cannot reproduce the authoritative renderer's per-view depth/alpha
context blend, and it preserves the source scene's production or diagnostic
label.

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
        ↓                         ↓
sealed visibility TilePlan   single-scene path
        ↓
tile-aware cloud → GS train → core ownership → scene/seam validation
```

The core package under `src/rtk_splat/core/` imports no ROS, bag, dataset, backend,
or workflow module. ROS 2 ZED/u-blox, ROS 1 CitrusFarm, ROS 1 Rosario v2, and
AgriGS are adapters under `src/rtk_splat/adapters/`. The CitrusFarm implementation handles an ordered
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
- `robots/` owns stable topics, sensor geometry, pose-source, and adapter clock/
  chunk-contract facts;
- `sequences/` owns recording paths, windows, artifact names, and a narrow set
  of documented scene overrides; and
- runtime resolution derives data-dependent controls only after their inputs
  exist.

The runtime-derived controls are frame stride or metric spacing from measured
motion, stereo range from focal-length-times-baseline and reliable disparity,
initial-cloud capacity from the VRAM-bound growth budget, training iterations
from the actual number of training views, and Gaussian capacity from
initial-cloud size and available VRAM. `auto` opts into a named
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

ROS 1 bag ingestion, including CitrusFarm and Rosario v2, uses the separate
optional extra:

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

After a complete metric-depth segment and global pose exist, production tile
count and context can be derived rather than authored:

```bash
rtk-splat tiles-plan \
  --config configs/sequences/headland.example.yaml \
  --segment /path/to/depth-segment \
  --workdir /new/tile-plan-workdir \
  --pose-name balanced-gpu-global \
  --tile-plan-name full-field-visibility-v1
```

This stage writes only a small sealed plan and plot. `--tile-count` is reserved
for controlled seam A/B tests. Downstream `cloud` and `train` accept the paired
`--tile-plan`/`--tile-id` arguments without copying a segment, and
`scene-publish` verifies every tile before exact core-owned concatenation and
held-out evaluation. The cached headland overnight launcher is documented in
[TILED_SCENE.md](docs/methods/TILED_SCENE.md); it is experiment-specific rather
than the generic package interface.

Normal export remains fail-closed on held-out RTK gates. A rejected candidate
can be rendered only through an explicit, separately named diagnostic path:

```bash
bash scripts/runs/citrusfarm_05_13d_uturn.sh run \
  --workdir /home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_auto_v1 \
  --render-on-georef-failure
```

This does not loosen the 15 cm/30 cm gates. A failed result is labelled
`diagnostic_render_only`, propagates its failed RTK checks through pose, cloud,
and run provenance, and writes `splat.DIAGNOSTIC_ONLY.ply` rather than
`splat.ply`. It is suitable for visual inspection only and is not eligible for
a metric georeferencing claim. If a default run made with the same source and
launcher snapshot already stopped at pose export, add `--resume-existing` with
the same workdir and diagnostic option; the verified COLMAP stages are reused.
Runs made before this feature was added intentionally fail the source-identity
check and require a fresh workdir.

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

A completed checkpoint can be re-exported without training or replacing its
canonical PLY. The command verifies the sealed selected-model hash and writes
an atomic bundle under `<run>/exports/<name>/`:

```bash
python -m rtk_splat.workflows.export_splat \
  --source-run /path/to/completed/run \
  --export-name full-model-no-crop-v1 \
  --opacity-threshold 0 \
  --no-crop
```

This changes only the viewer PLY. Metrics already evaluated from `params.pt`
do not change.

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

## Rosario v2 cross-dataset pilot

The completed Sequence 5 experiment uses 140--250 s from the exact first-bag
epoch: 103.37 m, two opposing adjacent-row passes, and one complete U-turn.
The separate dual-M2 PPK bag is an explicit offline method input. PGT,
conventional GNSS, IMU, and wheel odometry are excluded.

The one-time immutable ingest is complete and validated on NVMe. It contains
**1,014** rectified IR stereo frames with recorded metric depth and **1,645**
separately sealed RGB observations. Real-image stereo, depth scale, CameraInfo,
static-TF, timestamp, physical antenna-baseline, and heading-sign gates pass.
The raw PPK covariance and the separately declared effective weighting prior
are both retained.

The all-frame Global run is also complete: all 2,028 stereo images registered
at 0.80256 px mean reprojection error, fixed-scale ENU export passed, and the
44,400-presentation IR/depth GS selected corrected masked PSNR 21.15047. Its
checkpoint contains 2,462,523 Gaussians. A separate no-crop/zero-threshold PLY
export keeps all of them without changing the checkpoint or metric.

The camera-to-primary-antenna URDF transform remains a rough prior. The RGB-D
transfer and matched v5/v6 training arms completed on 1,012 associations, but
remain `diagnostic_render_only`: free train-view corrections sharpened the
images without improving raw held-out PSNR. A later unconstrained colour-only
COLMAP probe folded geometrically, so it cannot establish a pose ceiling.
Exact evidence is in
[the Rosario pilot record](docs/experiments/ROSARIO_V2_SEQUENCE5_PILOT.md).

## CitrusFarm generic-policy transfer evidence

The supported CitrusFarm candidate uses the 192 s window at 543--735 s,
stereo RGB, computed SGBM depth, and single-antenna Piksi RTK. It deliberately
tests the generic `quality_v1` runtime policy: 0.10 m metric sampling, stereo
range from measured fB, iterations from the actual training split, and Gaussian
capacity from the cloud and available VRAM. Its launcher writes only to a new
internal-disk work directory, does not use `tmux`, and resumes only stages
whose complete layered-config/code/command evidence still matches:

```bash
bash scripts/runs/citrusfarm_05_13d_uturn.sh plan
bash scripts/runs/citrusfarm_05_13d_uturn.sh preflight
bash scripts/runs/citrusfarm_05_13d_uturn.sh run \
  --workdir /home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_auto_v1
```

The expected automatic choices on this machine are about 2,288 stereo pairs,
20 m maximum stereo depth, 65,000 iterations and approximately 2.46 million
Gaussians; these are estimates, not authored results. The exact values and
inputs are written by the stages. The real read-only preflight previously
measured a 228.726 m GNSS path, a 0.119885 m rig baseline and +72.548749 ms
camera-to-GNSS clock correction. Recorded ZED depth/confidence are inventoried
but withheld from this arm; SGBM publishes a new immutable derived segment.

A 4 s real-data smoke published 10 stereo pairs, computed SGBM depth, and
sealed a 20-image GPU frontend with all 52 requested pairs verified. The first
full-window **frozen 0.15 m** attempt later published 1,495 stereo pairs and
registered all 2,990
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
directly comparable CitrusFarm score. Its complete settings remain unchanged
in `configs/reproductions/citrusfarm_05_13d_uturn_v2.yaml`.
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

The later 0.10 m generic-auto full-window reconstruction registered all 3,694
images but failed fixed-scale georeferencing and its monolithic GS degraded as
it exhausted the laptop's capacity. A bounded 330--410 s same-corridor retrace
then registered all 1,576 images, passed its fixed-scale gate (0.117 m median /
0.137 m p95), and completed a deliberately diagnostic GS at 21.198 dB masked /
22.632 dB corrected masked PSNR. This is evidence for bounded spatial training,
not an accepted Citrus metric model or a cross-scene comparison with headland.

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
The migration itself proves only the conversion boundary. A separately named
modern all-frame COLMAP/GS arm later reproduced the golden quality without
modifying this segment.

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
- The modern all-frame frontend has reproduced the headland reference; adaptive
  keyframe density is implemented but has not completed a matched GS A/B.
- Global Mapper is visual. The generic geodetic-submap sidecar consumes sealed
  raw-GNSS priors through the pinned pose-prior mapper, and the full submap
  assembly is implemented. The first complete synchronized field trajectory
  still fails independent absolute-RTK production gates.
- Visibility-bounded full-field GS tiles, exact ENU ownership, arbitrary-tile
  orchestration, automatic planning, layered publication, and no-monolith
  evaluation are implemented and have completed on all 10,227 selected field
  frames. The 32-tile result is diagnostic because of geodetic failure; the
  unsupported distant-ground/horizon compositor also needs improvement.
- CitrusFarm has strong visual reconstructions, rejected full-window
  georeferencing controls, and a completed bounded diagnostic retrace; it has
  no accepted metric GS.
- Rosario v2 has a complete production fixed-scale visual pose and IR/depth GS,
  plus completed colour diagnostics, but no independently validated RGB
  timing/extrinsic solution or production-eligible colour GS.
- Right-camera GS photometric supervision and learned stereo depth are not
  validated.

See [TODO.md](TODO.md) for the short active plan and [PROGRESS.md](PROGRESS.md)
for detailed evidence.
