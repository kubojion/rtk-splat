# RTK-Splat Progress

Last updated: 2026-08-11

This is the current source of truth for this repository. Superseded plans are
kept under `docs/archive/`.

## Repository cleanup status

The production-boundary cleanup is complete and did not alter any work
artifact or accepted metric:

- `9da4bb8` froze the pre-cleanup evidence and tag
  `pre-cleanup-20260802`;
- `92baca0` added strict profile -> robot -> sequence ownership, auditable
  runtime-derived controls, the fail-closed resolved-config ledger, and removed
  the real adapter/backend/diagnostic coupling; and
- `7acd7b6` moved the sole installed namespace to `src/rtk_splat`, made
  provenance and golden verification wheel-safe, and updated every supported
  launcher;
- `63b8a36` documented the final cleanup/runtime ownership; and
- `abfa4c1` made the active Citrus controls data-derived and auditable.

The current source-tree gate passed **364 tests plus 35 subtests** on
2026-08-11, including memory-bounded ingest, portable transfer, isolated
server-environment bootstrap, RTX 4090 preflight support, sealed TilePlan,
tile execution, production scene publication, server-launcher,
scientific-control, and source-mutation gates. At the earlier cleanup freeze,
the historical incremental and
Global external verifiers passed **38/38** and **42/42** exact checks. After the
2026-08-09 workspace cleanup, the incremental artifact still passes 38/38, but
the historical reduced-Global compact pose artifact is no longer present, so
that external verifier currently passes only 6/20. Its recorded metrics remain,
and the complete modern AB03 GPU pose/model control is intact. A clean wheel
installs only the `rtk_splat` namespace and works outside a Git checkout.

The only active tracker is [TODO.md](TODO.md). The present objective is a
seam-validated, georeferenced tiled reconstruction of the full 77-minute field
recording. Rosario colour research is frozen while that objective is pursued.

## Current reference result

The reference experiment is the 450-second headland sequence containing the end
of row 5, a U-turn, and the entry to row 1.

- 1,344 calibrated stereo pairs at approximately 3 Hz.
- 0.119846 m measured rectified stereo baseline.
- 65,000 GS iterations and a 2.5 million Gaussian cap.
- The raw-RTK and stereo-BA arms used the same SGBM depth, split, and GS
  settings. Pose backend and pose-derived initialization cloud were the
  material changes.

| Metric | `tile_turn2`, RTK | `tile_turn3_stereo_ba` | Change |
|---|---:|---:|---:|
| Masked PSNR | 20.6097 | **24.4449** | **+3.8352 dB** |
| Corrected masked PSNR | 21.1519 | **25.8480** | **+4.6960 dB** |
| SSIM | 0.3209 | **0.5944** | +0.2736 |
| Corrected LPIPS | 0.5333 | **0.3187** | −0.2146 |
| Test-pose-aligned masked PSNR | 20.9356 | **24.4383** | — |

Test-time pose alignment provides no gain after stereo BA, supporting the
conclusion that the refined trajectory is already locally consistent for this
reconstruction.

## Rosario v2 cross-dataset pilot: production pose and IR GS complete

The quality-first Sequence 5 pilot covers 140--250 s relative to the exact
first main-bag log timestamp. It contains two opposing traversals on adjacent
rows and one complete U-turn over 103.37 m. The separately distributed Reach
1/Reach 2 PPK bag is an explicit offline method input; PGT, conventional GNSS,
IMU, and wheel odometry are excluded.

The new ROS 1 Rosario adapter passed a real fail-closed preflight and published
one immutable segment on 2026-08-03:

- **1,014** rectified IR stereo frames with recorded metric depth over
  **109.812 s**, sampled at **0.10144 m** effective spacing;
- **1,645** full-resolution RGB observations in a separate, acquisition-bound
  sealed artifact; and
- complete raw/effective covariance, status, timestamps, and both antenna
  position streams. All **569/569** interpolated dual-position headings pass
  the physical-baseline gate.

The recorded IR baseline is **50.211037 mm**, versus **50.240891 mm** in the
official Kalibr files. Real-image epipolar checks pass the predeclared gates:
median sample-p95 vertical residual **0.8618 px**, aggregate fraction below
1 px **0.9692**, and at least **1,473** geometric matches per probe. Recorded
depth also passes: disparity disagreement p95 is at most **0.5624 px**,
depth/left timestamps are exactly equal, CameraInfo dimensions/K/P/R/D match
exactly, and the recorded static depth-from-left transform is **0 m / 0 deg**.

The offline PPK file does not expose carrier solution state and its recorded
covariance is a static placeholder. Raw `[1, 1, 4] m^2` diagonal covariance is
retained; a separately labelled, conservative `[0.10, 0.10, 0.20] m` effective
sigma is used for weighting only and is not an accuracy claim. The physical
dual-antenna baseline is internally consistent, but the camera-to-primary-
antenna URDF transform remains a rough prior pending the post-SfM metric-
integrity audit. The online Reach streams fail their physical-baseline gate and
cannot silently replace the PPK input.

Ingest took **13 min 48.81 s** over USB 2.0, peaked at **2.63 GB RSS**, and
published 1.77 GB canonical plus 2.67 GB RGB artifacts on NVMe. The sealed
downstream run then completed on 2026-08-04 without replaying the bags:

- Global Mapper registered **1,014/1,014 frames and 2,028/2,028 images** with
  **0.80256 px** mean reprojection error and **38.679** mean track length. Its
  solve took **400.5 s** and peaked at **5.87 GB RSS**.
- Fixed-scale ENU export passed as a production artifact. Held-out RTK-camera
  residuals were **0.16186 m median / 0.31344 m p95**, with 100% absolute-cap
  inlier support. The diagnostic Sim(3) scale was **1.008414** and was recorded
  but not applied. These are consistency residuals against the declared PPK
  prior, not survey-ground-truth camera accuracy.
- The IR/depth GS completed **44,400** presentations in **3 h 15 min**. The
  selected checkpoint is step 44,000 with corrected masked PSNR **21.15047**,
  and contains **2,462,523 Gaussians**.

The original viewer PLY retained 758,882 Gaussians after the configured 0.05
opacity threshold and spatial crop. A new non-destructive completed-checkpoint
export retained all **2,462,523** Gaussians with threshold 0 and no crop at
`runs/rosario-seq5-ppk-140-250-gs-v1/exports/full-model-no-crop-v1/splat.ply`.
It is 226,552,695 bytes; the source `params.pt`, original PLY, georeferencing,
and **21.15 dB** metric remain unchanged.

The RGB-D transfer subsequently published **1,012** diagnostic RGB/depth views
(886 train / 126 validation). Its inherited RGB pose/timing state is not
independently validated, so all colour outputs remain
`DIAGNOSTIC_ONLY` and metric-claim ineligible. The matched v5 control selected
step 20k at 20.066 masked / 20.340 corrected masked PSNR, SSIM 0.4108, and
LPIPS-CC 0.4721. Free per-training-view pose optimization in v6 made the image
visibly sharper and improved LPIPS-CC to 0.4336, but reduced the raw held-out
scores to 19.847 / 20.259 dB and SSIM 0.4093. Its small unregularized deltas
(median about 5.6 mm and 0.15 degrees) are diagnostic photometric adjustments,
not calibrated or georeferenced poses.

A 250-frame unconstrained monocular colour COLMAP probe registered every frame
at 1.079 px sparse reprojection, but its trajectory folded: global Sim(3)
camera-centre RMS was 5.807 m, one adjacent step rotated 178.8 degrees, and
independent temporal blocks implied incompatible scales. It rejects that
unconstrained monocular solve, not colour-aware bounded refinement in general.
No more Rosario work is scheduled for the current full-field milestone.
Complete evidence is in
`docs/experiments/ROSARIO_V2_SEQUENCE5_PILOT.md`.

## Pose reconstruction evidence

- Registered left frames: **1,344 / 1,344**.
- Registered total images: **2,688 / 2,688**.
- Visual-to-RTK alignment residual:
  - median: **5.11 cm**
  - p95: **10.72 cm**
  - maximum: **13.80 cm**
- Exporter Sim(3) scale: **0.985794**.
- Mean COLMAP reprojection error: **0.818 px**.
- COLMAP mapper runtime: approximately **581 minutes**.
- Complete feature, matching, mapping, and GS experiment: **15 h 16 min**.

The 5.11 cm value is a disagreement between the aligned visual camera trajectory
and the RTK-derived camera trajectory. It is not an independent measurement of
RTK accuracy and cannot alone distinguish RTK noise, rough TF, clock offset, or
visual deformation.

The run began while the stereo implementation was uncommitted. Git commit
`3f86826c151ffffab8932e6211bbe933108a9983` is the first matching post-run
source snapshot, not a falsely claimed clean run-time checkout. Exact hashes
and artifact statistics are in
[headland_stereo_ba.json](docs/experiments/golden/headland_stereo_ba.json).

## CitrusFarm full-window pose attempt

The first complete 543--735 s Citrus pose attempt reached the held-out export
gate on 2026-08-01. It did not reach cloud construction or GS training.

- Published **1,495 stereo pairs** over approximately 229 m.
- Registered **2,990 / 2,990 images** in one visual Global Mapper model.
- Mean reprojection error: **0.973 px**; mean track length: **30.43**.
- Diagnostic fixed-stereo Sim(3) scale: **1.001315**; scale was not applied.
- Measured wall time from ingest start to the rejected export: about **42 min**.

The original absolute-only export error was replaced by a covariance-normalized,
temporally held-out report. It uses full stored 3x3 position matrices and a
chi-square residual diagnostic, while retaining independent absolute metre and
support caps. Normalized checks are not authoritative for this recording because
its NavSatFix covariance is a static, driver-configured `APPROXIMATED` matrix and
the current camera-centre covariance still omits heading-times-lever-arm, timing,
visual-pose, and alignment-fit uncertainty.

The cached solve still fails honestly:

| Held-out quantity | Result | Gate |
|---|---:|---:|
| Median position residual | **0.2663 m** | <= 0.15 m |
| Absolute inlier support | **59.20%** | >= 80% |
| Inlier p95 | **0.3151 m** | <= 0.30 m |
| Median Mahalanobis squared (diagnostic) | **9.2416** | <= 4.1083 |
| 95% chi-square coverage (diagnostic) | **40.13%** | >= 80% |

The two untouched temporal blocks differ materially: **0.5135 m** versus
**0.1937 m** median. This is not explained by a single global scale or by simply
calling the receiver "worse RTK". No pose or staging artifact was published;
the immutable rejection report is retained beside the backend evidence.

A separate receiver-state audit decoded the bag-embedded Piksi message without a
ROS installation. It associated **1,920 / 1,920** NavSatFix samples within
10.115 ms to **2,496** receiver states. Every state reports `FIXED_RTK`, agrees
with `rtk_mode_fix=true`, and has no receiver error flag. Future ingest now
preserves this complete evidence and fails closed when it is required. The old
sealed v1 segment remains truthfully labelled from its retained evidence and is
not relabelled after the fact; a new immutable ingest is required for a
publication-grade receiver-confirmed reproduction.

### Optional RTK pose-refinement A/B

A generic, separately named `rtk_refinement` backend was implemented and run
against the cached Citrus v1 reconstruction on 2026-08-01. It cloned the
backend database, physically removed alternating temporal holdouts, and ran
COLMAP 4.1.1 pose-prior refinement with frozen cameras and stereo rig. Exactly
**897 calibration priors** entered optimization; **598 priors** remained
absent from the refinement factors and alignment fit. The source backend was
not modified. This is a refinement-factor holdout, not an end-to-end
GNSS-independent split: upstream RTK-guided frame/pair planning had access to
the full track.

The refinement was safe but did not pass:

| Quantity | Visual source | RTK refinement | Gate |
|---|---:|---:|---:|
| Registered images | 2,990/2,990 | **2,990/2,990** | exact |
| Reprojection error | 0.9727 px | **0.6744 px** | <= 2.0 px |
| Held-out RTK median | 0.2663 m | **0.2490 m** | <= 0.15 m |
| Held-out RTK support | 59.20% | **67.22%** | >= 80% |
| Held-out inlier p95 | 0.3151 m | **0.3034 m** | <= 0.30 m |
| Source-to-refined scale | -- | **1.003986** | within 0.5% |
| Maximum stereo-baseline change | -- | **1.4e-14 m** | <= 5e-5 m |

The 31.7-minute COLMAP solve peaked at **6.94 GB RSS**. It retained the
0.119885166 m rig baseline to numerical precision and changed no intrinsic or
rig parameter. It also retriangulated far more short tracks: reprojection and
observation count improved, while mean track length fell from 30.43 to 6.84.
That change remains visible and fails the deliberately conservative relative
track-retention gate.

More importantly, the refinement-factor holdout RTK gates fail even without that track
gate. Per-block analysis explains the limited gain. The difficult first
calibration/holdout pair improved from **0.622/0.514 m** median to
**0.517/0.397 m**, while the later held-out block was essentially unchanged
(0.194 to 0.191 m). Stock Cauchy pose-prior BA therefore helped but treated a
large coherent early trajectory disagreement too weakly to correct it. No pose,
cloud, GS model, or rendering claim was published.

The compact rejected-run receipt, including sealed artifact hashes, exact
resource use, and the predeclared pass/fail values, is
`docs/experiments/citrusfarm_rtk_refinement_v1.json`.

### Completed RTK-loss and fresh-initialization diagnostics

The cached quadratic/L2 arm completed without replaying bags, depth, features,
matching, Global Mapper, or GS. It used the same covariance whitening,
897/598 calibration/holdout split, seed, iteration limits, frozen cameras, and
frozen stereo rig as the Cauchy control. Its 32.3-minute solve registered all
2,990 images and improved held-out median error from 0.2490 m to **0.2284 m**
and support from 67.22% to **95.32%**. It still failed the 0.15 m median and
0.30 m inlier-p95 gates (0.3175 m), as well as the continuation-only relative
track-retention gate. It therefore remains rejected and produced no pose,
cloud, GS run, or PLY.

The sealed fresh-initialization arm then completed on 2026-08-02. It reused the
same cached frontend, private-prior protocol, quadratic loss, 897/598
factor/holdout split and fixed stereo rig as the continuation control. Its only
meaningful solver change was removing `--input_path`, so RTK calibration-block
factors were active while `pose_prior_mapper` built the reconstruction.

The candidate registered **2,990/2,990 images**, with **0.683789 px** mean
reprojection error, 6.163 mean track length and 2,229 mean observations per
image. Held-out RTK median improved from the L2 continuation's 0.22838 m to
**0.19595 m**, support improved from 95.32% to **99.16%**, and inlier p95
improved from 0.31752 m to **0.31138 m**. It nevertheless failed the unchanged
0.15 m median / 0.30 m p95 absolute gates and the 0.5% trajectory-scale gate
(source-relative scale **1.007524**). It is therefore a useful negative result,
not an accepted pose. No pose export, cloud, GS run or PLY was started.

The compact paired receipt is
`docs/experiments/citrusfarm_pose_prior_initialization_ab_v1.json`. The result
shows that stock whole-model pose-prior mapping is not sufficient; the next
method-level pose experiment is bounded geodetic stereo submaps, not another
loss or initialization sweep.

### Explicit diagnostic-render continuation

An opt-in visualization path is now implemented for candidates that pass exact
stereo registration and visual-quality checks but fail the final held-out RTK
georeferencing gates. The normal path remains fail-closed and the Citrus gates
remain **0.15 m median / 0.30 m inlier p95**.

`--render-on-georef-failure` selects separate pose and training names. A failed
result is sealed as `diagnostic_render_only`, propagates the failed status and
source hashes through pose, cloud, run, and PLY evidence, and exports only
`splat.DIAGNOSTIC_ONLY.ply`. Cloud and training require the explicit permission
again. The golden/publication verifier rejects diagnostic runs, mismatched
sidecars, empty or hash-mismatched PLYs, and failed markers. This path permits
local visual inspection only; it does not make a failed map eligible for a
metric georeferencing claim.

The real full-window diagnostic continuation subsequently reached GS. Its
failure and the bounded follow-up are recorded below; it produced no completed
parameter artifact or PLY.

### Generic-auto full-window GS failure and bounded follow-up

The fresh 543--735 s generic-auto run published **1,847 stereo pairs** and
registered **3,694/3,694 images**. Its visual reconstruction was locally
strong: **1.001269 px** mean reprojection error, **37.895351** mean track
length, and diagnostic Sim(3) scale **1.000721**. Fixed-scale georeferencing
still failed honestly at **0.293749 m** held-out median, **0.315424 m** inlier
p95 and **57.45%** support. Diagnostic-render authorization retained that
failure label and allowed GS to start without turning it into a metric claim.

The monolithic 227 m GS then degraded monotonically and was stopped after the
14,000-step evaluation:

| Step | Masked PSNR | Corrected masked PSNR | Near PSNR | SSIM | LPIPS-CC |
|---:|---:|---:|---:|---:|---:|
| 2,000 | **18.45** | **19.01** | **19.44** | **0.556** | **0.561** |
| 14,000 | 15.43 | 16.99 | 15.98 | 0.508 | 0.608 |

The source frames and COLMAP poses remained usable. The immediate failure was
the GS resource/optimizer interaction: a four-million-point cloud was clipped
at initialization to the measured **2,462,524-Gaussian** VRAM cap, leaving no
growth headroom; dense regularization and relocation operated across a long,
sparsely visible scene; and upstream MCMC continued injecting position noise
after its nominal refinement cutoff. The interrupted implementation saved only
diagnostic render JPEGs, not `params.pt`, metrics, or a PLY.

The generic correction now derives the initial-cloud ceiling from the same
measured VRAM policy. On this 7.656 GiB GPU it resolves to **820,841 points**,
allowing growth toward the safe **2,462,524** final cap rather than beginning
at it. Training atomically retains the earliest best held-out
`psnr_masked_cc` checkpoint, restores it for final evaluation/PLY export, and
stops the third-party MCMC callback at `refine_stop_iter`. Interrupted runs
retain an explicitly incomplete checkpoint but cannot pass completion gates.

The bounded same-corridor retrace at **330--410 s** then completed: **788**
stereo frames / **1,576** images, all registered, at **0.8793 px** mean
reprojection. Fixed-scale georeferencing passed at **0.1172 m** held-out median
and **0.1373 m** p95. Because the launcher explicitly requested diagnostic
rendering, the artifact correctly remained metric-claim ineligible rather than
silently promoting itself. Its GS selected step 34,500 at **21.1975** masked /
**22.6319** corrected masked PSNR, SSIM **0.6349**, and LPIPS-CC **0.4598**.
This supports bounded visibility/capacity as the practical full-field route;
it is not directly comparable to headland PSNR because the scene and split
differ.

## Contract-v2 work: Phase 0 accepted baseline

The two accepted headland results are now frozen independently:

- Incremental stereo BA: pose fingerprint `701d917f...`, masked PSNR
  **24.444888**, corrected masked PSNR **25.847967**.
- Reduced Global Mapper: pose fingerprint `8129ff17...`, masked PSNR
  **24.458305**, corrected masked PSNR **25.841772**.

Their exact compact-file hashes, complete reported metrics, mapper-specific
scale fields, and acceptance gates are recorded in
`docs/experiments/golden/`. The read-only verifier passes **38/38** checks for
the incremental result and **42/42** for the Global result. At the Phase 0
freeze, the then-current full suite passed **55 tests**. A repository-only
regression test freezes the key hashes and quality values without requiring
the external work directory in CI.

## Metric-integrity audit

The strict audit is non-destructive and is not consumed by training.

Trusted corrections:

- Optical-camera lever Z: **−0.03862885 m**.
- Antenna-baseline tangent mode 0: **+0.01502952 rad (+0.8611°)**.

Retained at their rough priors because the data did not support correction:

- Lever X and Y.
- Second baseline tangent mode.
- Clock offset.
- Rotation about the physical antenna baseline, which one baseline cannot
  observe structurally.

Held-out aggregate:

| Metric | Prior | Retained correction |
|---|---:|---:|
| Antenna median | 5.65 cm | **3.71 cm** |
| Antenna RMS | 5.70 cm | **4.58 cm** |
| Antenna p95 | **7.63 cm** | 7.84 cm |
| Baseline-angle median | 0.877° | **0.243°** |

The p95 position residual worsened by 2.8%, within the configured audit gate.
That tradeoff remains visible. These are calibration recommendations, not
enabled training parameters.

The same strict audit was run against the reduced Global Mapper's raw metric
trajectory on 2026-07-31. It used the same exact RTK fixes, covariance/status,
dual-antenna vectors, rough physical TF, temporal holdouts, and fixed stereo
scale as the incremental audit:

| Raw metric trajectory | Diagnostic visual-to-ENU scale |
|---|---:|
| Incremental COLMAP | 0.996173 |
| Reduced Global Mapper | 0.995212 |

The two strict estimates differ by only **0.096 percentage points**. Directly
aligning the two raw visual trajectories gives a relative scale of 0.998845,
with 6.5 mm median trajectory disagreement. Global Mapper also preserved the
encoded 0.119846250 m stereo rig baseline exactly. Therefore the earlier
0.984637 Global diagnostic did not identify a 1.5% Global-BA scale failure: it
was mostly confounded by the rough RTK-camera-centre alignment used by the
exporter. The remaining shared 0.38--0.48% diagnostic discrepancy cannot be
assigned specifically to the physical stereo baseline without an independent
target-based stereo calibration.

Only the antenna-baseline tangent correction was retained for the Global
trajectory. Lever and clock corrections were not observable/stable enough and
remain at their priors. The new diagnostic artifact is
`rtk_stereo_metric_integrity_global_v1_strict`; it publishes no poses and is
not consumed by training.

## Completed implementation: Phases 1, 3, and 4, plus Citrus preparation

The following describes code and test status. It does not imply that a new
COLMAP or GS experiment has completed.

### Phase 1: contract and repository boundary

Implemented and covered by unit/synthetic tests:

- The reusable `src/rtk_splat/core/` package is isolated from ROS, adapters, frontends,
  backends, diagnostics, and orchestration.
- `rtk_splat` is now the only installed top-level Python namespace. Adapters,
  frontends, backends, workflows, and diagnostics are explicit subpackages;
  the former collision-prone global package names no longer exist.
- Dataset-specific ingestion lives under `src/rtk_splat/adapters/`; the registry
  supports ROS 2 ZED/u-blox, ROS 1 CitrusFarm, ROS 1 Rosario v2, and AgriGS.
- Calibration ROS topics, u-blox message registration, message decoding, and
  bounded MCAP reading moved into `src/rtk_splat/adapters/calibration_bag.py`.
  Diagnostics retain only plain records, numerical analysis, COLMAP parsing,
  and artifact reporting, and import without the optional ROS stack.
- Immutable contract-v2 segments preserve exact nanosecond timestamps, full
  GNSS covariance/status, both stereo calibrations, explicit transform
  semantics, and the complete optional dual-antenna N/E/D baseline.
- Strict configuration resolves profile -> robot -> sequence. Data-dependent
  frame sampling, stereo range, training duration, and Gaussian capacity are
  derived at the stage that can measure their inputs, logged with provenance,
  and remain explicitly overridable. There is no local-machine default
  configuration.
- The supported Citrus sequence now contains **14 authored leaves**, down from
  54: unused evaluation paths, duplicated bag counts, frozen artifact names,
  and an unrelated RTK-refinement experiment were removed. Stable ROS1 chunk/
  clock policy moved to the robot profile; spacing, depth, iterations, and
  Gaussian capacity use the logged runtime rules. The historical 0.15 m run is
  unchanged at SHA-256 `5649e840d8af4189ec1a270c0873d249c23fc4a48b6e8f9f5d60b3bb88a8432e`.
- The strict normalized v1-to-v2 migration completed at
  `~/agromap4d_work/field_turn_contract_v2_normalized/segment` in 2.85 s
  (66.5 MiB maximum RSS). It validates 1,344 frames, preserves 2,273 raw GNSS
  and 2,273 raw heading samples, and retains 1,344 valid RTK-fixed position
  rows plus 1,344 valid dual-antenna headings. The hash receipt is
  `docs/experiments/migrations/headland_contract_v2.json`.
- SGBM depth derivation publishes a separate immutable v2 segment with
  symlinked source images; it cannot mutate the ingested segment.
- Generated caches and obsolete root scripts were removed, configurations and
  documentation were grouped, and existing golden artifacts remain
  read-only.

No input bag, historical work artifact, COLMAP reconstruction, or GS run was
modified by this refactor.

### Phase 2: deliberately skipped

Independent target-based stereo calibration was not performed. The current
production calibration was not changed. The remaining shared 0.4--0.5%
metric-scale uncertainty therefore remains an explicitly documented
limitation.

### Phase 3: mapper-neutral frontend and isolated backends

Implemented and unit-tested:

- `frontend-build`, `frontend-features`, `frontend-rig`,
  `frontend-priors`, and `frontend-match` create one sealed frontend containing
  symlinked canonical images, exact frame manifest, calibrated rig, features,
  filtered Cartesian RTK priors, verified pairs, quality, resolved
  configuration, and provenance.
- Global and incremental mappers each receive a separate, hash-verified,
  transaction-consistent database snapshot that includes committed WAL state.
  Global is the default candidate; incremental is the fallback/control.
- The terminal seal covers finalized JSON/pair evidence, the logical and raw
  database, and image-link targets plus contents; backend stages reverify it.
- `backend-prepare`, `backend-solve`, `backend-register`,
  `backend-quality`, and `backend-export` use verified resumable boundaries.
- The solve may use only keyframes, but image registration must recover the
  exact left and right image for every canonical timestamp.
- Pose publication is fail-if-existing and atomic. It includes matrices,
  centres, IDs, exact timestamps, names, alignment diagnostics, quality, and
  provenance.
- Export fits robust fixed-scale SE(3) alignment on alternating contiguous
  calibration blocks and enforces RTK gates on untouched blocks. Sim(3) is fit
  on calibration data only, is diagnostic only, and cannot silently rescale
  metric stereo geometry.
- The canonical `mapper:` profile now exposes the exact bounded Global Mapper
  controls used by the accepted reduced solve: three BA iterations, 60,000
  tracks maximum, 1,000 retained tracks per view, no final retriangulation,
  and CPU global-position/BA solving. Obsolete `global_mapper:` plans are
  rejected instead of being interpreted under new defaults.
- Each real Global solve runs in an isolated process group. It records
  per-attempt logs and CSV/JSON resource samples, lowers process priority, and
  terminates the group without publishing a pose when configured memory or
  disk floors are crossed. The incremental fallback path is unchanged.

The retained database position priors are evidence and can assist compatible
COLMAP behavior. The integrated Global Mapper solve remains visual and does
**not** optimize covariance-weighted RTK factors in its bundle adjustment.

### Phase 4: adaptive keyframes and RTK-guided pairs

Implemented and unit-tested:

- `all`, `dense`, `balanced`, and `sparse` keyframe presets;
- translation, rotation, elapsed-time, image-quality, and turn-aware
  selection;
- temporal/metric neighbors, view-direction and revisit links;
- mandatory same-timestamp stereo pairs; and
- attachment edges for every non-keyframe image; and
- a required physically admissible bounded-degree connected solve graph with
  fail-closed component diagnostics.

The real headland GPU/CPU feature and all-frame pose A/B is complete. Both arms
registered **2,688/2,688** images and passed fixed-scale gates. GPU feature
extraction took 77 s versus 4,053 s for CPU (about **52.6x faster**) and had
slightly lower mean reprojection error (1.2129 versus 1.2753 px). The GPU
all-frame Global pose then completed a matched 65k GS control at **24.3639 dB**
masked and **25.7326 dB** corrected masked PSNR: only -0.0810/-0.1153 dB from
the accepted reference and inside the predeclared 0.3 dB gate.

Only the dense/balanced/sparse adaptive-keyframe GS A/B remains unrun.
Phase 4 keyframe reduction is therefore implemented and planned, but not yet
demonstrated as a quality-preserving speed improvement.

One bounded planning-only smoke used the normalized real headland segment with
the `balanced` preset and image-quality decoding disabled. It completed in
3.73 s, selected 367/1,344 solve frames, planned 6,806 image pairs, retained
1,344 mandatory stereo pairs, and reported a connected 777-edge solve-frame
graph with maximum degree 8 plus registration paths for every frame. It did
not create a COLMAP database, terminal frontend seal, pose, cloud, or GS run;
it verifies real-data planning and artifact publication only.

Bounded RTK-factor submaps and rendering-quality changes remain outside this
implementation. A mapper-neutral, position-prior RTK refinement sidecar is now
implemented and real-data tested, but it failed the Citrus acceptance gates.
CitrusFarm source support is implemented, has passed a
real read-only preflight and bounded ingest/SGBM/sealed-frontend smoke, and has
completed one full-window visual reconstruction. That attempt failed the
held-out RTK export gate, so it produced no accepted pose or GS result.

The dataset-neutral core meets its enforced 2,000-line budget exactly.

## CitrusFarm ROS 1 candidate

The new `ros1_citrusfarm` adapter keeps dataset behavior at the ingestion
boundary. It reads an explicit ordered chain of 27 ZED ROS 1 bags and two
Piksi/base bags without requiring ROS, validates topics and chunk continuity,
decodes raw rectified stereo to lossless PNG, preserves sensor and bag-log
timestamps plus complete `NavSatFix` covariance/status, estimates observable
single-antenna course heading, and samples frames by travelled distance. Robot
geometry and topic names are configuration facts, not mapper conditionals.

The prepared sequence profile covers 543--735 s relative to the first GNSS
bag-log timestamp. Real read-only preflight evidence is:

- adapter preflight: **127.3 s**, **156.8 MB** reported maximum RSS;
- GNSS path length: **228.725603 m**;
- 0.15 m metric sampling estimate: **about 1,525 stereo pairs** (the ingest
  estimate; the completed ingest published **1,495 pairs**);
- recorded rectified stereo baseline: **0.119885166 m**;
- estimated camera-to-GNSS header correction: **+72.548749 ms**;
- final per-sequence configured correction: **+72.548749 ms** (zero residual);
  and
- estimated clock drift across the window: **-2.580854 ms**.

Those numbers describe the completed frozen 0.15 m experiment. The next
supported transfer run authors `frame_spacing_m: auto`; `quality_v1` resolves
that to a 0.10 m target (about 2,288 pairs from the measured path), while fB,
training-view count, initial-cloud size, and detected VRAM resolve the other
three automatic controls. It keeps the validated all-frame Citrus topology so
adaptive-keyframe behavior is not confounded with configuration cleanup.

A fresh read-only adapter preflight of that cleaned configuration passed on
2026-08-02: 228.725603 m produced an estimate of 2,288 pairs, the 0.119885166 m
baseline and +72.548749 ms correction were recovered, all 9,875 decoded
receiver states were `FIXED_RTK`, and the worst 11.061 ms chunk gap / 0.585 ms
overlap were well inside the new 100 ms / 10 ms platform gates. No artifact was
created. The complete launcher currently stops before bag payload reads or
decoding because the internal disk has 51 GiB free, below its unchanged 65 GiB
safety floor.

The configured offset and measured drift pass the configured 15 ms gates. The
configured camera/GPS transform composes the published UCR camera/lidar/GPS
calibrations; because one receiver supplies no orientation measurement, its
GPS axes are explicitly assumed body/lidar-aligned and the translation prior
retains 5 cm uncertainty per axis.

Recorded ZED depth and confidence are inventoried, but the primary experiment
does not consume them. It derives SGBM depth into a separately named immutable
segment so the pose/depth protocol remains stereo RGB + RTK and matches the
headland method boundary. LiDAR, IMU, ZED pose, and wheel odometry are not
method inputs for this candidate.

The Buffalo SSD is usable through the stable USB 2.0 connection. A complete
4.25 GB bag read finished at **44.4 MB/s** with no new kernel reset or I/O
errors, and sampled reads across the required later ZED chunks also passed.
The workflow treats the source bags as read-only and directs all generated
artifacts to the internal disk. This validates storage transport, not the
dataset contents.

A real bounded smoke then exercised the data path rather than only its
preflight. The 4 s interval at 543--547 s used 0.50 m spacing, read 39 stereo
candidates, and atomically published 10 pairs over 4.630 m in **135.7 s** with
about **184 MB** peak RSS. Left/right header residual was exactly zero, all
camera and bag-log timestamps were retained, and the largest selected GNSS
association residual was 43.0 ms. The first attempt failed closed on nested
calibration-provenance serialization; the writer published no segment, the
conversion was fixed recursively, and a regression test now covers it.

The successful segment was derived through SGBM and a real COLMAP 4.1 GPU
frontend. It retained 20/20 images in one calibrated stereo rig, extracted
198,786 descriptors, inserted all 10 covariance-aware RTK priors, and verified
all 52 requested pairs (36,133 verified correspondences). Depth plus the
sealed frontend completed in **10.5 s** and peaked at about **512 MB**. SGBM
valid coverage was 65.0--73.4% per frame (69.0% median), with a 2.30 m median
valid depth across frames. This
short, almost linear slice is intentionally not used to judge Global Mapper
or fixed-scale georeferencing observability, so no mapper or GS smoke is
claimed. The current repository-wide cleanup receipt is recorded above.

The public Citrus launchers are thin run specifications over one private shared
driver. Both provide plan, preflight, fresh run, and evidence-checked resume
modes without `tmux`. They run ingest, immutable SGBM depth, the all-frame GPU
frontend, bounded Global Mapper, all-frame registration and held-out quality
gates, pose export, cloud construction, and runtime-derived GS training. The measured
first attempt took about **42 min** to the export gate: 9m53s ingest, 3m29s
SGBM, 2m49s frontend build/features/rig/priors, 13m39s matching, and 11m18s
backend preparation/solve/registration/quality. The later generic-auto
full-window diagnostic GS exposed the optimizer failure documented above.

A full-window Citrus ingest, COLMAP reconstruction, rejected RTK-refinement
controls, and an incomplete diagnostic GS trajectory now exist, but no accepted
pose artifact or completed Citrus GS/PLY has been produced. In
particular, the headland 24.8 dB result cannot be promised or directly compared
across a different scene, camera, motion profile, and held-out image
distribution.

## Historical Global Mapper golden result

The following measured result predates the current sealed frontend. It reused
the successful incremental reconstruction's completed feature/match evidence
and is preserved as the speed/quality control.

The first full-track Global attempt reached Ceres global-position setup, grew
to **15.7 GiB** RSS, and held available memory below the configured **4 GiB**
floor. The safety monitor terminated it after **138 s**; the source database
was unchanged and no pose was published.

The separately named reduced profile retained all 53,058 verified pairs for
rotation averaging, targeted 1,000 selected long tracks per image, capped the
set at 60,000 tracks, and disabled final retriangulation so COLMAP could not
recreate the unbounded point set.

That successful resource profile is now represented directly by
`MapperConfig` and the reproduction YAML rather than by mismatched legacy flag
names. A read-only held-out replay of the accepted 2,688-image model using the
current five temporal blocks and exact 1,344 frontend priors/covariances gave:

| Held-out RTK quantity | Replay | Reproduction gate |
|---|---:|---:|
| Median residual | 0.09891653 m | <= 0.12 m |
| Inlier p95 residual | 0.12088889 m | <= 0.15 m |
| Inlier fraction | 1.0 | >= 0.95 |

The replay shows that the current gates accept the golden model. It does not
constitute a fresh frontend or mapper execution.

That reduced pose run completed on 2026-07-30:

- solve wall time: **1,083 s (18.1 min)**;
- peak process RSS: **8.8 GiB**; minimum host memory available: **5.9 GiB**;
- all **1,344 frames / 2,688 images** registered;
- 18,662 points and 2,759,770 observations;
- minimum optimized support: **749 observations/image**;
- mean reprojection error: **1.335 px**;
- fixed-scale RTK residual: **9.8 cm median / 27.4 cm p95**;
- diagnostic Sim(3) scale: **0.984637**, recorded but not applied; and
- median difference from the incremental fixed-scale control pose:
  **1.03 cm / 0.121 deg**.

The complete 65,000-iteration GS comparison finished on 2026-07-31:

| Metric | Incremental stereo BA | Reduced Global Mapper | Difference |
|---|---:|---:|---:|
| Masked PSNR | 24.4449 | 24.4583 | +0.0134 dB |
| Corrected masked PSNR | 25.8480 | 25.8418 | -0.0062 dB |
| SSIM | 0.5944 | 0.5904 | -0.0040 |
| LPIPS | 0.3232 | 0.3227 | -0.0005 |

These differences are a practical quality tie, not a quality gain. The mapper
stage fell from about 581 minutes to 18.1 minutes—about **32x faster**—while
retaining every frame. On the already canonical historical segment, the other
measured stages were approximately 63 minutes for feature extraction, 19
minutes for matching, and 4.2 hours for GS. The modern GPU frontend control
separately measured 77 s for features, 858 s for matching, and 776 s for its
Global solve. The adaptive-keyframe arms remain unmeasured.

## Genericity status

The enforced boundary is:

```text
dataset adapter
    -> immutable contract-v2 segment
    -> sealed mapper-neutral frontend
    -> isolated named pose backend
    -> sealed visibility TilePlan or single-scene path
    -> pose-matched cloud
    -> GS mapper/evaluator
```

Implemented behind the same adapter contract and covered by targeted
unit/synthetic checks:

- project ROS 2 ZED/u-blox input;
- AgriGS external-folder input; and
- CitrusFarm-style ordered ROS 1 bags with stereo `sensor_msgs/Image` and
  single-receiver Piksi `NavSatFix`; and
- Rosario v2 ROS 1 input with rectified IR stereo, aligned recorded depth,
  separately sealed RGB, and offline dual-position PPK evidence.

Real-data evidence differs by source. The headland contract migration is fully
published and validated. CitrusFarm has passed bag-chain, timing, calibration,
sampling, storage, environment, bounded contract publication, SGBM, sealed
frontend, complete visual reconstruction, and registration checks. Its first
  full-window pose failed the factor-held-out RTK export gate, and no Citrus
metric GS is accepted. Rosario has completed immutable publication, all-frame
stereo Global reconstruction, production fixed-scale pose export, one IR/depth
GS, and several metric-ineligible colour diagnostics.

Not implemented or verified:

- arbitrary ROS topic/message layouts without configuration;
- stereo cameras distributed across independent recordings;
- a full 77-minute global visual pose solve or automatic TilePlan;
- a complete real CitrusFarm pose and GS result; and
- an independently validated Rosario RGB extrinsic/clock state and a
  production-eligible colour GS result.

## Sealed visibility TilePlan

TilePlan schema v3 and the `tiles-plan` stage are implemented. They consume one
metric-depth segment and one complete global pose artifact, derive an oriented
spatial support graph from projected depth, recursively split under a derived
training-presentation budget, and publish disjoint ENU cores with overlapping
visibility context. Row count, waypoint labels, and fixed time intervals are
not inputs.

The artifact is atomic and non-overwriting. It content-seals both image streams,
all depth, the segment contract, the complete pose artifact, exact frame lists,
support geometry, configuration, and package state. Its verifier recomputes
selection, split preservation, ownership, capacity, and core-support coverage.
A bounded number of depthless frames may inherit adjacent temporal membership;
exceeding the declared fraction fails closed. Legacy pose status propagates and
cannot become a production metric claim.

The cached headland behaved as intended:

- automatic capacity returned one tile because 1,176 training frames fit under
  the resolved 1,300-frame budget;
- the controlled two-tile artifact `headland-two-tile-seam-v1` retained
  919/879 training frames and 132/126 validation frames;
- its core training-support coverage is 99.339%/99.356%; and
- its 2.482 m context shares 622 training frames.

The forced split lies near the middle of the measured dominant row direction;
it is not a row/U-turn hard-code. The plan is marked provisional because the
historical AB03 pose has `legacy_unassessed` evidence under the modern schema.
It is valid for a rendering/seam A/B but not a new georeferencing claim.

Tile execution and scene publication are now implemented. A tile cloud uses
only its sealed training IDs and crops metric points to the visibility context;
packed context masks constrain depth and photometric supervision without
copying the source segment. Training preserves the full context checkpoint but
exports only exact half-open core-owned Gaussian centres. `scene-publish`
rejects mismatched source/pose/plan/trainer/config identities, concatenates the
core tensors, renders every one of the 168 source validation views once with no
far-plane truncation, and separately gates a 1 m internal-boundary band. The
real cached plan supplies 20,206,365 seam depth pixels across 77 held-out views.
Quality failures publish an explicitly labelled diagnostic scene rather than a
false pass.

The controlled GPU run completed on 2026-08-10. It first trained a new
same-code 65k monolithic control, then the two 65k tiles, because the frozen
AB03 result used an earlier trainer implementation. All stages exited zero and
the terminal scene seal verifies against the complete source evidence.

| Held-out metric | Same-code monolith | Core-owned tiled scene | Change |
|---|---:|---:|---:|
| Masked PSNR | 24.3481 dB | **24.6528 dB** | **+0.3048 dB** |
| Corrected masked PSNR | 25.7284 dB | **26.1087 dB** | **+0.3803 dB** |
| SSIM | 0.5851 | **0.6126** | **+0.0274** |
| LPIPS-CC | 0.3221 | **0.2959** | **-0.0261** |

The same 168 source validation views were rendered exactly once. In the exact
1 m internal-boundary band, containing 20,206,365 metric-depth pixels across
77 held-out views, masked/corrected masked PSNR improved by 0.1580/0.0921 dB,
SSIM improved by 0.0278, and LPIPS-CC decreased by 0.0110. All 13 source,
identity, ownership, whole-view, and seam checks pass.

Each tile trained to 2.5 million Gaussians. Exact half-open ownership retained
3,751,281 tensors in the combined scene, and the opacity-filtered provisional
PLY retains 2,258,580. The run took **11 h 29 min 46 s** on the RTX 3080
Laptop, occupied **2.1 GiB**, and published
`scene.PROVISIONAL.ply` with SHA-256
`ea7e3eb71c1d0ef50226f870da3ed620649de12a1f3d3e9d6eafb3955b92e398`.
There were no OOMs, swaps, failed stages, or scientific warnings; only benign
dependency deprecation/future warnings appeared.

This proves that visibility context plus exact ENU core ownership can scale GS
capacity without creating a seam on this sequence. It is not an equal-compute
claim: the two-tile arm uses two independent 65k/2.5-million optimizations.
The average improved, but 28/168 frames lost more than 0.3 dB raw masked PSNR
and five lost more than 1 dB, so production reporting must add lower-tail and
temporal-block checks. Stage-level recovery verifies completed artifacts;
exact within-optimizer resume remains future work for the multi-day server
build.

The whole recording audit found about 4,640 s and 69,088 raw stereo frames.
The completed stride-five ingest retained 10,227 pairs; its eventual split is
expected to yield about 8,949 training frames. The current presentation budget
therefore implies at least about seven tiles before halo duplication, likely
roughly 8--12 after visibility context. The final topology cannot be known
until the global pose artifact exists. See
[TILED_SCENE.md](docs/methods/TILED_SCENE.md).

## Full-field deployment preparation

The first guarded laptop/server path is implemented and documented in
[SERVER_RUN.md](SERVER_RUN.md):

- ROS2 ingest now spools each selected compressed stereo pair to disk and
  atomically consumes it during publication instead of retaining the complete
  recording in RAM;
- heading association keeps the original 150 ms freshness definition while
  bounding the accepted full-field dropout shape to at most 0.5% of selected
  frames, two consecutive frames, and 0.5 s nearest-source residual. Invalid
  rows remain explicitly invalid;
- the derived depth segment can be materialized as a self-contained,
  terminally SHA-256-sealed directory with no symlinks, so the raw bag does not
  need to move to the server;
- the server preflight binds the portable data, clean Git snapshot, exact
  Python/COLMAP stack, supported 24 GB RTX GPU/driver, RAM, storage mount, and
  LPIPS weights;
  the first unmeasured full solve defaults to 120 GiB RAM and 500 GiB free
  space; and
- the full launcher prepares one global pose/automatic TilePlan, selects the
  highest-workload tile for a smoke test, then trains arbitrary planned tiles
  sequentially and publishes a production scene without a monolithic GS
  reference. Completed stages are verified and reused; an interrupted tile is
  preserved and retried under a new immutable attempt name.

The full local ingest/depth/portable publication has now completed, the
repository was made public, and the portable segment is being transferred to
the server. The latest implementation is not yet on remote `main`: on
2026-08-11 local `HEAD` was `ece3855` (already seven commits ahead before the
new bootstrap edits) while remote `main` was `8dff161`. No server pose solve or
server GS has started. Exact optimizer-level resume is not claimed; the
recovery boundary is the completed tile.

The local artifact has 10,227 frames over 4,621.30 s and 30,687 sealed files
totalling 31,546,138,468 bytes (29.38 GiB), with no symlinks. USB-2 ingest took
2 h 03 min, SGBM 44 min 50 s, and materialization 1 min 32 s; every stage
exited zero without swap. The portable inventory identity is
`2d10e9f39dee8e89125b4644b269c06d3344611d7b341fb271e45dcea17b843c`.

The server inventory on 2026-08-11 found an idle NVIDIA GeForce RTX 4090 with
24,564 MiB, driver 580.95.05, 24 available CPU threads, 125 GiB reported RAM,
and about 17 TiB free on the 19 TiB `/data` volume. System `nvcc` and COLMAP are
absent. This is expected: the tested PyTorch wheel carries its CUDA 12.1
runtime and COLMAP 4.1.1 CUDA is installed in a separate project-local Conda
prefix. A guarded bootstrap now places both environments and their package
caches under `/data/jkobo/rtk-splat`, leaving base Conda, system CUDA/drivers,
Docker, and other users unchanged. The server preflight now accepts either a
24 GB RTX 3090 or 4090 and still performs real gsplat and COLMAP CUDA smokes.
The first server installation revealed that gsplat's former
`docs.gsplat.studio/whl/pt24cu121` listing is now empty. The environment now
uses the official v1.5.3 CPython-3.10/PyTorch-2.4/CUDA-12.1 GitHub release
wheel directly and pins its published SHA-256, avoiding index drift.

The finalized real-bag ingest probe covered the first 120 s and published 316
frames over 108.92 s in **29 min 18 s**. It peaked at **452,984 KiB RSS** with
no swap. Exactly one heading association (frame 315) was 190.482 ms from its
nearest source, so it remained explicitly invalid; the observed invalid
fraction was 0.316%, longest run one frame, and all bounded full-field policy
checks passed. All 316 initial visual poses remained available as rough
initialization. This verifies memory and dropout behavior, not full-recording
runtime or final georeferencing.

## Known scientific limitations

- Validation views participated in the historical SfM pose estimation.
- The headland sequence lacks independent camera-pose ground truth.
- One successful scene is insufficient to claim generalization or SOTA.
- The controlled two-tile scene passes rendering and seam gates, but its
  `legacy_unassessed` source pose makes it ineligible for a new metric-
  georeferencing claim.
- Global Mapper does not optimize RTK factors in bundle adjustment. The
  optional sidecar constrains camera positions only; it does not add an
  explicit dual-antenna heading factor.
- Current right imagery contributes to depth and visual pose reconstruction;
  right-camera GS photometric supervision is not validated.
- Tile planning, tile-aware cloud/training, core-cropped export, scene
  publication, and whole-scene/boundary-band validation completed one real GPU
  A/B. Arbitrary-N orchestration and no-monolith production publication are now
  implemented; exact within-training resume remains deliberately unclaimed.
- Independent target-based stereo calibration has not resolved the remaining
  shared scale uncertainty.
- CitrusFarm course heading is weak or unavailable during near-stationary
  motion, and the single-receiver GPS-frame orientation used by the composed
  camera extrinsic is an explicit assumption rather than a measured heading.
- CitrusFarm's provided trajectory is GNSS-derived and is used only for window
  selection/evaluation context; it is not an independent pose ground truth for
  a method that already consumes Piksi GNSS.
- Rosario's PPK sigma is an external prior rather than surveyed ground truth;
  its camera/antenna transform is not independently surveyed, and its current
  RGB extrinsic/clock pair is not jointly observable enough for a metric claim.

## Next full-field preparation

1. Finish transferring the completed portable segment, restore ownership of
   only `/data/jkobo/rtk-splat` to `imoroz`, and verify its terminal SHA-256
   inventory. The raw 151.9 GB MCAP does not need to move.
2. Pull the environment-bootstrap commit into a clean server checkout, install
   the pinned project-local Python and COLMAP environments, and pass the server
   preflight. The first full solve defaults to a 120 GiB RAM and 500 GiB free-
   space gate.
3. Run and gate the complete global visual pose
   reconstruction, and generate an automatic TilePlan without a forced count.
   The exact 20,454-image pose solve is now the main unvalidated bottleneck.
4. Train one 2.5-million-Gaussian tile on the RTX 4090 as a resource/quality
   smoke, then launch the remaining roughly 8--12 planned tiles (the planner's
   sealed output is authoritative).

The RTX 4090 and 19 TiB data volume are sufficient for the validated sequential
per-tile GS workload. A separate guarded arbitrary-tile server launcher is now
implemented; the cached two-tile script must still not be repurposed. Recovery
is exact between sealed stages and tile attempts, but an interrupted optimizer
restarts only that active tile under a new name. A provisional end-to-end budget
is 2--4 days and 0.5--1 TB of fast working storage, subject to the full pose
solve and first server-tile measurements.

The adaptive frame-density A/B is optional: run it before the server only if
reduced sampling will be used in production. Otherwise the first full-field
experiment should retain all selected frames and spend the extra runtime to
avoid introducing an unvalidated variable. No paper-level or SOTA claim should
be made from one field or from cross-dataset PSNR comparisons.
