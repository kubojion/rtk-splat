# RTK-Splat Progress

Last updated: 2026-08-01

This is the current source of truth for this repository. Superseded plans are
kept under `docs/archive/`.

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

- The reusable `rtk_splat/core/` package is isolated from ROS, adapters, frontends,
  backends, diagnostics, and orchestration.
- `rtk_splat` is now the only installed top-level Python namespace. Adapters,
  frontends, backends, workflows, and diagnostics are explicit subpackages;
  the former collision-prone global package names no longer exist.
- Dataset-specific ingestion lives under `rtk_splat/adapters/`; the registry
  supports ROS 2 ZED/u-blox, ROS 1 CitrusFarm, and AgriGS.
- Calibration ROS topics, u-blox message registration, message decoding, and
  bounded MCAP reading moved into `rtk_splat/adapters/calibration_bag.py`.
  Diagnostics retain only plain records, numerical analysis, COLMAP parsing,
  and artifact reporting, and import without the optional ROS stack.
- Immutable contract-v2 segments preserve exact nanosecond timestamps, full
  GNSS covariance/status, both stereo calibrations, explicit transform
  semantics, and the complete optional dual-antenna N/E/D baseline.
- Robot defaults and sequence-specific settings are deep-merged from explicit
  configuration. There is no local-machine default configuration.
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

The real headland GPU/CPU feature A/B and
all/dense/balanced/sparse keyframe A/B have **not** been run. Phase 4 is
prepared for measurement, not yet demonstrated as a speed or quality
improvement.

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

The dataset-neutral core remains below its enforced 2,000-line budget.

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
  result, not this estimate, will be authoritative);
- recorded rectified stereo baseline: **0.119885166 m**;
- estimated camera-to-GNSS header correction: **+72.548749 ms**;
- final per-sequence configured correction: **+72.548749 ms** (zero residual);
  and
- estimated clock drift across the window: **-2.580854 ms**.

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
claimed. The repository suite passes **207 tests** in the RTK-Splat
environment after this integration.

`scripts/runs/citrusfarm_05_13d_uturn.sh` provides plan, preflight, fresh run,
and evidence-checked resume modes without `tmux`. It runs ingest, immutable
SGBM depth, the all-frame GPU frontend, bounded Global Mapper, all-frame
registration and held-out quality gates, pose export, cloud construction, and
the matched 65,000-iteration/2.5-million-Gaussian training profile. The measured
first attempt took about **42 min** to the export gate: 9m53s ingest, 3m29s
SGBM, 2m49s frontend build/features/rig/priors, 13m39s matching, and 11m18s
backend preparation/solve/registration/quality. GS remains an estimated
**3--5 hours** because this pose did not pass and training was not started.

A full-window Citrus ingest, COLMAP reconstruction, and rejected RTK-refinement
control now exist, but no accepted
pose artifact, cloud, GS model, or rendering metric has been produced. In
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
minutes for matching, and 4.2 hours for GS. These timings do not yet measure
the new standalone frontend or adaptive-keyframe arms.

## Genericity status

The enforced boundary is:

```text
dataset adapter
    -> immutable contract-v2 segment
    -> sealed mapper-neutral frontend
    -> isolated named pose backend
    -> pose-matched cloud
    -> GS mapper/evaluator
```

Implemented behind the same adapter contract and covered by targeted
unit/synthetic checks:

- project ROS 2 ZED/u-blox input;
- AgriGS external-folder input; and
- CitrusFarm-style ordered ROS 1 bags with stereo `sensor_msgs/Image` and
  single-receiver Piksi `NavSatFix`.

Real-data evidence differs by source. The headland contract migration is fully
published and validated. CitrusFarm has passed bag-chain, timing, calibration,
sampling, storage, environment, bounded contract publication, SGBM, sealed
frontend, complete visual reconstruction, and registration checks. Its first
full-window pose failed the factor-held-out RTK export gate, and GS remains
unverified.

Not implemented or verified:

- arbitrary ROS topic/message layouts without configuration;
- stereo cameras distributed across independent recordings;
- full-field submaps and a custom RTK-constrained local BA/submap graph;
- a fresh real v2 headland run through the new frontend/backend interface; and
- a complete real CitrusFarm pose and GS result.

## Known scientific limitations

- Validation views participated in the historical SfM pose estimation.
- The headland sequence lacks independent camera-pose ground truth.
- One successful scene is insufficient to claim generalization or SOTA.
- The historical incremental reference used legacy bounded Sim(3) alignment;
  the current backend uses fixed-scale SE(3), so a provenance-matched control
  is still required.
- Global Mapper does not optimize RTK factors in bundle adjustment. The
  optional sidecar constrains camera positions only; it does not add an
  explicit dual-antenna heading factor.
- Current right imagery contributes to depth and visual pose reconstruction;
  right-camera GS photometric supervision is not validated.
- Full-field chunking, submap consistency, and tile merging are not
  implemented.
- Independent target-based stereo calibration has not resolved the remaining
  shared scale uncertainty.
- CitrusFarm course heading is weak or unavailable during near-stationary
  motion, and the single-receiver GPS-frame orientation used by the composed
  camera extrinsic is an explicit assumption rather than a measured heading.
- CitrusFarm's provided trajectory is GNSS-derived and is used only for window
  selection/evaluation context; it is not an independent pose ground truth for
  a method that already consumes Piksi GNSS.

## Next controlled experiment

1. Preserve both the successful visual reconstruction and the rejected
   position-prior refinement. Do not reinterpret the 1.72 cm aggregate median
   gain as acceptance.
2. Preserve the completed fresh pose-prior mapper arm as a rejected baseline;
   it improved held-out geometry but failed the existing RTK and scale gates.
3. Implement bounded
   RTK-anchored local submaps with robust temporal/block consistency and an
   RTK-constrained submap graph. That is the method-level path; repeatedly
   rerunning a whole-model COLMAP refinement is not.
4. Once a pose mechanism passes, make one new immutable Citrus ingest so the
   verified receiver-state stream is retained, then reproduce the accepted pose
   into a new work directory. If interrupted without code/config changes, use
   the launcher's evidence-checked resume mode.
5. If the pose passes, complete one matched 65,000-iteration Citrus GS run and
   report its scene-specific train/validation metrics, runtime, peak resources,
   and visual failure modes. Compare approaches on the same Citrus split; do
   not compare raw PSNR directly with headland.
6. Retain the validated normalized headland segment for the pending efficiency
   A/B: compare `gpu` and `cpu_reference` features, then pose-only `all`,
   `dense`, `balanced`, and `sparse` Global arms with incremental as the
   fallback/control.
7. Train only the headland all-frame control and selected adaptive arm with
   matched seed, split, depth, GS configuration, and provenance.

The Phase 4 acceptance target remains a material pose-runtime reduction with no
more than **0.2--0.3 dB** masked-PSNR loss and no georeferencing regression. A
further 3.8 dB pose-only gain is not assumed. No paper-level, SOTA, or
cross-dataset claim should be made from adapter preflight or a single
CitrusFarm run.
