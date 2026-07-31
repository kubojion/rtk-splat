# RTK-Splat Progress

Last updated: 2026-07-31

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

## Completed implementation: Phases 1, 3, and 4

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
  currently supports ROS 2 ZED/u-blox and AgriGS.
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

Phases 5--7—RTK-factor submaps, CitrusFarm support, and rendering-quality
changes—are outside this implementation.

After integration, the complete repository suite passes **155 tests**. The
dataset-neutral core contains **1,932 lines**, below the enforced 2,000-line
budget.

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

Verified through adapter conformance tests:

- project ROS 2 ZED/u-blox input; and
- AgriGS external-folder input.

Not implemented or verified:

- CitrusFarm ROS 1 ordered split-bag ingestion;
- arbitrary ROS topic/message layouts;
- stereo cameras distributed across independent recordings;
- full-field submaps and RTK-constrained local BA; and
- a fresh real v2 headland run through the new frontend/backend interface.

Read-only inspection of the mounted CitrusFarm sequence found ROS 1 split bags,
7,592 synchronized 10 Hz ZED2i stereo pairs, recorded depth/confidence, and
single-receiver fixed RTK over 12 min 39 s. The canonical contract and mapper
can represent single-RTK input, but the source still needs a tested ROS 1
split-bag adapter before end-to-end compatibility can be claimed.

## Known scientific limitations

- Validation views participated in the historical SfM pose estimation.
- The headland sequence lacks independent camera-pose ground truth.
- One successful scene is insufficient to claim generalization or SOTA.
- The historical incremental reference used legacy bounded Sim(3) alignment;
  the current backend uses fixed-scale SE(3), so a provenance-matched control
  is still required.
- Global Mapper does not optimize RTK factors in bundle adjustment.
- Current right imagery contributes to depth and visual pose reconstruction;
  right-camera GS photometric supervision is not validated.
- Full-field chunking, submap consistency, and tile merging are not
  implemented.
- Independent target-based stereo calibration has not resolved the remaining
  shared scale uncertainty.

## Next controlled experiment

1. Use the validated normalized headland contract-v2 segment; derive another
   immutable depth segment only if a different depth backend is tested.
2. Compare `gpu` and `cpu_reference` features with the `all` preset.
3. From independently sealed but controlled evidence, run pose-only `all`,
   `dense`, `balanced`, and `sparse` Global arms; retain incremental as a
   fallback/control.
4. Require complete stereo registration, acceptable reprojection and temporal
   continuity, fixed-scale RTK behavior no worse than the control, and stable
   resource use.
5. Use inexpensive pose/rendering proxies to select one adaptive arm.
6. Train only the all-frame control and selected arm with matched seed,
   split, depth, GS configuration, and provenance.

The Phase 4 acceptance target is a material pose-runtime reduction with no more
than **0.2--0.3 dB** masked-PSNR loss and no georeferencing regression. A
further 3.8 dB pose-only gain is not assumed. No paper-level, SOTA, or
cross-dataset claim should be made before this A/B and an independent dataset
evaluation.
