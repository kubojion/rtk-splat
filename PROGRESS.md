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
the incremental result and **42/42** for the Global result. The full suite
passes **55 tests**. A repository-only regression test freezes the key hashes
and quality values without requiring the external work directory in CI.

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

## Phase 1 repository cleanup

Completed in the current working tree:

- Removed tracked bytecode and generated caches; added ignore rules.
- Added package metadata and one discoverable test command.
- Grouped reproduction, benchmark, and environment configurations.
- Replaced six root-level run scripts with one guarded, parameterized
  reproduction script; five obsolete scripts were removed.
- Made the long-run check path genuinely read-only.
- Fingerprint new COLMAP resume markers so changed commands or calibration do
  not silently reuse stale completed steps.
- Added an exact golden-artifact verifier and nondeterministic acceptance gates.
- Made `--config` explicit; there is no local-machine default.
- Fixed the external TUM trajectory source's incomplete track structure.
- Stopped silently accepting unimplemented depth backend names.
- Store both rectified camera calibrations in new canonical segments so future
  COLMAP pose backends need not reopen a ROS bag.
- Corrected ENU CRS propagation into new segment metadata.
- Preserved the metric-integrity implementation and tests without enabling it
  in normal mapping.

No COLMAP, GS, input bag, or existing work artifact was modified or rerun.

## Global Mapper backend

Implemented on 2026-07-30 as an explicit, isolated A/B backend:

- `global-prepare` clones and SHA-256 verifies the completed stereo
  feature/match database under a new pose artifact.
- `global-solve` runs COLMAP 4.1.1's integrated Global Mapper with fixed
  intrinsics, fixed stereo extrinsics, CPU optimization, lowered scheduling
  priority, and no-swap memory protection.
- `global-export` uses robust fixed-scale SE(3) georegistration. Sim(3) scale is
  diagnostic only and cannot silently rescale metric stereo geometry.
- Registration, rig completeness, intrinsics, baseline, metric-scale, and broad
  temporal-continuity checks are solver-integrity gates.
- Reprojection, output observations/tracks, and RTK residual targets remain
  visible diagnostics; different mappers produce different point sets, and the
  current RTK fit still uses a rough camera/antenna TF.
- Failed candidates retain diagnostics but do not publish the standard pose
  files consumed by cloud construction and training.
- The guarded reproduction script stops after pose validation by default.
  Exploratory GS is explicit and starts only after integrity gates pass.
- Torch/NumPy training initialization now has a recorded configurable seed.
  A formal backend A/B still requires retraining an incremental fixed-SE(3)
  control with the same seed; the historical Sim(3) run is context, not that
  control.
- Per-image retained-observation coverage is audited after solving; a candidate
  with fewer than 250 optimized observations in any image cannot be published.

The full-track headland attempt is preserved as
`colmap_global_headland_v1`. Rotation/track setup reached Ceres global
positioning, where RSS grew to **15.7 GiB** and available memory remained below
the configured **4 GiB** floor. The monitor safely terminated the process
after **138 s**. The source database was unchanged and no pose was published.

The separately named `colmap_global_headland_reduced_v1` profile retained:

- all 53,058 verified pairs were available to rotation averaging;
- target coverage was 1,000 selected long tracks per image;
- 60,000 tracks was the hard ceiling;
- the reference-model proxy reaches the target at 53,866 tracks / 6.91 million
  observations, with at least 1,001 tracks in every image;
- retriangulation is disabled because COLMAP would otherwise discard the cap
  and recreate the full-memory problem.

That reduced pose run completed on 2026-07-30:

- solve wall time: **1,083 s (18.1 min)**;
- peak process RSS: **8.8 GiB**; minimum host memory available: **5.9 GiB**;
- all **1,344 frames / 2,688 images** registered;
- minimum optimized support: **749 observations/image**;
- mean reprojection error: **1.335 px**;
- fixed-scale RTK residual: **9.8 cm median / 27.4 cm p95**;
- diagnostic Sim(3) scale: **0.984637**, recorded but not applied; and
- median difference from the incremental fixed-scale control pose:
  **1.03 cm / 0.121 deg**.

Every structural publication gate passed, but the RTK/reprojection result
targets did not. The artifact is therefore safe to use for the planned
exploratory GS A/B, not evidence that Global Mapper already improves quality.
The mapper-only solve was about 32 times faster than the historical 581-minute
incremental mapper stage, after reusing the same completed feature/match front
end.

The complete 65,000-iteration headland GS continuation is prepared as
`scripts/reproduce/headland_global_gs_overnight.sh`. Its dry run passes all
49 tests and verifies 1,344 left images, 1,344 right images, 1,344 depth maps,
the 1,176/168/0 split, AC power, host memory, CUDA/VRAM, and available disk. It
cannot start or retry COLMAP. It validates the final pose provenance, finite
metrics, complete validation coverage, loadable finite checkpoint tensors,
non-empty PLY geometry, and final iteration. The expected runtime is about
4--5 hours because the completed pose/front end are reused.

The complete Global Mapper GS continuation finished on 2026-07-31:

| Metric | Incremental stereo BA | Reduced Global Mapper | Difference |
|---|---:|---:|---:|
| Masked PSNR | 24.4449 | 24.4583 | +0.0134 dB |
| Corrected masked PSNR | 25.8480 | 25.8418 | -0.0062 dB |
| SSIM | 0.5944 | 0.5904 | -0.0040 |
| LPIPS | 0.3232 | 0.3227 | -0.0005 |

These differences are practically a quality tie, not evidence of a quality
gain. The important measured result is speed: the mapper stage fell from about
581 minutes to 18.1 minutes while retaining all 1,344 frames and comparable GS
quality. A fresh run on the already canonical headland segment would still
need about 63 minutes of feature extraction, 19 minutes of matching, 18
minutes of Global mapping, and about 4.2 hours of GS optimization. A
paper-grade backend attribution still needs a newly trained, provenance-matched
incremental fixed-scale control.

## Genericity status

The intended stable core boundary is now documented and partially enforced:

```text
dataset adapter
    → canonical stereo segment
    → named pose backend
    → pose-matched cloud
    → GS mapper/evaluator
```

Verified adapters:

- Project ROS 2 ZED/u-blox recordings.
- AgriGS external-folder format.

Not yet verified:

- CitrusFarm.
- Arbitrary ROS topic layouts.
- Stereo cameras split across separate bag files.
- A bag-free COLMAP preparation using a newly generated portable segment.

The last item is supported in code for new metadata but still needs an
end-to-end adapter test before being claimed.

Read-only inspection of the mounted CitrusFarm sequence found ROS 1 split bags,
7,592 synchronized 10 Hz ZED2i stereo pairs, recorded depth/confidence, and
single-receiver fixed RTK over 12 min 39 s. Global stereo SfM is conceptually
compatible, but the present adapter cannot decode that dataset through YAML
alone: it still needs ROS 1/ROS 2 autodetection, ordered split-bag streaming,
raw/compressed image decoding, timestamp synchronization, optional recorded
depth, and a covariance-aware single-RTK mode.

## Known scientific limitations

- Validation views participated in SfM pose estimation.
- The headland sequence lacks independent camera-pose ground truth.
- One successful scene is not enough to claim generalization or SOTA.
- Current right imagery contributes to stereo depth and BA, but right-camera GS
  supervision is not validated.
- Full-field chunking, submap consistency, and tile merging are not implemented.
- Metric stereo should ultimately use a fixed-scale SE(3) georegistration model;
  the reference exporter still records its legacy bounded Sim(3) alignment.

## Next implementation phase

1. Split stereo feature extraction/matching into a mapper-neutral front-end
   artifact so Global Mapper can be the primary new-field backend without
   first running incremental COLMAP. Retain incremental mapping as a tested
   fallback and reference.
2. Replace fixed frame stride and fixed-index matching with adaptive
   distance/rotation/quality keyframes and an RTK-guided temporal/spatial match
   graph.
3. Add bounded overlapping submaps and covariance-aware RTK constraints so a
   77-minute field is never one unbounded SfM or GS optimization.
4. Implement and contract-test the generic ROS 1/ROS 2 dataset adapter, then
   use CitrusFarm as the first external, single-RTK, faster-motion acceptance
   test.
5. Train a provenance-matched incremental fixed-SE(3) control and repeat the
   selected production path on the hangar-to-field sequence.

The first speed target is to preserve quality within 0.3 dB while materially
reducing pose-estimation time. A further 3.8 dB pose-only gain is not assumed.
