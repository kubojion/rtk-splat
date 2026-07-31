# Mapper-Neutral Global and Incremental Backend

## Current status

The Phase 3/4 backend interface is implemented and unit-tested. It can consume
a sealed contract-v2 frontend, solve a keyframe model, register all remaining
stereo images, run quality gates, and publish a fixed-scale metric pose
artifact. No fresh real-headland A/B has yet exercised this new interface.

The measured Global Mapper result later in this document is a historical golden
experiment. It established that a reduced Global solve could match the
incremental reconstruction's GS quality much faster, but it reused the older
incremental run's feature/match database. It does not prove the speed or quality
of the new standalone frontend or adaptive keyframes.

## Purpose

The current design separates correspondence construction from pose solving:

- build images, stereo rig, features, RTK priors, and verified matches once;
- seal that frontend as immutable evidence;
- run Global as the primary mapper candidate;
- retain incremental as an isolated fallback and scientific control; and
- restore every non-keyframe left/right image before pose publication.

This removes the previous operational dependency in which Global could be
tested only after a completed incremental reconstruction.

## Non-destructive artifact boundary

COLMAP opens databases in writable mode. A backend therefore never opens the
sealed frontend database directly. `backend-prepare`:

1. rehashes the terminal seal, including finalized JSON/pair evidence, image
   link targets and contents, and the database's committed view;
2. creates a private backend workspace and transaction-consistent SQLite
   snapshot, including committed WAL-only rows;
3. verifies the snapshot hash, integrity check, schema, and table inventory;
4. records the selected mapper and keyframe set; and
5. leaves both the canonical segment and frontend unchanged.

Global and incremental arms must use different backend names. Existing backend
or pose destinations fail closed instead of being overwritten.

## Explicit workflow

The installed `rtk-splat` command exposes the current stages:

```bash
rtk-splat frontend-build --config CONFIG.yaml \
  --frontend-name all-gpu --keyframe-preset all --feature-profile gpu
rtk-splat frontend-features --config CONFIG.yaml --frontend-name all-gpu
rtk-splat frontend-rig --config CONFIG.yaml --frontend-name all-gpu
rtk-splat frontend-priors --config CONFIG.yaml --frontend-name all-gpu
rtk-splat frontend-match --config CONFIG.yaml --frontend-name all-gpu

rtk-splat backend-prepare --config CONFIG.yaml \
  --frontend-name all-gpu --backend global --backend-name all-gpu-global
rtk-splat backend-solve --config CONFIG.yaml \
  --backend-name all-gpu-global
rtk-splat backend-register --config CONFIG.yaml \
  --backend-name all-gpu-global
rtk-splat backend-quality --config CONFIG.yaml \
  --backend-name all-gpu-global
rtk-splat backend-export --config CONFIG.yaml \
  --backend-name all-gpu-global --pose-name all-gpu-global
```

Commands are explicit by design. There is no `all` command, and normal
cloud/training execution never launches a pose solver implicitly.

The Python API is:

- `MapperConfig`;
- `prepare_mapper_backend()`;
- `run_mapper_solve()`;
- `run_image_registration()`;
- `run_quality_summary()`; and
- `export_pose_artifact()`.

All are defined in `rtk_splat.backends.mapper`.

## Solve and all-frame registration

`backend-solve` filters the private database to the selected keyframe images
and runs either:

- COLMAP's integrated Global Mapper, the default candidate; or
- COLMAP's incremental mapper, the optional fallback/control.

`backend-register` then uses the complete backend database and COLMAP image
registrator to add non-keyframes. Publication requires the exact canonical
left and right image name for every timestamp; a left-only, mismatched, or
partially registered reconstruction fails.

Keyframes therefore bound the expensive solve without changing the final
camera set available to cloud construction and GS supervision.

## RTK semantics

RTK has three distinct roles:

1. accepted position/heading evidence informs keyframe and match planning;
2. filtered Cartesian position priors are inserted in the frontend database
   and preserved in each backend snapshot; and
3. trusted visual-camera/RTK correspondences determine the geographic
   alignment at export.

The integrated Global Mapper remains a **visual** solver. It does not optimize
RTK residuals or covariance-weighted RTK factors inside bundle adjustment.
Database-prior retention must not be presented as RTK-constrained Global BA.

The exporter divides trusted priors into deterministic contiguous temporal
blocks. It robustly estimates a fixed-scale SE(3) transform using alternating
calibration blocks, then measures and gates residuals only on untouched
holdout blocks. A Sim(3) fit uses calibration blocks only, is recorded as a
scale diagnostic, and is never applied. This keeps calibrated stereo depth,
rig baseline, trajectory, and initialization geometry in one metric scale
without evaluating alignment on its own fitting samples.

## Resume and publication gates

Prepare, solve, register, quality, and export have separate verified markers.
A stage resumes only when its inputs, command, configuration, and expected
outputs still match.

Before publishing `viewmats.npy` and `cam_centers.npy`, the backend checks:

- exact left/right registration for all canonical timestamps;
- a physically admissible connected solve-frame graph before mapping;
- unchanged calibrated camera and stereo-rig geometry;
- finite poses and acceptable temporal continuity;
- adequate registered observations;
- at least three trustworthy camera-centre priors;
- held-out fixed-scale RTK residual and inlier gates; and
- complete quality and provenance records.

Failed candidates keep their logs, candidate poses, and diagnostics but do not
publish a normal pose artifact consumable by cloud construction or training.
The output artifact also records image IDs, exact timestamps, names, quality,
and provenance.

## Historical measured headland control

The historical experiment used 2,688 rectified images (1,344 stereo pairs),
two calibrated PINHOLE cameras, a 0.119846250 m stereo transform, 26.38 million
cached keypoints, and 53,058 cached verified pairs. It did not rerun feature
extraction or matching and did not use an IMU.

The first full-track Global attempt reached Ceres global-position setup, grew
to **15.7 GiB** RSS, and held available memory below the configured **4 GiB**
floor. The safety monitor terminated it after **138 s**; no pose was
published.

The separately named reduced profile retained all 53,058 verified pairs for
rotation averaging and selected long tracks with a 60,000-track ceiling. It
disabled final retriangulation because that operation would rebuild an
unbounded point set from the full graph.

The reduced solve completed with:

- **1,083 s (18.1 min)** solve time;
- **8.8 GiB** peak process RSS and **5.9 GiB** minimum host memory available;
- **1,344/1,344 frames** and **2,688/2,688 images** registered;
- 18,662 points and 2,759,770 observations;
- at least 749 observations per image;
- **1.335 px** mean reprojection error;
- **9.8 cm median / 27.4 cm p95** fixed-scale RTK residual;
- diagnostic Sim(3) scale **0.984637**, not applied; and
- **1.03 cm / 0.121 deg** median difference from the incremental
  fixed-scale control trajectory.

Its completed 65,000-iteration GS comparison was:

| Metric | Incremental stereo BA | Reduced Global | Difference |
|---|---:|---:|---:|
| Masked PSNR | 24.4449 | 24.4583 | +0.0134 dB |
| Corrected masked PSNR | 25.8480 | 25.8418 | -0.0062 dB |
| SSIM | 0.5944 | 0.5904 | -0.0040 |
| LPIPS | 0.3232 | 0.3227 | -0.0005 |

This is a practical quality tie. The mapper stage was about **32x faster** than
the historical 581-minute incremental mapper stage because both arms reused
the same completed frontend evidence. The experiment did not show a quality
gain and is not evidence for the unrun adaptive-keyframe workflow.

## Planned controlled A/B

The next experiment must use a newly validated contract-v2 segment and the
current sealed frontend:

1. `gpu` versus `cpu_reference` features with every frame retained;
2. pose-only `all`, `dense`, `balanced`, and `sparse` keyframe arms;
3. Global primary and incremental fallback/control from verified snapshots;
4. 100% all-frame stereo registration and no georegistration regression;
5. pose/rendering proxies to select one candidate; and
6. GS only for the baseline and selected candidate.

The acceptance target is a material pose-runtime reduction, no structural or
georeferencing regression, and no more than 0.2--0.3 dB masked-PSNR loss.
Those outcomes remain hypotheses until the real A/B completes.

## Scientific scope

Global Mapper is an efficient backend, not the paper contribution by itself.
The current implementation establishes clean evidence boundaries and a fair
way to measure mapper/keyframe choices. Covariance-weighted RTK factors inside
local BA, RTK-anchored submaps, CitrusFarm support, and rendering-quality
experiments belong to later Phases 5--7 and are not implemented here.
