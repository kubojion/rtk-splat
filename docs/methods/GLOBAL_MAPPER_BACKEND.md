# Global Mapper pose backend

## Purpose

This backend is a controlled speed/quality A/B against the successful
incremental COLMAP reconstruction. It uses COLMAP 4.1.1's integrated Global
Mapper (the maintained GLOMAP implementation) while preserving the same:

- 2,688 rectified images;
- two calibrated PINHOLE cameras;
- 0.119846250 m stereo transform;
- 26.38 million cached keypoints; and
- 53,058 cached verified image pairs.

It does not rerun feature extraction or matching, and it does not use an IMU.
RTK remains the post-reconstruction geographic anchor.

## Non-destructive boundary

COLMAP opens databases in writable mode even when a pipeline only consumes
their contents. `global-prepare` therefore:

1. validates the completed source stereo sidecar;
2. records the source database size, timestamps, schema, counts, and SHA-256;
3. creates an independent full database copy under a new pose artifact;
4. verifies the copy's SHA-256 and SQLite inventory; and
5. rechecks the source after copying.

The Global Mapper process sees only the copy. Its images path is a read-only
symlink to the existing image tree. It never opens the successful database.

## Explicit stages

```bash
python -m rtk_splat.cli global-prepare \
  --config configs/reproductions/headland_stereo_ba.yaml \
  --pose-artifact colmap_global_headland_reduced_v1

python -m rtk_splat.cli global-solve \
  --config configs/reproductions/headland_stereo_ba.yaml \
  --pose-artifact colmap_global_headland_reduced_v1

python -m rtk_splat.cli global-export \
  --config configs/reproductions/headland_stereo_ba.yaml \
  --pose-artifact colmap_global_headland_reduced_v1
```

These stages are never invoked by `all`.

The guarded end-to-end headland A/B is:

```bash
scripts/reproduce/headland_global_mapper.sh \
  --python /path/to/rtk-splat/python \
  --colmap /path/to/colmap-4.1.1
```

The default stops after the pose export. After reviewing that result, add
`--resume --with-gs` to start cloud construction and the 65,000-iteration
exploratory GS arm. It still starts only if the pose integrity checks pass.

For an unattended run, the guarded wrapper is:

```bash
scripts/reproduce/headland_global_gs_overnight.sh --check
scripts/reproduce/headland_global_gs_overnight.sh
```

It consumes only an accepted `colmap_global_headland_reduced_v1` pose. If that
pose producer is still active, it waits; it never launches another COLMAP
process. Before GS it verifies the complete 1,344-pair canonical segment,
depth, split, AC power, host memory, disk, CUDA/VRAM idleness, and empty
destination run name.
It checks the final checkpoint tensors, PLY vertex count, provenance, finite
metrics, complete validation split, and configured final iteration before
reporting completion. Keep the terminal open, or run the wrapper inside
`tmux`; sleep is inhibited, but the four-hour optimizer has no checkpoint
resume.

## Solver profiles

The initial experiment used the complete track set:

- calibrated intrinsics are fixed;
- the measured sensor-from-rig transform is fixed;
- every per-frame rig pose remains optimizable;
- global positioning and BA optimize positions and points;
- three global BA rounds and retriangulation remain enabled;
- Global Mapper uses eight CPU threads; and
- GPU positioning/BA are disabled because this machine has 8 GiB VRAM.

That full-profile attempt is preserved under
`colmap_global_headland_v1`. During Ceres global-position problem setup, RSS
rose to 15.7 GiB and system available memory remained below 4 GiB. The safety
monitor terminated it after 138 seconds; it produced no publishable pose.

The separately named reduced profile retains all 53,058 verified pairs for
rotation averaging, then asks COLMAP to keep the longest tracks until every
image has more than 1,000 selected tracks, with a hard ceiling of 60,000
tracks. A proxy using the completed incremental model reaches that coverage at
53,866 tracks and 6.91 million observations, versus 18.21 million observations
in the reference model. COLMAP's final retriangulation is disabled because it
would delete the capped structure and rebuild it from the complete
correspondence graph. The GS initialization remains independent: it is fused
from stereo depth under the exported poses.

The verified-pair graph is multiscale temporal
(`Δframe = 0, 1, 2, 4, ..., 512`). It contains useful long-baseline links but
no retrieval-based loop closures. No database pair is removed in this
experiment.

### Measured reduced-profile result

The reduced solve completed in 1,083 seconds (18.1 minutes), with 8.8 GiB peak
process RSS and 5.9 GiB minimum host memory available. It registered every one
of the 1,344 frames / 2,688 images. The selected model has 18,662 points,
2,759,770 observations, at least 749 observations in every image, and 1.335 px
mean reprojection error.

Fixed-scale export produced a 9.8 cm median and 27.4 cm p95 residual to the
rough-TF RTK camera-centre track. A diagnostic-only Sim(3) fit reported scale
0.984637; it was not applied. The published trajectory differs from the
incremental fixed-scale control by 1.03 cm median / 3.33 cm p95 in position and
0.121 deg median / 0.150 deg p95 in rotation.

All structural gates passed, but the RTK and reprojection result targets did
not. This authorizes an exploratory GS comparison; it does not establish a
quality improvement. Reusing the identical completed front end, the mapper
stage was about 32 times faster than the historical 581-minute incremental
mapper stage.

The host has no swap. Resource samples are written every two seconds and the
solver is terminated if available memory remains below 4 GiB for two
consecutive samples, or if free disk falls below 5 GiB. Full and reduced
attempts always use distinct artifact names.

## Metric export

The stereo baseline is a metric constraint. The exporter therefore applies a
robust fixed-scale SE(3) transform from the visual world to local ENU. It also
fits a Sim(3), but records its scale only as a diagnostic and never applies it
to the published poses.

This matters because applying a free visual scale to poses while continuing to
use metric stereo depth makes the two geometry sources inconsistent. The prior
incremental result used the legacy bounded Sim(3) path and reported a scale of
0.985794. Global Mapper explicitly restores the reconstruction to the original
rig scale, so the new A/B directly tests whether the model is metric without
that correction.

## Integrity gates and result targets

Standard `viewmats.npy` and `cam_centers.npy` are written only if the
solver-integrity checks pass:

- exactly 1,344 registered frames and 2,688 registered images;
- every frame contains the matching left and right image;
- both intrinsics and the stereo transform remain unchanged;
- every registered image retains at least 250 optimized 3D observations;
- diagnostic scale remains in `[0.98, 1.02]`;
- temporal translation/rotation steps stay below the measured continuity
  limits.

Mean reprojection error, total observation count, and track length are recorded
against targets but are not publication gates: the reduced solver deliberately
selects a different point set, so those aggregate output statistics are not
paired pose-quality measurements.

The desired fixed-scale RTK residual remains 6 cm median, 12 cm p95, and 15 cm
maximum. It is reported as a result target, not a solver-integrity gate,
because the current comparison fits and evaluates the same trajectory against
a rough camera/antenna TF. A paper-grade accuracy claim requires the existing
covariance/status evidence and held-out temporal blocks rather than using this
post-fit number as ground truth.

Failed candidates retain `candidate_viewmats.npy`, diagnostics, logs, and
quality records, but cannot be consumed by cloud construction or training.

## Scientific interpretation

This experiment can establish a substantially faster pose backend and may
improve metric consistency. Its GS score against the historical result changes
both mapper and alignment policy, so a winning candidate still requires an
incremental-plus-fixed-SE(3) control with the same recorded training seed.
Ideally the final paper reports several matched seeds and raw held-out metrics.

By itself Global Mapper is not the paper's method contribution:
Global Mapper still optimizes visual reprojection and does not place RTK
factors inside its BA. A successful result is the baseline for the next method
step—RTK-anchored chunking or a custom RTK-constrained local/global optimizer.
