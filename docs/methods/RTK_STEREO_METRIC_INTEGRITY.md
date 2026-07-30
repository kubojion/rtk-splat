# RTK–Stereo Metric-Integrity Sidecar

This diagnostic stage audits and calibrates an existing metric stereo-COLMAP
trajectory against a primary RTK position antenna and the complete
primary-to-secondary dual-antenna vector. It does not run COLMAP, Gaussian
training, depth, or cloud generation, and it does not publish camera poses.

Run it explicitly:

```bash
python3 -m rtk_splat.cli integrity-audit \
  --config configs/reproductions/headland_stereo_ba.yaml
```

It is deliberately absent from the `all` stage. Outputs are written
atomically to:

```text
segment/calibration_artifacts/<calibration.output_artifact>/
```

An existing artifact name is never reused or overwritten.

## Model

The input is the raw, fixed-scale left-camera trajectory from COLMAP
`frames.txt`/`rigs.txt`; exported poses that already contain an RTK Sim(3)
alignment are not used as metric input.

For camera time `tau` and GNSS time `t`, with
`t = tau + clock_offset`, the sidecar predicts:

```text
position_antenna_ENU =
    R_ENU_visual (camera_center_visual + R_visual_camera lever_camera)
    + translation_ENU

baseline_ENU =
    R_ENU_visual R_visual_camera baseline_camera
```

Stereo scale remains fixed at 1.0. A free-scale Sim(3) is solved separately
as a diagnostic and is never combined with extrinsic calibration.

The bounded calibration variables are:

- three camera-to-primary-antenna lever corrections in optical camera axes;
- two direction corrections tangent to the configured antenna baseline;
- one camera/GNSS header-clock offset.

The baseline length remains the configured physical length. With one
dual-antenna vector, rotation about that vector is structurally unobservable;
that twist is not parameterized and stays exactly at the supplied TF prior.
The retained observable directions are converted back to one explicit
`T_body_camera`.

## Evidence and trust

One bounded rosbag pass preserves:

- exact camera, NavSatFix, RELPOS and PVT header and bag-log nanoseconds;
- full NavSatFix status, service, covariance type and 3×3 ENU covariance;
- RELPOS integer N/E/D, high-precision components, SI values, accuracies,
  heading, iTOW, carrier state and every validity flag;
- moving-base PVT fix/carrier/time status;
- SHA-256 of every reconstructed stereo pair.

For very large single-file MCAP bags, the interval is read as one forward scan
over the indexed chunk range. This avoids the installed reader's all-chunk
random-seek fan-out on fragmented exFAT media without changing message
contents or timestamps.

A correction is retained only when it passes:

1. covariance-weighted robust fixed-scale optimization;
2. projected-Jacobian observability after eliminating global SE(3);
3. configured bounds and deterministic multi-start convergence;
4. four leave-contiguous-interval temporal refits, including a minimum
   correction-to-fold-spread ratio;
5. reduced-model refitting after unsupported modes are removed;
6. aggregate and per-block held-out median, RMS, p95 and baseline-angle gates.

Every rejected or structurally unobservable parameter is restored exactly to
its configured prior.

## Artifact contents

- `observations.npz`: complete raw evidence plus explicitly named solver arrays
- `raw_visual_poses.npz`: raw chronological COLMAP rig trajectory
- `result.json`: candidates, observability, mode decisions and retained TF
- `residuals.npz`: before/after unaligned residuals for every eligible frame
- `observability.npz`: projected information/SVD evidence
- `temporal_folds.json`: fit/held-out frame IDs and per-block results
- `diagnostics.json`: stream, baseline and coordinate checks
- `provenance.json`: conventions, physical priors and immutable source hashes
- `REPORT.md`: human-readable verdict

No `viewmats.npy`, training configuration, or normal-pipeline activation is
written.

## Current headland audit

The strict result is:

```text
/home/jion_kubo/agromap4d_work/field_turn/segment/calibration_artifacts/
rtk_stereo_metric_integrity_v2_strict
```

It retains two corrections:

- optical-camera lever Z: `-0.03862885 m`;
- baseline tangent mode 0: `+0.01502952 rad` (`+0.8611 deg`).

All other lever, baseline-direction and clock modes remain at their priors.
On 90 frames in two held-out temporal blocks, median antenna error changes
from 5.65 cm to 3.71 cm, RMS from 5.70 cm to 4.58 cm, and median baseline-angle
error from 0.877° to 0.243°. Position p95 changes from 7.63 cm to 7.84 cm
(+2.8%), within the configured 5% tail ceiling; this tradeoff must remain
visible in any paper or deployment decision.

The artifact is an audit recommendation only. It is not enabled in normal
training.
