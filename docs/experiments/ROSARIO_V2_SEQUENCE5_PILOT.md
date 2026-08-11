# Rosario v2 Sequence 5 pilot

This note freezes the completed bounded Rosario v2 pilot: immutable ingest,
production IR/stereo reconstruction and GS, diagnostic RGB-D training arms,
and the rejected unconstrained colour-pose probe. The IR result is eligible
under its declared PPK consistency gates; every colour result remains
diagnostic-only and ineligible for a metric georeferencing claim.

## Selected experiment

- Sequence: `2023-12-26-15-10-15`, Sequence 5.
- Window: 140--250 s relative to the first main-bag log timestamp
  `1703614215726730591` ns.
- Geometry: approximately 103.37 m of travel, two opposing traversals on
  adjacent rows, and one complete U-turn.
- Sampling: the generic quality profile resolves to 0.10 m; exact depth-safe
  publication produced 1,014 IR stereo keyframes.
- Quality input: the separately distributed Reach 1/Reach 2 PPK bag is an
  explicit offline method input. It is not described as online RTK.
- Excluded inputs: PGT, conventional-GNSS bags, every IMU topic, and wheel
  odometry.

The trajectory figure and its machine-readable values are outside the source
tree at:

```text
~/agromap4d_work/trajectory_plans/rosario_v2_seq5_recommended_140_250.png
~/agromap4d_work/trajectory_plans/rosario_v2_seq5_recommended_140_250.json
```

The two passes are adjacent rows, not the same physical corridor. Their
opposing views still provide a useful U-turn/overlap stress test, but they must
not be described as a literal same-row retrace.

## Completed immutable ingest

The PPK-input pilot was published and contract-validated on 2026-08-03 at:

```text
~/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment
~/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment.rgb_observations
```

- The bounded bag contained 1,602 synchronized IR candidates. Exact 0.10 m
  sampling selected 1,020 pairs; six without an exact recorded-depth
  association were explicitly dropped, leaving **1,014** canonical frames.
- The canonical segment spans **109.812452 s** and **103.367452 m**. Its
  effective spacing is **0.101440 m**.
- Left/right and depth/left timestamp residuals are exactly **0 ns** for every
  published frame.
- The separate sealed RGB artifact contains **1,645** monotonically timed
  observations over **109.745739 s** and has the same acquisition ID as the
  canonical segment.
- The segment is **1,766,735,534 bytes** and RGB artifact is **2,673,081,992
  bytes**. Ingest took **13 min 48.81 s** over USB 2.0 and peaked at **2.63 GB
  RSS**; downstream stages read the NVMe artifacts and do not repeat bag
  decoding.
- Contract validation reports rectified stereo, recorded depth, complete
  dual-position GNSS, and no IMU capability. Three sampled depth files retain
  metric `float32` optical-z, a boolean validity mask, and the original
  `uint16` units.

The durable preflight report is
`preflight_ppk_calibration_v1.json` in the work directory (SHA-256
`802fe8f273b093d14f63c06fc74ff85396538ee54db1d1587dc318c00a0fd5a2`).
The segment and RGB manifest SHA-256 values are respectively
`465f692be2f639c07e1c2cba136e259b863cb12781aa55de0f3b9def2289da78` and
`3a1eaf904ed24205dcd38f273acfddd7e4a4b51b5a87e83a04b8bb552e5d4fc7`.

## Input and calibration decisions

Calibration is treated as evidence, not as an unquestioned constant. Recorded
metadata and the published Rosario calibration are both preserved, compared,
and then checked against the images or depth wherever the recording permits.

| Item | Evidence | Current decision |
|---|---|---|
| IR stereo geometry | Recorded `CameraInfo.P` baseline 50.211037 mm; official Kalibr baseline 50.240891 mm; difference 0.029854 mm. Three real-image checks have stereo vertical-residual p95 0.798--0.903 px. The aggregate fraction below 1 px is 0.9692, versus a predeclared 0.90 gate. | Use the already rectified IR pixels with recorded `P`, retaining Kalibr as an A/B candidate. Trustworthy for the pilot. |
| Recorded depth | IR/depth headers differ by 0 ns in all three probes. Stereo-disparity scale ratios are 1.000088, 0.999971, and 1.000025; disparity-error p95 is 0.529--0.562 px. Depth and left `CameraInfo` dimensions/K/P/R/D match exactly, and their recorded static transform is 0 m/0 degrees. | Preserve raw `uint16`, publish metric `float32` optical-z plus a validity mask, and retain scale/sentinel provenance. Trustworthy for the pilot. |
| PPK position | Raw NavSatFix covariance is static `[1, 1, 4]` m2 and is preserved. The configured effective sigma `[0.10, 0.10, 0.20]` m is an external conservative prior, not a receiver claim or survey result. Carrier state is unavailable. | Use as `unknown_valid` postprocessed position evidence. Do not claim centimetre absolute accuracy from this file alone. |
| Dual-position heading | Reach 2 is interpolated at exact Reach 1 headers. Raw receiver skew is +10 ms; baseline median is 0.79294 m with 100% inside the 0.8 +/- 0.02 m gate. Straight-motion disagreement is 2.27 degrees for the selected sign and 177.73 degrees for the reversed sign. | Use the full synchronized two-position baseline and retain both raw receiver streams. Trustworthy for the pilot. |
| Camera-to-primary-antenna transform | The official URDF chain is internally consistent, but it was not independently surveyed on this recording. A recorded `camera_left` TF has incompatible frame semantics and is rejected rather than silently substituted. | Rough prior only. It must pass the post-reconstruction metric-integrity/observability audit before any absolute camera-georeferencing claim. |
| IR-to-RGB geometry and clock | Recorded factory TF and official Kalibr differ by 1.50 mm and 0.252 degrees. RGB is approximately synchronous in the bounded probes, but existing cross-spectral checks are not sufficient to jointly validate extrinsic and clock. | Preserve the full RGB stream and both priors. Every current RGB-D transfer is diagnostic-only, including one with low held-out residuals, until a separate joint extrinsic/clock observability method is implemented and validated. |
| Online Reach streams | The two online positions do not preserve the physical antenna baseline; no sample passes the strict 0.8 +/- 0.02 m test in this window. | Keep only as a deliberately rejected/degraded ablation. Never silently fall back from PPK to this stream. |

No calibration can be proven “100% accurate” from metadata. The policy here is
fail-closed: observable quantities must pass measured gates; unobservable
quantities retain their priors and an explicit untrusted status.

## Implementation boundary

Dataset-specific work remains in `rtk_splat.adapters.ros1_rosario_v2`. The
adapter reads ROS 1, handles Rosario topic/frame defects, and publishes the
same immutable segment contract used by the generic frontend and backends.
No Rosario topic, filename, calibration constant, or no-IMU special case was
added to the mapper, bundle adjustment, cloud builder, or Gaussian trainer.

The canonical segment contains rectified grayscale IR stereo, recorded metric
depth, raw/effective GNSS covariance, full dual-position evidence, and the
rough camera/antenna prior. RGB is a separate sealed observation artifact. A
generic post-pose workflow motion-compensates recorded depth into RGB using the
actual depth and RGB timestamps. Sealed held-out registration evidence can be
evaluated and recorded, but it cannot promote the current output beyond
diagnostic-only: constant motion can make camera translation and clock offset
jointly unobservable even when separate checks appear to pass.

## Reproduction and validation commands

Quality-input preflight:

```bash
PYTHONPATH=src python -m rtk_splat.adapters.ros1_rosario_v2 preflight \
  --config configs/sequences/rosario_v2_sequence5_ppk_140_250.yaml
```

The existing artifacts are immutable, so rerunning ingest at the configured
work directory correctly refuses to overwrite them. A reproduction must use a
new work directory; validation of the completed artifact is safe in place:

```bash
rtk-splat ingest \
  --config configs/sequences/rosario_v2_sequence5_ppk_140_250.yaml \
  --workdir ~/agromap4d_work/rosario_v2_sequence5_ppk_140_250_reproduction_v1

rtk-splat validate \
  --config configs/sequences/rosario_v2_sequence5_ppk_140_250.yaml
```

The online ablation is intentionally separate:

```text
configs/sequences/rosario_v2_sequence5_online_140_250.yaml
```

It is expected to fail its dual-baseline preflight on the current recording.

## Completed IR/depth baseline

The downstream pose-to-GS run completed on 2026-08-04 and is sealed in
`scripts/runs/rosario_v2_sequence5_ppk_140_250_overnight.sh`. It consumes the
already materialized NVMe segment, not the ROS bags or Buffalo disk. The
launcher pins the input/config hashes, verifies the runtime and resources,
then runs the all-frame GPU frontend, bounded Global Mapper, fixed-scale ENU
pose export, recorded-depth cloud construction, and grayscale-IR GS training.

Measured results:

- 1,014/1,014 frames and 2,028/2,028 stereo images registered;
- 0.80256 px mean reprojection error and 38.679 mean track length;
- production fixed-scale export with 0.16186 m held-out RTK median and
  0.31344 m p95 consistency residual;
- 44,400 GS presentations in 3 h 15 min; and
- selected step 44,000 corrected masked PSNR 21.15047 with 2,462,523 model
  Gaussians.

Run the read-only preflight first, then start the pipeline with sleep inhibited:

```bash
cd ~/agrorob_ws/src/AgroMap-4D/rtk_splat
bash scripts/runs/rosario_v2_sequence5_ppk_140_250_overnight.sh preflight
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
  --why="Rosario RTK-Splat overnight" \
  bash scripts/runs/rosario_v2_sequence5_ppk_140_250_overnight.sh run
```

The completed work directory is
`~/agromap4d_work/rosario_v2_sequence5_ppk_140_250_overnight_v1`.

The run first attempts a production pose and `splat.ply`. If and only if the
held-out RTK residual gate rejects an otherwise valid reconstruction, it
automatically retries under distinct `diagnostic-render` names and produces
`splat.DIAGNOSTIC_ONLY.ply`. That fallback allows visual inspection but is
explicitly ineligible for a metric georeferencing claim. Missing images,
failed visual-quality gates, corrupt artifacts, and pose-integrity failures
remain fatal and cannot be bypassed.

Completed stages can be verified and reused after a non-training interruption:

```bash
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
  --why="Resume Rosario RTK-Splat overnight" \
  bash scripts/runs/rosario_v2_sequence5_ppk_140_250_overnight.sh run \
  --resume-existing
```

The trainer does not serialize optimizer state. An interrupted training
directory is therefore preserved but deliberately not resumed as if it were a
complete run; use a new work directory for a clean restart.

The conventional viewer export kept 758,882 Gaussians after its 0.05 opacity
threshold and crop. The same sealed `params.pt` was non-destructively exported
with threshold 0 and no crop to:

```text
runs/rosario-seq5-ppk-140-250-gs-v1/exports/full-model-no-crop-v1/splat.ply
```

That PLY contains all 2,462,523 Gaussians. This is an export/display change;
the checkpoint and 21.15047 dB selected metric are unchanged.

## Completed diagnostic RGB experiments

The RGB launcher reused the accepted pose and did not rerun bags, COLMAP, or
the mapper:

```bash
bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_overnight.sh \
  preflight
bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_overnight.sh \
  run --acknowledge-diagnostic-render-only
```

The sealed streams yield exactly 1,012 unique in-gate RGB/depth associations
(886 train / 126 validation). Retained timestamp residuals are at most 62.5 us;
two approximately 66.7 ms endpoint mismatches are rejected. Full read-only
reprojection preflight measured 0.35326 aggregate RGB depth coverage and
0.54351 retention of source-valid depth. Automatic training resolves to 44,300
presentations.

This experiment is deliberately diagnostic. The current RGB factory
extrinsic/zero-clock prior is not independently jointly validated, no sealed
held-out correspondences exist, and no rolling-shutter correction is applied.
The only permitted output name is `splat.DIAGNOSTIC_ONLY.ply`.

Matched training outcomes:

| Arm | Selected step | Masked PSNR | Corrected masked PSNR | SSIM | LPIPS-CC |
|---|---:|---:|---:|---:|---:|
| v5 inherited pose | 20,000 | **20.066** | **20.340** | **0.4108** | 0.4721 |
| v6 free train-view pose | 18,000 | 19.847 | 20.259 | 0.4093 | **0.4336** |

V6 learned small unregularized corrections (median about 5.6 mm and 0.15
degrees) and looked sharper, but did not improve raw held-out PSNR. Evaluation
views were not refined, and the corrections have no gauge or physical prior;
they are evidence that image/pose consistency matters, not calibrated poses.

A separate 250-frame colour-only COLMAP probe registered 250/250 frames with
1.079 px mean sparse reprojection, but its reconstruction was invalid. One
adjacent transition rotated 178.8 degrees, a global Sim(3) fit had 5.807 m
camera-centre RMS and scale 0.5507, and temporal sub-blocks implied scales
0.432 and 266.674. The probe therefore rejects unconstrained monocular SfM on
this repetitive forward-motion window. Its printed "not the poses" conclusion
is not supported and must not be used as a scientific result.

## Frozen decision

Rosario colour development is frozen while the full 77-minute field map is the
active milestone. A future resumption must preserve the accepted metric
IR-stereo/PPK skeleton and test an asynchronous, bounded RGB timing/extrinsic
model on held-out temporal blocks. Unconstrained monocular poses or free
per-view training deltas cannot be promoted.

The headland score of 24.44 dB cannot be promised on Rosario: the cameras,
scene, speed, exposure, split, and supervision are different. The purpose of
this pilot is a defensible cross-dataset transfer test, not a guaranteed PSNR
match.
