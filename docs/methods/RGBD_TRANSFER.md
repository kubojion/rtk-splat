# Calibrated RGB-D transfer

`rtk_splat.workflows.rgbd_transfer` converts an immutable, mapped stereo
segment into an immutable RGB-D segment. It reuses an accepted production pose
artifact; it does not rerun COLMAP, alter bundle adjustment, or read IMU data.

Status: the workflow has completed on Rosario v2 and produced valid diagnostic
artifacts. It deliberately cannot promote them to a production metric claim;
the RGB timing/extrinsic state and rolling-shutter registration remain
unvalidated. Results belong in
`docs/experiments/ROSARIO_V2_SEQUENCE5_PILOT.md`, not in this method contract.

The workflow is dataset-independent. A dataset adapter owns the additional RGB
stream and publishes the artifact below beside, not inside, the canonical
stereo segment.

## RGB observation input, schema version 1

```text
rgb_observations/
├── rgb_observations_meta.json
├── frames.npz
├── calibration.json
├── manifest.json
└── images/
    ├── rgb_000000.png
    └── ...
```

`frames.npz` has these required, non-object arrays:

- `frame_id`: contiguous `int64`, `0..N-1`.
- `timestamp_ns`: strictly increasing `int64` sensor timestamps.
- `image_path`: Unicode, normalized relative paths.
- `log_timestamp_ns`: optional `int64` bag/log timestamps, one per frame.

No IR association is stored here. The transfer computes it from timestamps.
Each source depth frame selects at most one nearest corrected RGB observation,
so a higher-rate RGB stream is reduced deterministically to approximately the
source depth rate. The output records both original indices, both timestamps,
and signed residuals. Counts and exact indices for out-of-gate source depth
frames, duplicate-association losers, and unused RGB observations are retained
as evidence.

`rgb_observations_meta.json` is:

```json
{
  "schema_version": 1,
  "artifact_type": "calibrated_rgb_observations",
  "acquisition_id": "adapter-generated-opaque-id",
  "n_frames": 123,
  "timestamp_unit": "ns",
  "timestamp_source": "exact RGB sensor header",
  "clock": {
    "rgb_to_source_clock_offset_ns": 0,
    "provenance": {"method": "measured or verified clock relationship"}
  },
  "provenance": {"acquisition_id": "adapter-generated-opaque-id"}
}
```

The adapter generates one immutable `acquisition_id` before publishing either
artifact and stores it in both the canonical source segment and this sealed
sidecar. The transfer rejects a valid sidecar from any other recording; camera
frame names alone are not an acquisition binding.

The signed clock correction maps an RGB timestamp into the canonical source
left-camera clock:

```text
t_source_query = t_rgb + rgb_to_source_clock_offset_ns
```

This is one declared global offset, not an estimated per-frame correction.
The output stores every selected source timestamp and both signed residual
conventions. The primary transfer report uses
`corrected_rgb_timestamp - source_depth_timestamp`; the generic observation
sidecar uses its documented opposite convention, `source - query`. The report
states explicitly that a global offset may be insufficient and that no
rolling-shutter correction is applied or claimed.

`calibration.json` is:

```json
{
  "schema_version": 1,
  "camera_frame_id": "rgb_optical_rectified",
  "source_camera_frame_id": "ir1_optical_rectified",
  "source_camera_geometry": "source_segment_left_rectified",
  "rgb_camera_geometry": "recorded_factory_pinhole_direct",
  "camera": {
    "model": "PINHOLE",
    "width": 1280,
    "height": 720,
    "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
    "distortion": [0, 0, 0, 0]
  },
  "image_geometry": "rectified",
  "rectification": {
    "input_camera": {"model": "OPENCV", "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]], "distortion": [0.01, -0.02, 0, 0]},
    "method": "adapter-selected explicit pixel representation",
    "provenance": {"method": "recorded CameraInfo or calibrated rectification"}
  },
  "T_rgb_source_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "transform_convention": "rgb_from_source_camera",
  "extrinsic_translation_sigma_m": [0.005, 0.005, 0.005],
  "extrinsic_translation_sigma_frame_id": "rgb_optical_rectified",
  "extrinsic_provenance": {"method": "independent calibrated rig transform"}
}
```

The adapter must produce the declared pixels. The transfer accepts only an
explicit zero-distortion `PINHOLE` representation and copies it losslessly; it
does not choose or apply a lens model from a topic name. The geometry label can
be `output_camera_rectified` or an explicitly selected
`recorded_factory_pinhole_direct` representation. `rectification` retains the
full input model, selected method and evidence, including when the selected
factory representation makes the pixel transform identity. This keeps raw
Kalibr undistortion out of the generic transfer and prevents applying it twice.
`T_rgb_source_camera` must map the canonical rectified source-left frame into
the exact declared RGB representation.

`manifest.json` seals every metadata file and every referenced image:

```json
{
  "schema_version": 1,
  "artifact_type": "calibrated_rgb_observations",
  "files": {
    "frames.npz": {"sha256": "...", "size_bytes": 1234}
  }
}
```

## Source requirements

The canonical source must declare rectified stereo and validated depth with:

- `format: npz_depth_valid`
- `units: m`
- `quantity: optical_z`
- `aligned_to: left`

Each depth NPZ must contain same-sized `depth` and boolean `valid` arrays.
Raw sensor scale or sentinel values are never inferred here. The adapter must
first prove and convert those semantics. Every source frame must also provide
strictly increasing `depth_timestamp_ns` and
`depth_sync_residual_ns = depth_timestamp_ns - timestamp_ns`; the image-frame
timestamp is never silently substituted for depth acquisition time. The source
must declare exactly one of
single/dual RTK and must declare no IMU capability. The selected pose artifact
must be modern, manifest-sealed, production, `PASSED`, and eligible for metric
claims. Its sealed provenance must bind the exact source segment root,
acquisition ID, core-file hashes and image inventory copied through the
frontend/backend plan. Matching frame IDs and timestamps are not sufficient.

## Operation

For every source depth frame with a unique in-gate RGB match, the workflow:

1. Applies the sealed clock correction, selects the nearest corrected RGB
   observation without loosening the configured gate, and records unmatched
   source/RGB evidence.
2. Interpolates accepted and RTK-initial full SE(3) poses at the selected RGB
   timestamp, and independently interpolates the accepted pose at the explicit
   depth timestamp.
3. Applies the calibrated source-camera-to-RGB rigid transform.
4. Verifies and copies the adapter-selected rectified RGB pixels losslessly.
5. Motion-compensates metric source depth to the RGB time/frame and resolves
   collisions with a nearest-surface z-buffer.
6. Writes per-output reference-camera positions/covariance while leaving raw
   GNSS/heading evidence solely in the sealed source segment.
7. Reapplies the source artifact's absolute and covariance-aware RTK gates to
   the transferred camera centres. No target RTK values are fitted.
8. Requires the configured minimum depth/RGB association fraction, aggregate
   projected-depth image coverage, and retained source-depth fraction.

RGB and depth pixels alone do not generically prove a cross-modal calibration:
texture, foliage, occlusion and smooth surfaces can make weak edge scores pass
or fail for the wrong reason. Therefore a transfer without independent visual
evidence is published as `diagnostic_render_only`, remains `PASSED` for its RTK
transfer, and is explicitly ineligible for a metric production claim.

Optional validation accepts a manifest-sealed
`held_out_rgb_depth_registration_evidence` directory. It contains independent
held-out source-camera 3-D points, corresponding RGB pixels, explicit source
depth and RGB timestamps, and a declaration bound to the exact source segment,
source pose artifact, RGB artifact, RGB calibration and clock offset. The
workflow recomputes motion- and clock-compensated reprojection errors and
applies the configured correspondence-count, median and p95 pixel gates. A
supplied but failing or mismatched evidence artifact publishes nothing.

All RGB-D outputs remain diagnostic in the current method, including those
whose held-out residuals and separate clock/extrinsic sensitivity checks pass.
Constant-velocity motion can make clock offset and camera translation jointly
indistinguishable even when each separate check looks observable. Production
promotion is therefore disabled with
`joint_extrinsic_clock_observability_not_established`. A future production gate
requires a joint 7-state Jacobian rank/condition analysis across sufficiently
different motion blocks; this workflow does not pretend to solve that problem.

Failure publishes neither requested output. Successful outputs record hashes,
clock/extrinsic provenance, exact source/RGB indices, missing/unused counts,
signed synchronization residuals, valid-pixel counts, interpolation brackets,
effective covariance, and all gate results.

## Command

```bash
python -m rtk_splat.workflows.rgbd_transfer \
  --source-segment /path/to/ir_segment \
  --source-pose-artifact /path/to/accepted_ir_pose \
  --rgb-observations /path/to/rgb_observations \
  --destination-segment /new/path/rgbd_segment \
  --destination-pose-artifact /new/path/pose_artifacts/rgb-transfer-v1 \
  --max-depth-sync-residual-ns 20000000 \
  --max-pose-interpolation-gap-ns 100000000 \
  --min-depth-rgb-association-fraction 0.95 \
  --min-projected-depth-coverage-fraction 0.01 \
  --min-projected-depth-retained-fraction 0.10
```

All five synchronization/projection thresholds above are required; there are
no hidden dataset timing or depth-survival defaults. Add
`--held-out-registration-evidence /path/to/sealed-evidence` only when genuine
independent correspondences exist; it strengthens diagnostic evidence but does
not promote the artifact to production. Its pixel thresholds are also recorded
in provenance. Raw RTK
and heading streams remain solely in the immutable source segment. The derived
segment contains only derived per-output reference-camera positions/covariance
and explicitly declares that no target RTK was acquired and no new single- or
dual-RTK capability exists.
