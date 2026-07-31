# Canonical Segment Contract (v2)

The mapping core starts at a validated, dataset-independent segment. Adapters
own ROS, bag, folder-layout, image-decoding, and dataset-specific details; the
core reads only this contract.

Contract v2 is intentionally write-once. An adapter builds a new segment in a
private staging directory, validates it, and atomically publishes it without
replacing an existing destination. Downstream stages must treat the published
segment as input and write new, named artifacts rather than altering adapter
evidence.

## Layout

```text
segment/
  segment_meta.json
  calibration.json
  frames.npz
  manifest.json
  images/
    ...
  depth/                       # required when a depth capability is declared
    ...
  observations/
    gnss.npz                   # required position/evidence table
    heading.npz                # required for dual RTK; otherwise optional
    imu.npz                    # required when imu_present is true
```

Paths stored in `frames.npz` are normalized paths relative to the segment.
Absolute paths and `..` traversal are rejected. An adapter may create the bulk
files itself or link to immutable source data, provided every declared path is
readable when the segment is validated.

`manifest.json` contains non-overlapping `train`, `val`, and `test` frame-ID
lists. Together the three lists must cover every frame exactly once.

## Frames and calibration

Every `frames.npz` contains:

- `frame_id`: contiguous integer IDs starting at zero.
- `timestamp_ns`: strictly increasing `int64` timestamps for the reference
  image stream.
- `left_image_path`: one path per frame.

A stereo segment additionally contains `right_image_path`,
`right_timestamp_ns`, and `stereo_sync_residual_ns`. The residual is exact
integer nanoseconds and must equal `right_timestamp_ns - timestamp_ns`.

Depth/RGB-D segments contain `depth_path`. Each depth file uses NPZ keys:

- `depth`: metric optical-z depth, in metres.
- `valid`: a boolean mask with the same image dimensions.

Initial poses are optional, but their fields are all-or-nothing:
`initial_viewmat` (`N x 4 x 4` world-to-left-camera matrices),
`initial_camera_center_m` (`N x 3` world-frame camera centres), and
`pose_valid` (`N`). Valid centres must agree with the corresponding matrices.

`calibration.json` declares `contract_version: 2` and at least a `left` camera.
Each camera records its model, width, height, `K`, and distortion coefficients.
A stereo segment also requires a right camera, a non-zero rigid
`T_right_left`, and:

```json
{"transform_conventions": {"T_right_left": "right_from_left"}}
```

`T_right_left` therefore maps a point expressed in the left optical frame into
the right optical frame. Camera optical axes follow OpenCV: x right, y down,
z forward.

## Capabilities

`segment_meta.json` declares every capability as a boolean:

```text
stereo, rgbd,
single_rtk, dual_rtk,
depth_recorded, depth_computed,
imu_present,
images_raw, images_rectified
```

At least one of `stereo` or `rgbd` is required. Exactly one image-geometry flag
is true. `single_rtk` and `dual_rtk` are mutually exclusive acquisition modes,
but both may be false for an oracle or other non-RTK position source.
`depth_recorded` and `depth_computed` are mutually exclusive; a stereo-only
segment may declare neither.

These flags describe evidence actually present in the segment. They are not
requests for a backend to synthesize missing data.

## Coordinate and sensor semantics

The following `segment_meta.json` records are required:

- `coordinate_frame`: `type` is `local_enu` or `cartesian_metric`,
  `world_frame_id` names the frame, and `units` is `m`. `local_enu` also
  requires a finite WGS84 latitude, longitude, ellipsoidal altitude,
  ellipsoid, and explicit vertical datum.
- `position_observation`: names the source `type`, physical `quantity`,
  `sensor_frame_id`, `coordinates: ENU_m`, and
  `covariance_frame: ENU_m2`. RTK observations must explicitly describe the
  GNSS antenna phase centre; they are not camera positions.
- `heading_observation`: required for dual RTK. It declares the
  `primary_to_secondary` baseline, ordered components
  `[north, east, down]`, and both antenna frame IDs.
- `initial_pose`: required when initial pose arrays are present. It identifies
  the left-camera frame and pose source, declares
  `position_quantity: left_camera_center`, records whether the antenna-to-camera
  lever arm was applied, and gives the three-axis translation-prior uncertainty
  in metres. `extrinsic_translation_sigma_frame_id` must equal the declared
  camera frame so covariance propagation has no hidden axis convention.
  RTK-derived camera centres must apply the lever arm explicitly.

World positions use metric ENU ordering `[east, north, up]`; dual-antenna
heading evidence deliberately retains the receiver's complete NED baseline
`[north, east, down]`. Adapters must preserve this distinction rather than
reducing the baseline to yaw or silently treating an antenna position as a
camera centre. Frame IDs and transform direction must be explicit so a new
robot can supply its own measured geometry without changing mapping code.

When depth is present, `depth_observation` must declare:

```json
{
  "format": "npz_depth_valid",
  "units": "m",
  "quantity": "optical_z",
  "aligned_to": "left",
  "invalid_convention": "valid=false and depth=0"
}
```

`aligned_to` may be `left` for stereo or `rgb` for an RGB-D source. The invalid
convention must be stated explicitly even if a dataset uses a different
sentinel before adaptation.

## Time and observation evidence

All canonical timestamps are exact `int64` nanoseconds. `timebase` records:

- `frame_timestamp_source`
- `observation_timestamp_source`
- `unit: ns`
- integer `association_clock_offset_ns`

Camera and position clocks remain separate. The clock offset defines the
association query; adapters must not rewrite original sensor stamps to make
streams appear synchronized.

Every row associated with a frame in `observations/gnss.npz` or
`observations/heading.npz` contains `frame_id`, `frame_timestamp_ns`,
`source_index`, and `source_timestamp_ns`. The position table additionally
carries per-frame `enu_m`, full `covariance_enu_m2`, raw integer `fix_status`
and `carrier_status`, boolean `position_valid`, and normalized
`position_quality`. The fixed vocabulary is:

```text
invalid, unknown_valid, standalone, differential, rtk_float, rtk_fixed, oracle
```

This keeps receiver-specific status integers as evidence while giving generic
frontends portable filtering semantics. ROS NavSat status `0` is a valid
standalone fix; an unavailable carrier solution is `-1` and does not by itself
invalidate a finite position. Invalid rows may use all-NaN ENU/covariance.
Every valid RTK row requires finite ENU and complete, symmetric,
positive-semidefinite covariance.

When a GNSS antenna observation becomes a camera-centre prior, the current
frontend adds the ENU GNSS covariance to the configured camera-frame
translation covariance after rotating it into ENU. The translation sigma's
frame is therefore mandatory. Heading uncertainty acting through the lever arm
and uncertainty of the lever arm itself are not yet propagated by this
frontend; that omission is recorded in the prior-stage report rather than
silently claiming a complete stochastic model.

Dual-RTK heading rows carry the complete `baseline_ned_m`,
`acc_heading_rad`, and `valid` flag. `observations/imu.npz`, when declared,
contains strictly increasing `timestamp_ns`, `accel_mps2`, and `gyro_radps`;
`orientation_xyzw` is optional.

Adapters should also retain the complete source streams as `raw_*` arrays,
including exact sensor and bag-log timestamps, covariance, status, carrier
state, baseline, accuracy, and validity when available. If raw arrays are
present, `raw_timestamp_ns` is mandatory and each associated
`source_timestamp_ns` must exactly equal
`raw_timestamp_ns[source_index]`. Repeated raw sensor stamps are preserved in
log order.

Visual matching and bundle adjustment can improve local consistency, but they
cannot manufacture global accuracy. Backends must use the recorded covariance
and status rather than assuming every dataset has centimetre-grade RTK.

## Adapter boundary and publication

An adapter must:

1. Decode and validate source images without exposing ROS or dataset imports to
   the core package.
2. Preserve exact timestamps, synchronization residuals, calibration, complete
   sensor evidence, and provenance.
3. Express camera, antenna, coordinate, time, depth, and transform semantics
   explicitly.
4. Declare only capabilities supported by the written evidence.
5. Produce deterministic, leakage-safe train/validation/test splits.
6. Publish to a new destination with `SegmentWriter`; an existing destination
   is an error and is never overwritten.

The supported dispatch boundary is `rtk_splat.adapters.registry.publish_from_config`.
Adding a new dataset means writing an adapter that publishes the same v2
contract, not adding conditionals to the mapper.

## One-time migration from the validated v1 headland layout

There is no permanent v1 compatibility path in the mapping core. The migration
utility exists only to convert the previously validated headland segment and
its metric-integrity observation archive:

```bash
python -m rtk_splat.adapters.migrate_v1_to_v2 \
  --source-segment /path/to/v1/segment \
  --destination-segment /path/to/new-v2/segment \
  --observations /path/to/observations.npz \
  --config /path/to/original-config.yaml
```

`--config` is optional and supplies rough geometry plus provenance; it does not
apply a calibration correction. The association tolerance defaults to 150 ms
and can be set with `--association-tolerance-ms`.

Migration is non-destructive and fail-closed:

- the v1 source is read but never modified;
- the destination must be new and outside the source tree;
- images and depth are referenced through symlinks instead of copied;
- exact stereo, GNSS, PVT, and dual-antenna samples are associated and
  preserved;
- input hashes and migration parameters are recorded; and
- any missing evidence, inconsistent hash, association outside the configured
  tolerance, or invalid v2 result aborts publication.

Because bulk data is linked, retain the source segment at its recorded absolute
path (or deliberately materialize a separate archival copy). After validation,
all new frontend and backend work should consume the v2 destination directly;
do not rerun the migration into the same path and do not re-enable v1 readers.
