# Canonical Data Contract

The mapping core begins at a canonical segment directory. How a dataset became
that directory is an adapter concern.

## Required segment

```text
segment/
  segment_meta.json
  manifest.json
  images/
    left_000000.jpg
    right_000000.jpg
    ...
  depth/
    000000.npz
    ...
  viewmats.npy
  cam_centers.npy
  camera_stamps.npy
  pose_stamps.npy
  stereo_dt_s.npy
```

`viewmats.npy` contains `N × 4 × 4` OpenCV world-to-camera matrices.
`cam_centers.npy` contains their corresponding world-frame centers. The base
artifact represents the adapter's initial metric pose source.

Each depth NPZ contains:

- `depth`: left-camera metric depth in meters.
- `valid`: a boolean validity mask with the same image dimensions.

`manifest.json` explicitly records train, validation, and test frame IDs. Cloud
construction uses training frames only.

## Portable calibration metadata

New stereo adapters must store:

```json
{
  "intrinsics": {"width": 0, "height": 0, "fx": 0, "fy": 0, "cx": 0, "cy": 0},
  "stereo_calibration": {
    "left": {"width": 0, "height": 0, "fx": 0, "fy": 0, "cx": 0, "cy": 0, "d": [], "r": [], "p": []},
    "right": {"width": 0, "height": 0, "fx": 0, "fy": 0, "cx": 0, "cy": 0, "d": [], "r": [], "p": []}
  },
  "n_frames": 0,
  "world_origin": {},
  "crs": {}
}
```

This lets a pose backend operate without ROS, original topic names, or the
source bag. Legacy segments without `stereo_calibration` may still fall back to
CameraInfo in their configured bag.

## Named pose artifacts

```text
segment/pose_artifacts/<name>/
  viewmats.npy
  cam_centers.npy
  init_cloud.npz
  quality.json
  ...
```

A named backend never overwrites base poses. Its cloud stores the SHA-256 pose
fingerprint from which it was constructed. Training fails if the selected poses
and cloud disagree.

Backend-specific databases, logs, models, and diagnostics live inside this
directory. Consumers depend only on validated poses, centers, quality metadata,
and the pose-matched cloud.

## Coordinate conventions

- World: metric, right-handed. Georeferenced runs use local ENU.
- Camera: OpenCV optical axes, x right, y down, z forward.
- Poses: world-to-camera matrices.
- Camera centers: world coordinates.
- Stereo images: rectified when consumed by built-in SGBM or calibrated-rig BA.
- Depth: positive distance along the left optical z axis, in meters.
- Time: seconds. `camera_stamps` and `pose_stamps` remain separate when clocks
  differ.

An adapter must record its geographic origin, vertical datum, time convention,
calibration provenance, and pose provenance. Unknown values should be explicit,
not invented.

## Adapter responsibilities

An adapter must:

1. Validate image count, dimensions, naming, and left/right synchronization.
2. Preserve original timestamps and calibration.
3. Produce full 6-DoF initial metric poses or clearly identify a limited pose
   source.
4. Record accuracy/status/covariance when available.
5. Create a deterministic split manifest.
6. Avoid embedding robot-specific TF assumptions in the mapping core.

RTK quality can vary between datasets. Visual matching and BA improve local
consistency, while RTK covariance should control global anchor weight in future
pose backends. Frame matching cannot manufacture absolute accuracy when the
global position source is weak; the artifact must report that uncertainty.

## Current compatibility

The project ROS 2 adapter and AgriGS folder importer have produced working
segments. CitrusFarm does not yet have an adapter and is therefore not listed
as supported. Its eventual adapter should convert its ROS 1 recordings,
calibration, registered depth, RTK, and ground truth into this contract without
changing the mapper.
