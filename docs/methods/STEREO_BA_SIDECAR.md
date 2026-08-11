# Historical calibrated stereo pose sidecar (accepted v1 experiment)

> Archived v1 experiment record. The current implementation is the sealed
> frontend plus named mapper backend described in `GLOBAL_MAPPER_BACKEND.md`.

This document preserves how the accepted `tile_turn3_stereo_ba` result was
produced. Its v1 launcher and flat CLI were deliberately retired during the
contract-v2 migration; the commands below are historical, not runnable
instructions. New comparisons use the sealed frontend launchers documented in
`docs/experiments/SEALED_FRONTEND_AB.md`.

This experiment replaces per-frame RTK attitude construction with locally
consistent stereo visual bundle-adjusted poses while preserving RTK as the
global ENU/georeferencing anchor. It does not use the IMU.

The original files remain authoritative and are never overwritten:

```text
segment/viewmats.npy
segment/cam_centers.npy
segment/init_cloud.npz
```

The experiment is isolated under:

```text
segment/pose_artifacts/colmap_stereo/
  colmap/                     # images are symlinks, database/models are local
  viewmats.npy                # created only after a successful export
  cam_centers.npy
  quality.json
  init_cloud.npz              # rebuilt from the refined poses
```

## One-time COLMAP installation

Use the tested lock file so the working `rtk-splat` environment is not
disturbed. It pins the CUDA/OpenBLAS FAISS runtime required by the current
conda-forge COLMAP package:

```bash
conda env create -f configs/environments/colmap-4.1.1-cuda.yml
```

The retired launcher previously verified the segment and tools without
starting compute:

```bash
scripts/reproduce/headland_stereo_ba.sh --check \
  --python /path/to/rtk-splat/python \
  --colmap /path/to/colmap
```

## Historical overnight A/B

The accepted run used:

```bash
cd /home/jion_kubo/agrorob_ws/src/AgroMap-4D/rtk_splat
scripts/reproduce/headland_stereo_ba.sh \
  --python /path/to/rtk-splat/python \
  --colmap /path/to/colmap
```

The script uses `colmap_stereo_repro` and `headland_stereo_ba_repro` by
default. It refuses to overwrite existing pose or run directories and
deliberately does not power off the machine. Its explicit stages are:

1. `stereo-prepare`: validate both rectified `CameraInfo` messages and the
   0.119846250 m baseline, then build a zero-copy rig workspace.
2. `stereo-solve`: DSP-SIFT, rig-aware sequential matching, and calibrated
   stereo incremental bundle adjustment with intrinsics/baseline fixed.
   Covariant DSP-SIFT is limited to eight CPU workers to stay within the
   machine's 30 GiB RAM; matching uses CUDA, while sparse BA stays on CPU to
   avoid an 8 GiB GPU-memory failure during the 2,688-image solve.
3. `stereo-export`: choose the largest model, require every left frame, and
   robustly Sim(3)-align camera centres to the immutable RTK ENU trajectory.
4. `cloud`: rebuild the stereo-depth cloud in the refined pose frame.
5. `train`: run the same 65k/2.5M settings as `tile_turn2`.

The contract-v2 CLI has no catch-all command. Long COLMAP and GS stages remain
explicit and write new artifacts.

## Acceptance gates

The exporter fails closed instead of publishing a questionable artifact:

- at least 95% registration, and currently all 1,344 left frames must be
  registered for dense frame indexing;
- stereo scale after RTK alignment must be in `[0.98, 1.02]`;
- RTK alignment uses a 0.15 m RANSAC gate and records median/p95/max residuals;
- every cloud records a pose hash, and training refuses a mismatched cloud;
- new resumable COLMAP markers fingerprint the command, COLMAP identity,
  calibration, rig, and frame manifest; changed inputs fail closed;
- all output rotations and camera centres are validated before use.

The original golden artifact has legacy plain-text completion markers because
it predates marker fingerprinting. They remain readable for preservation and
are reported explicitly as unverified; new sidecars use the stronger schema.

The original golden output is `tile_turn3_stereo_ba`; a new reproduction uses
the selected `--run` name. Compare it directly with `tile_turn2`. Pose
improvement should first appear as sharper leaf/ground edges, fewer doubled
structures, lower test-time alignment correction, and higher held-out masked
PSNR/SSIM with lower LPIPS.

## Paper-evaluation caveat

This first quality A/B estimates COLMAP poses using every image, including
the existing validation views. No validation RGB/depth enters GS training or
cloud initialization, but the validation images do influence their own SfM
poses. That is normal for a reconstruction-quality experiment, but it is not
the final paper protocol. Publication metrics need train-frame-only mapping
followed by frozen-map localization of held-out frames (and ideally a
separate traversal in `test`).
