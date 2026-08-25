# Active RTK-Splat roadmap

Last updated: 2026-08-25.

The complete field has been rendered successfully as a sealed diagnostic
32-tile scene. It is not production-georeferenced because the synchronized
trajectory failed independent RTK gates. See
`docs/milestones/FULL_FIELD_DIAGNOSTIC_V1.md` for the measured result.

## Milestone closeout

- [x] Merge the full-field implementation into `main` through GitHub PR #1.
- [x] Train all 32 automatically planned field tiles.
- [x] Publish and verify the sealed layered diagnostic scene.
- [x] Produce a portable, hash-bound full-field Gaussian PLY.
- [x] Record the visual result, geodetic failure, horizon finding, and artifact
  retention set.
- [ ] Copy the review bundle and PLY to independent storage and verify hashes.
- [ ] Remove only clean merged Git worktrees; retain every provenance-linked
  pilot/run root.

## Production geodetic acceptance

- [ ] Build one calibration-only audit joining camera timestamps, raw GNSS,
  covariance/status, vehicle speed/heading/turn state, configured lever arm,
  and recording-time base-frame or steering-mode state when available.
- [ ] Test timestamp latency and lever-arm/frame conventions by blocked
  calibration cross-validation. Never fit, tune, select, or filter with
  held-out RTK.
- [ ] Separate a transform/software defect from raw-sensor limitations and
  visual/GNSS trajectory tension.
- [ ] Implement only a proven generic correction with sealed decision evidence,
  tampering tests, focused tests, and the complete suite.
- [ ] Rerun the minimum boundary probe before any full-field pose or GS rerun.
- [ ] Keep the 120 mm median, 200 mm p95, 10 mm regression, robust-inlier,
  calibration, visual, track, baseline, overlap, and fixed-scale gates
  unchanged.

## Distant-ground and horizon quality

- [ ] On frozen accepted tile checkpoints, classify depth-supported foreground,
  top-connected depthless sky/background, and depthless distant ground.
- [ ] A/B one-layer and multi-layer rendering on predeclared calibration views
  to isolate cross-tile context artifacts.
- [ ] Test conservative confidence/alpha compositing or a bounded far-field
  background without modifying metric foreground or camera poses.
- [ ] Fix all settings before one held-out evaluation; report whole-scene,
  seam, route-block, and lower-tail results.
- [ ] Add a portable reduced-resolution review export if the 1.18 GiB PLY is
  impractical for common viewers.

## Release and reproducibility

- [ ] Tag the merged diagnostic milestone after independent artifact archival.
- [ ] Keep generic behavior in `src/rtk_splat/`; keep field paths and exact run
  identities in configs and reproduction launchers only.
- [ ] Turn `SERVER_RUN.md` and the server Codex handoff into clearly marked
  historical records; never rerun them against immutable completed outputs.
- [ ] Delete remote feature branches only after the merged PR and artifact
  archive are confirmed sufficient.

## Deferred

- Adaptive dense/balanced/sparse headland efficiency A/B.
- Independent target-based stereo recalibration.
- Hardware camera/GNSS synchronization, independently surveyed lever arm, and
  surveyed ground-control/check points.
- Right-camera photometric supervision and learned stereo depth validation.
- Production colour Rosario mapping pending independent timing/extrinsic
  evidence.
