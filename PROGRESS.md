# RTK-Splat progress

Last updated: 2026-08-25.

This file is the current status, not an execution log. Frozen experiment
evidence lives in `docs/experiments/`; the first complete field milestone is
recorded in `docs/milestones/FULL_FIELD_DIAGNOSTIC_V1.md`.

## Current milestone

The first complete 77-minute field pipeline has finished:

- 10,227 selected stereo frames were represented by one sealed, fixed-scale
  corrected-pose artifact;
- an automatic visibility TilePlan produced 32 tiles;
- every tile completed sequential Gaussian training on the RTX 4090;
- the sealed layered scene contains all 32 tiles and evaluates 1,278 unique
  held-out frames plus 1,265 seam frames; and
- the scene passed structural, source-binding, tile inventory, overlap,
  baseline, calibration, scale, rendering, and seam checks.

The whole-scene visual result is 21.1594 dB masked PSNR, 22.1405 dB
colour-corrected masked PSNR, 0.5085 SSIM, and 0.4149 colour-corrected LPIPS.
This is a successful full-field rendering and bounded-compute milestone.

It is not yet a production geodetic result. The final synchronized trajectory
failed the unchanged independent RTK gates: 366.483 mm held-out median versus
120 mm allowed, 279.334 mm inlier p95 versus 200 mm allowed, and 41.09% robust
inliers versus 80% required. The scene is therefore sealed as
`diagnostic_render_only`; no code or documentation may promote it to a metric
georeferencing claim.

## What is accepted

- The July 2026 1,344-frame headland incremental stereo-BA reference remains
  the strongest measured local visual result: 24.4449 dB masked PSNR and
  25.8480 dB colour-corrected masked PSNR.
- The reduced Global Mapper headland control retained every frame with nearly
  identical visual quality and much shorter mapping time.
- Contract v2, sealed mapper-neutral frontends, fixed-calibration stereo
  backends, generic geodetic submaps, raw-GNSS auditing, automatic TilePlan,
  tile-aware training, half-open ownership, layered scene publication, and
  seam evaluation are implemented and tested.
- The two-tile headland controlled A/B passed all declared scene/seam checks.
- The complete field proves that the bounded tiled renderer scales to this
  recording without one monolithic GS optimization.

Exact historical values and hashes are in `docs/experiments/golden/`,
`docs/methods/TILED_SCENE.md`, and
`docs/milestones/FULL_FIELD_DIAGNOSTIC_V1.md`.

## Active findings

### Geodetic acceptance

Local submaps can pass independently and overlap visually below a millimetre,
while the leakage-free synchronized trajectory still disagrees with held-out
RTK. Continuous-boundary and calibration-only timing/lever-arm probes did not
justify weakening the 120 mm gate. The remaining diagnosis must use calibration
observations only and explicitly test camera/GNSS timing, lever-arm convention,
world/local transform composition, recording-time base-frame or motion-mode
changes, and gradual visual drift. Held-out RTK remains final evaluation only.

### Central-field rendering

The conspicuous upper-image artifact is a distant-ground/horizon problem, not
only a sky mask problem. Reliable metric-depth supervision covers roughly
75--78% of typical held-out frames, while layered alpha coverage is often
98--100%. Open central views can blend four to six independently trained
context layers in depthless distant regions; edge views often have closer
depth-supported trees or buildings and fewer active layers. This relationship
is descriptive rather than a causal A/B.

The next visual experiment should keep poses and tile checkpoints frozen,
separate depth-supported foreground from top-connected sky/background and
depthless distant ground, then compare a conservative compositor or bounded
far-field representation. It must report whole-scene, seam, temporal-block,
and lower-tail metrics and must not use held-out pixels for tuning.

## Repository state

GitHub PR #1 merged the full feature history into `main` at
`a8473ae8b46994a214631b1d4116670cca902ecd`. The reusable Python package is
dataset-neutral; field-specific paths and recording identities are confined to
configuration and reproduction launchers.

Large pilot roots remain outside Git and are immutable. Several apparently
old `probe`, `fix`, and `diagnostic` roots are still referenced by the accepted
pose and tile provenance and must not be deleted. See
`docs/operations/ARTIFACT_RETENTION.md` before any storage cleanup.

## Active next stage

1. Archive and independently rehash the complete diagnostic milestone.
2. Run a calibration-only geodetic root-cause study without weakening gates.
3. Run a frozen-checkpoint horizon/layer-compositing A/B.
4. Accept a new production pose only if every existing RTK, calibration,
   baseline, scale, visual, track, and overlap gate passes.
5. Republish and retrain only after a proven generic correction; otherwise
   retain this result as a diagnostic visual milestone.

Gaussian Splatting is not run merely to test geodetic hypotheses. Production
publication remains fail-closed.
