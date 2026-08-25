# Full-field diagnostic milestone v1

Status: completed 2026-08-25. This is the current evidence summary for the
first complete 77-minute field rendering. It records a successful visual
scaling milestone and a failed production georeferencing result; those two
claims must remain separate.

## Repository state

GitHub PR #1 merged `feature/geodetic-full-field-gs-v1` into `main` as merge
commit `a8473ae8b46994a214631b1d4116670cca902ecd`. The merge preserves the 36
feature commits, including the generic sealed geodetic-submap sidecar,
diagnostic assembly, TilePlan, tiled training, seam evaluation, and layered
scene publisher. The reusable package contains no field path or fixed frame
window; recording-specific paths remain in configurations and launchers.

## Completed artifact

The sealed diagnostic scene is stored outside Git at:

```text
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-gs-v1/
  scene_artifacts/field1-geodetic-diagnostic-full-field-layered-scene-v1
```

It contains exactly 32 completed tile layers and evaluates 1,278 unique
held-out frames plus 1,265 seam frames from a 10,227-frame trajectory. The
authoritative representation is `sealed_tile_layers`: complete context
models are routed from the sealed TilePlan and blended using rendered depth,
alpha confidence, and core/context geometry. No held-out image or residual is
used to choose blend weights.

Whole-scene results:

| Metric | Value |
|---|---:|
| Masked PSNR | 21.1594 dB |
| Colour-corrected masked PSNR | 22.1405 dB |
| SSIM | 0.5085 |
| Colour-corrected LPIPS | 0.4149 |

Seam-band results:

| Metric | Value |
|---|---:|
| Masked PSNR | 21.8609 dB |
| Colour-corrected masked PSNR | 22.9246 dB |
| SSIM | 0.5079 |
| Colour-corrected LPIPS | 0.1619 |

All structural, source-binding, fixed-scale, calibration/baseline, visual
overlap, ownership, tile-completeness, and scene checks passed. This proves
that the implementation can build and render the complete field without a
single unbounded GS optimization.

## Why the artifact is diagnostic

The fixed-scale trajectory is structurally valid but failed independent
absolute RTK gates after full-field synchronization:

| Check | Measured | Production limit |
|---|---:|---:|
| Held-out RTK median | 366.483 mm | 120 mm maximum |
| Held-out inlier p95 | 279.334 mm | 200 mm maximum |
| Held-out robust inlier fraction | 41.09% | 80% minimum |
| Calibration inlier fraction | 42.64% | 80% minimum |
| Consecutive calibration outliers | 266 | 5 maximum |

The 150 mm value used during the exploratory continuation was a warning, not
a production gate. The final median exceeded that warning by 216.483 mm. The
artifact is therefore sealed as `diagnostic_render_only`, carries
`GEOREFERENCING_FAILED.json`, and is ineligible for metric-georeferencing or
survey-accuracy claims. It may be used as a clearly labelled qualitative
rendering result.

## Distant-ground and horizon artifact

The prominent central-field artifact is not only sky. The metric-depth mask
typically supervises about 75--78% of a frame, while the layered renderer has
alpha support over about 98--100%. In open central views, distant ground,
trees, and sky are often beyond reliable stereo depth, yet several nearby
spatial tile layers project low-confidence Gaussians into those pixels.

The completed held-out evidence has this descriptive relationship:

| Active layers | Frames | Mean masked PSNR | Mean alpha coverage |
|---:|---:|---:|---:|
| 1 | 48 | 24.010 dB | 92.03% |
| 2 | 180 | 22.305 dB | 97.96% |
| 3 | 342 | 21.447 dB | 97.84% |
| 4 | 239 | 20.984 dB | 98.93% |
| 5 | 196 | 20.346 dB | 99.36% |
| 6 | 232 | 20.126 dB | 99.71% |

This is a correlation, not a causal A/B: route position, scene content, and
layer count are confounded. It nevertheless supports the observed mechanism.
At field edges, closer trees/buildings provide better stereo support and fewer
tile layers are visible. In the open middle, the distant horizon occupies more
of the image and multiple independently trained context layers can create
floating or blurred splats.

The next rendering experiment must separately label:

1. finite metric foreground supported by depth;
2. top-connected, depthless sky/background; and
3. depthless distant ground near the horizon.

A conservative alpha/background compositor or bounded far-field model should
be evaluated first on frozen tile checkpoints. It must not change poses, hide
metric foreground errors, use held-out pixels for tuning, or relabel a visual
improvement as better georeferencing.

## Portable full-field PLY

The generic `rtk-splat-scene-export` command now creates an immutable viewer
bundle by concatenating every tile's opacity-pruned, uniquely core-owned PLY:

```bash
rtk-splat-scene-export \
  --source-scene /path/to/sealed/layered-scene \
  --destination /new/export-bundle
```

For this milestone it published:

```text
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-gs-v1/
  review_artifacts/field1-geodetic-full-field-ply-v1/
  scene.DIAGNOSTIC_ONLY.ply
```

The PLY contains 13,768,810 Gaussians, is 1,266,731,100 bytes, and has SHA-256
`b3cbc5cd2d9a1c1304ce452c514f630ed6e38be6921c9aab1c4ff55ca1382bef`.
Its terminal manifest and all 32 source hashes pass re-verification.

A standard PLY cannot represent the authoritative per-view context blend.
This union may expose harder seams or unsupported distant regions and must
remain `DIAGNOSTIC_ONLY`. Use a viewer that understands Gaussian scale,
rotation, opacity, and spherical harmonics; a generic mesh importer is not
equivalent.

Copy the sealed bundle from a local workstation with resumable `rsync`:

```bash
mkdir -p ~/Downloads/field1-full-ply
rsync -ah --partial --info=progress2 \
  imoroz@pit-bcs-gpu01-prod:/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-gs-v1/review_artifacts/field1-geodetic-full-field-ply-v1/ \
  ~/Downloads/field1-full-ply/
sha256sum ~/Downloads/field1-full-ply/scene.DIAGNOSTIC_ONLY.ply
```

The expected hash is the value above. [SuperSplat](https://superspl.at/editor)
is the preferred first viewer because it directly understands 3D Gaussian
splats. Blender does not natively interpret a 3DGS PLY as Gaussians; it needs a
compatible add-on, and 13.77 million Gaussians / 1.18 GiB may be impractical.
If the full file exceeds local browser or GPU memory, inspect the individual
tile PLYs or create a separately sealed reduced preview rather than altering
this full export.

## Next milestone

The next stage has two independent tracks:

1. diagnose the production georeferencing disagreement using calibration-only
   evidence, including camera/GNSS timing, the measured lever arm, transform
   conventions, and any recording-time base-frame/motion-mode changes; and
2. improve distant-ground/horizon rendering on frozen poses and tiles with a
   conservative, separately evaluated foreground/background policy.

Only after the first track passes all unchanged held-out RTK gates may the pose
artifact become production eligible. Only after the second improves frozen
whole-scene, seam, temporal-block, and lower-tail visual metrics should it
replace the diagnostic renderer.
