# Sealed visibility tile planning

Status: TilePlan schema v3 plus tile-aware cloud construction, GS training,
half-open core export, layered scene publication, and seam evaluation are
implemented and tested. The two-tile headland validation passed on 2026-08-10.
The first complete 10,227-frame field then automatically planned and trained 32
tiles and published a sealed diagnostic scene on 2026-08-25. A TilePlan remains
an input artifact rather than a map; only completed, verified tile runs can
publish a scene.

## Purpose

A long field should not be trained as one unbounded Gaussian model. RTK-Splat
first seals one complete fixed-scale camera trajectory, either from one solve or
an audited submap assembly, then divides only the mapping work. Every tile keeps
that same ENU coordinate frame and the same camera poses. There is no per-tile
Sim(3), ICP, or scale correction.

Each tile has two different regions:

- a disjoint spatial core that uniquely owns exported Gaussian centres; and
- an overlapping context halo whose visible cameras may supervise the core and
  seam.

This is the Option A policy. Stitching is deterministic ownership and
packaging in one ENU frame, not a second registration problem.

## Planner inputs and automatic decisions

Final planning requires one validated contract-v2 segment with metric depth and
one complete global pose artifact. It does not consume waypoint row labels,
fixed time windows, or the feature matcher's view-angle gate.

The planner:

1. derives a horizontal basis from the solved camera orientations, with the
   trajectory as a degenerate fallback;
2. projects confidence-filtered metric depth into ENU and quantizes it into
   spatial support cells;
3. measures which frames observe each candidate core and its halo;
4. recursively splits the most expensive rectangular core until every tile
   satisfies the resolved training-view budget; and
5. preserves the source train/validation/test split exactly.

The automatic training-frame budget is

```text
floor(iteration_budget /
      (image_presentations_per_view * supervised_cameras_per_frame))
```

An explicit numeric `train.iterations` is the iteration budget. With `auto`,
the profile's maximum iteration policy is the conservative planning budget.
The current quality profile resolves to 1,300 training frames per tile for
65,000 iterations, 50 presentations per view, and left-camera supervision.

The automatic context halo is half the measured 75th-percentile usable depth,
bounded to 2--8 m. A bounded fraction of depthless frames is allowed; those
frames inherit only the memberships of their nearest supported temporal
neighbours. Exceeding the configured fraction fails closed. Every core must
also meet a measured training-support coverage gate.

This capacity protects image presentations. The tile-aware cloud stage also
uses only the selected training views, crops projected points to the context,
records the measured initial point count, and applies the existing
VRAM/Gaussian-capacity policy before training.

## Immutable artifact

`tiles-plan` publishes exactly once under
`<workdir>/tile_plan_artifacts/<name>/`:

```text
tile_plan.json
visibility.npz
source_inventory.json
quality.json
provenance.json
plan.svg
manifest.json
```

The terminal manifest seals every file. The source inventory content-hashes
all selectable left/right images, every depth file, the segment contract, and
the complete pose artifact. The builder snapshots sources before loading poses
and again before atomic publication. The verifier recomputes support counts,
frame selection, capacity, split preservation, core support coverage, and
quality claims rather than trusting their JSON values.

Internal boundaries are lower-closed and upper-open. The scene's outer maximum
is included with a sealed one-nanometre numerical tolerance, so each in-scene
Gaussian centre has exactly one owner and a seam point cannot be duplicated.

Legacy or diagnostic pose status is propagated. A plan made from a
`legacy_unassessed` pose is marked `provisional: true` and is ineligible for a
metric georeferencing claim; planning never promotes it merely because its
geometry is usable for a rendering A/B.

## Command

```bash
rtk-splat tiles-plan \
  --config configs/sequences/<sequence>.yaml \
  --segment /path/to/depth-segment \
  --workdir /new/workdir \
  --pose-name <global-pose-name> \
  --pose-artifact-root /path/to/pose_artifacts \
  --tile-plan-name <name>
```

Omit `--tile-count` for production. The planner will keep a scene as one tile
when it fits. `--tile-count` exists only for controlled seam experiments.
Publication refuses an existing name.

The downstream stages consume the sealed plan directly; they do not materialize
or copy a per-tile segment:

```bash
rtk-splat cloud ... --tile-plan /path/to/plan --tile-id tile-0000
rtk-splat train ... --tile-plan /path/to/plan --tile-id tile-0000
```

Each cloud is namespaced by plan and tile, uses exactly that tile's training
frame IDs, and stores packed full-resolution context masks. Training applies
those masks to depth and photometric supervision, retains the full context
checkpoint, and exports only final Gaussian centres owned by the sealed core.
The terminal run provenance binds the source, plan, tile, frame selections,
cloud, poses, configuration, and training implementation.

`scene-publish` has two explicit modes. The default `controlled-ab` mode
requires every planned tile and one same-code monolithic control. It rejects a
control or tile made with different training code or scientific settings (the
output run name is the only ignored config field). The `production` mode is the
large-scene deployment path: it requires every planned tile exactly once,
verifies a shared trainer/configuration and production georeferencing, and
publishes absolute held-out/completeness/seam evidence without pretending an
infeasible full-field monolith exists. Non-production poses are rejected unless
`--diagnostic-scene` is explicit; that path can publish only a provisional
`DIAGNOSTIC_ONLY` artifact.

The hard-union mode concatenates exact core-owned tensors on CPU. The layered
mode instead keeps each complete context model sealed and routes a bounded set
of layers per view, blending rendered colour with depth, alpha confidence, and
core/context geometry. Each source validation view is evaluated once, with no
held-out residual used to select layers or weights. Quality is measured over
all held-out frames and over pixels whose metric depth lies within 1 m of an
internal core boundary. A controlled gate failure is explicitly labelled and
sealed; it never silently becomes an accepted scene.

## Cached headland validation

The automatic plan correctly retained one tile: the cached headland has 1,176
training frames, below the 1,300-frame budget. A deliberate two-tile seam arm
was then published as `headland-two-tile-seam-v1`:

| | tile 0000 | tile 0001 |
|---|---:|---:|
| Training frames | 919 | 879 |
| Validation frames | 132 | 126 |
| Core-visible training frames | 854 | 752 |
| Unique core support cells | 2,118 | 1,707 |
| Training-covered core support | 99.34% | 99.36% |

The split is near the middle of the dominant row direction, not at a hard-coded
row or U-turn. The 2.482 m measured halos share 622 training frames. The real
1 m seam band contains 20,206,365 valid metric-depth pixels across 77/168
held-out views. This is a conservative seam stress test rather than a boundary
inferred from row labels.

The controlled GPU run trained one current-code monolithic control and the two
tiles with the same scientific configuration. The terminal scene verifier and
all 13 declared quality checks pass:

| Held-out metric | Same-code monolith | Core-owned tiled scene | Change |
|---|---:|---:|---:|
| Masked PSNR | 24.3481 dB | **24.6528 dB** | **+0.3048 dB** |
| Corrected masked PSNR | 25.7284 dB | **26.1087 dB** | **+0.3803 dB** |
| SSIM | 0.5851 | **0.6126** | **+0.0274** |
| LPIPS-CC | 0.3221 | **0.2959** | **-0.0261** |

Every one of the 168 source validation views is rendered exactly once. In the
exact 1 m metric-depth seam band, the tiled scene improves masked/corrected
masked PSNR by **0.1580/0.0921 dB**, SSIM by **0.0278**, and LPIPS-CC by
**0.0110**. Representative seam renders show no ownership discontinuity.

Each tile reached its 2.5-million-Gaussian cap. Exact core ownership retained
1,926,976 and 1,824,305 tensors, for 3,751,281 in the CPU-resident union;
opacity filtering retained 2,258,580 in the 207,789,939-byte viewer PLY. The
largest held-out view used 2,699,217 Gaussians. The complete controlled run took
11 h 29 min 46 s on the RTX 3080 Laptop and used 2.1 GiB of new storage.

This is a successful bounded-capacity scaling result, not an equal-total-compute
or equal-total-parameter claim. The tiled arm used two 65k optimizations and up
to 5 million source Gaussians, while the control used one 65k/2.5-million model.
The average improved, but performance was not uniformly better: 28/168 held-out
frames lost more than 0.3 dB raw masked PSNR and five lost more than 1 dB. A
production evaluator should therefore report temporal-block and lower-tail
regressions in addition to means.

The source pose is a historical `legacy_unassessed` artifact, so this plan is
correctly provisional even though its older RTK audit passed. It is suitable
for comparing rendering and seams, not for creating a new metric-accuracy
claim.

## Completed 77-minute field result

The recording contains about 4,640 s and 69,088 raw stereo frames. Stride-five
ingest retained 10,227 pairs. The planner used reconstructed metric support and
visibility—not row names, time windows, or a forced count—and produced 32
tiles. All 32 trained sequentially at the frozen 65k-iteration,
2.5-million-Gaussian cap on the RTX 4090.

The sealed layered scene evaluates 1,278 unique held-out frames and 1,265 seam
frames. Its whole-scene masked/corrected masked PSNR is 21.1594/22.1405 dB,
SSIM is 0.5085, and corrected LPIPS is 0.4149. Seam-band masked/corrected
masked PSNR is 21.8609/22.9246 dB, SSIM is 0.5079, and corrected LPIPS is
0.1619. Every structural, source-binding, ownership, tile-completeness, visual
overlap, scale, baseline, and seam check passed.

The result is deliberately diagnostic. Its complete synchronized trajectory
failed independent production georeferencing gates: 366.483 mm held-out RTK
median versus 120 mm allowed, 279.334 mm inlier p95 versus 200 mm allowed, and
41.09% robust inliers versus 80% required. The scene is therefore labelled
`diagnostic_render_only` and may not support survey or metric-georeferencing
claims. Exact evidence and the distant-ground/horizon finding are in
`../milestones/FULL_FIELD_DIAGNOSTIC_V1.md`.

## Cached overnight reproduction

The launcher first trains a current-code 65k monolithic control, then
the two 65k tiles sequentially. This is necessary because the frozen AB03
24.364/25.733 dB result used an earlier trainer; it remains an absolute target,
not the causal tiling control. The scene gate uses the new control's dynamically
sealed hashes and separately reports the historical target delta.

```bash
bash scripts/experiments/headland_two_tile_overnight.sh preflight
bash scripts/experiments/headland_two_tile_overnight.sh run
```

The completed run took 11 h 29 min 46 s, close to its 12-hour estimate. The
launcher uses no `tmux`, runs one GPU job at a time, inhibits sleep, never
powers off the machine, and never reads the 77-minute bag. It can skip a
completed cloud, model, or scene only after verification. It does **not**
provide exact within-training resume: an interrupted optimization is retained
as evidence and must be restarted in a new workdir.

The sealed output is
`headland-two-tile-seam-controlled-v1/scene.PROVISIONAL.ply`, SHA-256
`ea7e3eb71c1d0ef50226f870da3ed620649de12a1f3d3e9d6eafb3955b92e398`.
The current cached pose is `legacy_unassessed`, so this local A/B remains
provisional and cannot create a new metric-georeferencing claim.

## Portable viewer export

The layered artifact remains authoritative because standard Gaussian PLY has
no representation for its per-view context routing and blend. For inspection,
`rtk-splat-scene-export` verifies the sealed scene and every tile binding, then
streams each opacity-pruned, uniquely core-owned tile PLY into one immutable
hard-union bundle:

```bash
rtk-splat-scene-export \
  --source-scene /path/to/sealed/layered-scene \
  --destination /new/export-bundle
```

The command refuses overwrite, rejects schema or hash disagreement, publishes
atomically, and preserves the source georeferencing status. A diagnostic source
therefore produces `scene.DIAGNOSTIC_ONLY.ply`. Because this hard union omits
context blending, it can show harder seams or worse unsupported distant regions
than the authoritative evaluation renders.
