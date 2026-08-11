# Sealed visibility tile planning

Status: TilePlan schema v3 plus tile-aware cloud construction, GS training,
half-open core export, controlled scene publication, and seam evaluation are
implemented and tested. The cached two-tile headland GPU validation completed
on 2026-08-10 and passed every declared source, ownership, rendering, and seam
check. A TilePlan remains an input artifact rather than a map; only completed,
verified tile runs can publish a scene.

## Purpose

A long field should not be trained as one unbounded Gaussian model. RTK-Splat
first solves one metric global camera trajectory, then divides only the mapping
work. Every tile keeps that same ENU coordinate frame and the same camera poses.
There is no per-tile Sim(3), ICP, or scale correction.

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

Both modes concatenate exact core-owned tensors on CPU and render each source
validation view once with conservative frustum-only culling and no arbitrary
far plane. Quality is measured over all held-out frames and over pixels whose
metric depth lies within 1 m of an internal core boundary. A controlled gate
failure still publishes an explicitly named `scene.QUALITY_FAILED.ply` and
sealed evidence; it never silently becomes an accepted scene.

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

## Full 77-minute field analysis

The recording contains about 4,640 s and 69,088 stereo frames. The executed
route contains six long traversals plus substantial entry, exit, pause, and
headland motion. Roughly one quarter of the duration is outside the six
straight intervals. Recording-order traversals alternate direction, but
physical neighbouring rows include both same- and opposite-heading cases;
therefore neither “six row tiles” nor “no cross-row visibility” is justified
without the actual reconstructed support graph.

At stride five, the recording would contain roughly 13,800 frame pairs and
about 12,100 training frames. With the current 1,300-frame presentation budget,
that implies a lower bound near ten tiles before halo duplication. Metric
0.10 m sampling may reduce the total substantially, but that must be measured
after ingest rather than assumed.

There is currently no complete 77-minute contract-v2 depth segment or global
pose artifact. The route/GNSS audit can estimate workload, but it cannot
produce a scientifically final visibility plan. The server workflow must
ingest once, compute depth once, solve and seal one global pose artifact, and
then run `tiles-plan` without a forced count. The resulting plan—not waypoint
names—will determine whether headlands become their own cores or context for
neighbouring tiles.

The successful experiment validates the downstream GS tiling path only. The
full pose stage is still one global visual solve and has not been exercised at
roughly 13,800 stereo pairs/27,600 images. That frontend, matcher, and mapper
memory/runtime is now the largest scaling uncertainty. Arbitrary-tile
production publication is implemented; the remaining prerequisite is to
create and verify the full pose/TilePlan and tile runs.

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

Before the multi-day server build:

1. copy and checksum the portable depth segment on server-local storage and
   inventory CPU RAM,
   NVMe, CUDA, gsplat, and COLMAP;
2. build the full global pose and let the planner choose tile
   count automatically; and
3. complete one 2.5-million-Gaussian server tile as a resource and quality
   smoke test before launching the remainder.

The production launcher accepts restarting only the interrupted tile under a
new immutable attempt. Exact within-optimizer resume remains deliberately
unclaimed because the CUDA/MCMC path is not guaranteed bitwise deterministic
across process restarts.

The RTX 3090 and 40 TB storage are sufficient for sequential tile training at
the validated per-tile cap. They do not by themselves prove that the upstream
global pose solve or final arbitrary-tile evaluator scales. Reserve 0.5--1 TB
of fast working storage. The guarded first run defaults to a 120 GiB physical-
RAM gate (a 128 GB-class host) because a linear extrapolation of the accepted
Global Mapper memory is already near 91 GiB before OS/database headroom. A
provisional end-to-end estimate is 2--4 days, not a single overnight: roughly
10--15 tiles at 1.5--3.5 hours each on the 3090, plus an uncertain 10--30 hours
for frontend, matching, pose solve, and final publication.
