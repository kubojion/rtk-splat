# RTK-Splat TODO

Last updated: 2026-08-11

This is the only active project tracker. Evidence and completed experiment
details belong in `PROGRESS.md` and `docs/experiments/`, not here. Check an
item only after its artifact or test result exists.

## Now: full-field headland map

- [x] Selected Option A on 2026-08-08: disjoint ENU cores, visibility-derived
  training halos, deterministic half-open core ownership, a canonical tiled
  scene manifest, and an optional concatenated viewer PLY. Do not use ICP or
  free per-tile Sim(3); add a fixed-scale SE(3) graph only if seam tests fail.
- [x] Add a sealed, hash-bound TilePlan sidecar and a data-driven visibility
  planner. Reference one immutable segment and pose artifact by frame ID; do
  not copy or reseal images, depth, or poses per tile. The full recording will
  therefore be ingested and depth-derived once upstream. Detect structure from
  projected metric-depth support and solved camera poses rather than requiring
  waypoint labels. Do not hard-code six tiles or along-row-only halos.
  TilePlan schema v3 uses projected metric depth/global poses, seals all image/depth/
  pose bytes, preserves splits, derives training capacity, gates core support,
  and propagates non-production pose status.
- [x] Make `cloud` and `train` consume a verified TilePlan plus tile ID without
  copying the source segment. Measure each tile's initial-cloud/Gaussian demand
  and fail before training when its resolved GPU budget is insufficient.
- [x] Add deterministic half-open core-cropped export, scene-manifest
  publication, and held-out seam/overlap evaluation.
- [x] Train two overlapping tiles from the cached headland segment and pass
  source-binding, ownership, whole-scene rendering, and 1 m seam-band gates.
  The merged scene improved over the same-code monolith by 0.305 dB masked and
  0.380 dB corrected masked PSNR. Its historical pose remains
  `legacy_unassessed`, so this is rendering/seam evidence rather than a new
  metric-georeferencing claim.
- [x] Choose and implement the first-server recovery policy. Completed stages
  and tiles are re-verified and reused; an interrupted tile is preserved and
  restarted under a new immutable attempt. Exact optimizer-level resume remains
  later work because the CUDA/MCMC path cannot yet promise exact continuation.
- [x] Add an arbitrary-number-of-tiles production launcher and a production
  scene publisher that does not require an infeasible full-field monolithic GS
  control. Preserve per-tile verification and evaluate absolute held-out,
  seam, georeferencing, ownership, and completeness evidence.
- [ ] Add per-frame and temporal-block regression reporting to the production
  scene gate. The accepted two-tile mean improved, but 28/168 held-out frames
  lost more than 0.3 dB raw masked PSNR; five lost more than 1 dB.
- [x] Ingest and depth-process the 77-minute bag locally and publish the
  portable checksum-sealed segment.
- [ ] Finish copying the portable segment to
  `/data/jkobo/rtk-splat/datasets/field1_0703_full77/segment`, restore ownership
  of only the private project subtree to `imoroz`, and pass its terminal hash
  verification.
- [x] Record the initial server hardware inventory: idle RTX 4090 24 GB,
  driver 580.95.05, 125 GiB reported RAM, 24 CPU threads, and about 17 TiB
  free below `/data`. System `nvcc` and COLMAP are absent, as expected.
- [ ] Commit and push the current full-field plus environment-bootstrap changes;
  remote `main` was seven commits behind local `HEAD` before this prompt's
  edits. Then pull that exact clean commit on the server.
- [ ] Pull the environment-bootstrap/4090-preflight commit on the server,
  install the two isolated prefix Conda environments, and pass bootstrap plus
  full server preflight. Do not modify system CUDA, base Conda, or other users.
- [ ] On the server, solve and seal the complete global pose artifact, then
  generate the final automatic TilePlan.
  The full pose solve, not per-tile GS, is now the main unmeasured scaling risk.
- [ ] Run one complete 2.5-million-Gaussian tile on the RTX 4090 to measure
  wall time, VRAM, RAM, and output size before launching the remaining tiles.
- [ ] Build and validate the full 77-minute field as independently sealed,
  georeferenced tiles; publish a scene manifest and optional viewer export.

## Decisions before the server run

- [ ] Decide how to record the missing historical reduced-Global compact pose
  artifact after the 2026-08-09 workspace cleanup. The old GS metrics remain,
  but its exact external verifier cannot pass unless an external backup is
  restored. This does not block tiling: the complete modern AB03 GPU pose,
  cloud, model, and 65k quality control remain intact.
- [ ] Defer the adaptive frame-density A/B and retain all globally registered
  views for the first Option A validation. Disk space is no longer the blocker
  (91 GiB was free after cleanup), but before a later causal A/B add a
  current-tree `all` control or shared feature cache.
- [ ] Define the public package boundary for dataset-specific adapters and run
  launchers. Keep the mapping package generic; do not publish personal paths,
  delayed schedulers, or one-off experiment orchestration as its main API.

## Generic package and configuration

- [ ] Design a short public mapping configuration that declares only input
  contract, calibration/pose source, quality profile, and explicit overrides.
  Derive sampling, depth range, iteration count, Gaussian budget, and tile size
  from measured data and hardware, and seal every resolved value with its
  inputs. Do not implement this until the full-field design is selected.
- [ ] Freeze or commit the current uncommitted CitrusFarm/Rosario work before
  moving or retiring experiment scripts.
- [ ] Move reproducibility launchers and machine scheduling outside the
  installed package contract; retain only generic, tested workflow commands in
  the release-facing repository surface.
- [ ] After the two-tile behavior is frozen, split the large TilePlan workflow
  module into a pure visibility/planning module and a sealed artifact/workflow
  module without changing schema v3 or its public CLI.
- [ ] Add isolated CI, wheel-install testing, licensing, and a small release
  example that starts from a published segment contract.

## Research and evaluation

- [ ] Compare full-field tiling against the accepted single-headland model:
  masked PSNR/SSIM/LPIPS, fixed-scale ENU residuals, overlap geometry, seam
  renders, completeness, runtime, RAM/VRAM, and storage.
- [ ] Establish fair public baselines and fixed splits before making SOTA
  claims. Report sensor inputs and do not compare PSNR across different scenes.
- [ ] Keep Rosario colour development frozen unless the full-field work is
  complete or a bounded, held-out asynchronous RGB/stereo pose experiment is
  explicitly resumed.

## Recently completed

- [x] Accepted incremental stereo-BA and reduced-Global headland results are
  frozen as golden regression references.
- [x] Modern mapper-neutral GPU frontend + Global + 65k GS reproduced the
  golden headland quality (24.364 dB masked, 25.733 dB corrected masked;
  -0.081/-0.115 dB from the accepted reference).
- [x] GPU/CPU feature-pose A/B completed; both registered 2,688/2,688 images,
  while GPU extraction was about 52.6x faster on this machine.
- [x] Documentation was audited and consolidated on 2026-08-08; obsolete outer
  plans now point to this tracker and measured experiment records.
- [x] Before workspace cleanup, source validation passed 315 tests plus 35
  subtests and the external incremental/Global verifiers passed 38/38 and
  42/42. After cleanup the incremental verifier still passes 38/38; the
  historical reduced-Global pose sidecar is absent, so its external verifier
  now passes only 6/20. The modern AB03 control is intact.
- [x] Completed the read-only Option A, disk, and adaptive-A/B readiness audit:
  two cached tiles should add about 2--5 GiB. Cleanup reduced the work directory
  from about 96 to 37 GiB and left about 91 GiB free.
- [x] Verified the six-row route geometry and current pair graph: opposing
  straight passes have no direct revisit edges, but adjacent rows are strongly
  connected visually through the continuously observed U-turns. This supports
  row-plus-turn training context without making it a dataset-specific rule.
- [x] Published and fully re-verified the selected cached headland TilePlan
  `headland-two-tile-seam-v1`: 919/879 training frames, 99.34%/99.36%
  training-covered core support, a measured 2.482 m halo, and honest
  `legacy_unassessed`/provisional georeferencing status. The automatic arm
  correctly retained one tile because 1,176 training frames fit the 1,300-frame
  budget.
- [x] Audited the whole source inventory and completed the stride-five ingest:
  10,227 pairs (about 8,949 eventual training views), implying at least about
  seven current-capacity tiles and likely roughly 8--12 after halo duplication.
  The final automatic topology is now blocked on the global pose, not depth or
  waypoint interpretation.
- [x] Source validation after TilePlan implementation passed 329 tests plus 35
  subtests on 2026-08-09.
- [x] Implemented the downstream Option A path: selected/context-cropped tile
  clouds, packed context supervision, tile-bound training, exact core export,
  atomic scene publication, and one-pass source validation evaluation.
- [x] Prepared and read-only preflighted the controlled cached-headland
  overnight launcher. It trains a same-code monolithic control plus two tiles,
  removes far-plane evaluation bias, uses deterministic colour correction, and
  gates 20,206,365 metric-depth seam pixels across 77 held-out views. The
  frozen AB03 model remains the absolute quality target because its trainer
  differs from the current implementation.
- [x] Final validation passed 362 tests plus 35 subtests on 2026-08-10;
  launcher syntax, full live source/TilePlan rehash, CUDA preflight, portable
  transfer/server orchestration, and the scientific reviewer audit also passed.
- [x] Completed the controlled two-tile GPU experiment on 2026-08-10. All 13
  sealed scene checks passed. Over all 168 held-out views the merged scene
  reached 24.653 dB masked / 26.109 dB corrected masked PSNR, SSIM 0.6126,
  and LPIPS-CC 0.2959, improving over the same-code monolith by
  +0.305/+0.380 dB, +0.0274 SSIM, and -0.0261 LPIPS-CC. The exact 1 m seam
  band also improved by +0.158/+0.092 dB. The 11 h 30 min controlled run used
  2.1 GiB and published a verified provisional scene with 3,751,281 core-owned
  Gaussians.
- [x] Prepared the guarded full-field handoff: memory-bounded ROS2 ingest,
  bounded sparse-heading handling, a self-contained checksum-sealed segment
  transfer, frozen full-field configuration, server environment preflight,
  arbitrary-tile retry orchestration, and production scene publication without
  a monolithic reference. Operational commands live in `SERVER_RUN.md`.
- [x] Passed the finalized 120-second real-bag ingest probe: 316 frames,
  29 min 18 s on USB 2, 453 MB peak RSS, and one explicitly invalid 190 ms
  heading association within the bounded sparse-dropout policy.
- [x] Completed local full-field ingest/depth/portable publication, made the
  repository public, and prepared a project-local server bootstrap that
  supports the discovered RTX 4090 without changing system or shared software.
- [x] Revalidated the complete source tree after the server adaptation: 364
  tests plus 35 subtests passed on 2026-08-11; shell syntax and diff checks
  passed.

## Update rule

At the end of every completed prompt: update the date, check only verified
work, add the next concrete blocker, and remove duplicated or obsolete tasks.
