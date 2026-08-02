# Pipeline and Boundaries

## Status and evidence

The contract-v2 architecture for Phases 0, 1, 3, and 4 is implemented and
covered by unit and synthetic tests, and the real headland input has been
published under the normalized contract. The mapper-neutral COLMAP frontend
and adaptive keyframe arms are prepared for a controlled real-data A/B, but
that A/B has not yet been run. Historical headland results in `PROGRESS.md`
were produced before this frontend refactor and remain the measured quality
and runtime reference.

An ordered ROS 1 CitrusFarm adapter, robot/sequence profiles, and a staged
543--735 s reproduction launcher are also implemented. The full window has
completed ingest, SGBM, sealed frontend, and a 2,990-image visual model. A
separately named position-prior refinement sidecar is implemented and was run
with 897 calibration priors and 598 priors absent from its factors. It improved but failed
the RTK gates, so no pose or GS artifact was published.
Phase 2 was deliberately skipped; custom RTK-factor submaps and
rendering-quality changes remain outside the current implementation.

## Artifact graph

```text
dataset-specific adapter
        |
        v
immutable contract-v2 segment
        |
        v
sealed mapper-neutral frontend
  images + rig + features + priors + verified pairs
        |
        +-------------------------+
        |                         |
        v                         v
private Global snapshot     private incremental snapshot
        |                         |
        +-----------+-------------+
                    v
       optional RTK position-refinement sidecar
          calibration priors only; heldouts removed
                    |
                    v
          named metric pose artifact
                    |
                    v
        pose-matched cloud -> GS run
```

Each expensive or scientifically meaningful boundary is a named artifact.
Stages fail rather than silently overwrite an existing artifact. A backend
works from its own verified database snapshot, so Global and incremental
experiments cannot contaminate the sealed frontend or one another.

## Core and adapter boundary

`rtk_splat/core/` is the small, dataset-independent core. It defines the canonical
segment reader/writer, geometry, generic configuration, pose/cloud artifact
primitives, and golden verification. Source ingestion, visual frontends,
mapper/training backends, diagnostics, and orchestration live in sibling
subpackages of the single installed `rtk_splat` namespace. Import-isolation
tests enforce that boundary.

`rtk_splat/adapters/` owns all source-specific behavior:

- ROS version, bag storage, topics, and message types;
- image decoding and timestamp synchronization;
- calibration extraction;
- GNSS status and covariance decoding;
- optional dual-antenna heading data; and
- external trajectory or ground-truth parsing.

The registry currently implements `ros2_zed_ublox`, `ros1_citrusfarm`, and
`agrigs`. The Citrus adapter reads an explicitly ordered ROS 1 bag chain
without a ROS installation, validates chunk gaps/overlaps and required topics,
decodes raw rectified stereo losslessly, preserves Piksi timestamps,
status/covariance and ROS bag-log times, estimates single-antenna course
heading, and samples by travelled distance. Its topic and geometry facts live
in robot/sequence YAML rather than the mapping core. Arbitrary unconfigured
topic/message layouts are not inferred automatically.

## Immutable contract-v2 segment

A published segment retains:

- exact integer nanosecond timestamps;
- both rectified RGB streams and both camera calibrations;
- the metric stereo transform and explicit transform semantics;
- RTK position, complete covariance, and receiver status;
- the complete dual-antenna N/E/D baseline when available; and
- explicit capabilities for optional data rather than fabricated defaults.

Single-antenna data is valid: it simply lacks the dual-antenna-heading
capability. Downstream code must react to that declared capability.

When computed SGBM depth is needed, `depth --derived-segment NEW_PATH`
publishes another immutable v2 segment. It symlinks the source images and adds
the computed depth products and capability metadata; it never edits the
ingested source segment.

The one-time migration has also been exercised on the real 1,344-frame
headland input. The normalized output is
`~/agromap4d_work/field_turn_contract_v2_normalized/segment`; its compact
hashes, exact association residuals, capabilities, and resource use are
recorded in `docs/experiments/migrations/headland_contract_v2.json`. Neither
the source v1 segment nor the earlier v2 migration was modified. No new COLMAP
or GS result is claimed from this migration yet.

The CitrusFarm primary candidate follows the same boundary. Its source segment
will contain RGB plus single-RTK evidence; SGBM then publishes a separately
named immutable depth segment. Recorded ZED depth and confidence topics are
inventoried in preflight provenance but deliberately withheld from this arm.
The real preflight measured a 0.119885 m stereo baseline and a +72.548749 ms
camera-to-GNSS clock offset, now frozen per sequence. Drift across the window
was -2.581 ms, within the configured 15 ms gate. The bounded smoke published
10 stereo frames and sealed all 52 planned frontend pairs; it does not prove
full-window mapper or GS quality.

## Sealed mapper-neutral frontend

The frontend is built once per experiment arm:

1. `frontend-build` symlinks canonical images and writes the immutable frame
   manifest, stereo rig, keyframe plan, match-pair plan, quality summaries,
   resolved configuration, and provenance.
2. `frontend-features` extracts features into the frontend database.
3. `frontend-rig` applies the calibrated stereo rig.
4. `frontend-priors` filters usable RTK evidence and inserts pose priors.
5. `frontend-match` verifies the planned correspondence graph and seals the
   transaction-consistent database, every finalized JSON/pair input, and the
   contents and resolved targets of all image symlinks.

The frontend does not run a mapper. Both backends receive an independently
verified copy of the same sealed evidence.

Every canonical timestamp retains both its exact left and right image names.
Keyframes reduce the expensive visual solve; they do not discard non-keyframe
images. Mandatory same-timestamp stereo pairs and attachment edges allow the
registration stage to restore every frame.

## Adaptive keyframes and match graph

The implemented presets are:

| Preset | Translation threshold | Rotation threshold |
|---|---:|---:|
| `all` | every frame | every frame |
| `dense` | 0.08 m | 1 deg |
| `balanced` | 0.10 m | 2 deg |
| `sparse` | 0.15 m | 3 deg |

All adaptive presets also impose a default 1.5 s maximum elapsed interval.
Planning can use translation, rotation, time, image quality, turn behavior,
and revisits. The match graph combines temporal and metric neighborhoods,
view direction, revisit links, mandatory stereo edges, and non-keyframe
attachment edges. Before optional edge pruning, it reserves a deterministic
physically admissible bounded-degree spanning graph across solve frames; an
unbridgeable or degree-infeasible solve fails with component diagnostics.

These policies are implemented and tested. Their registration, speed, and GS
quality on the real headland segment remain unmeasured.

## Mapper backends

Global is the default candidate; incremental is the optional fallback and
control. Both backends use the same lifecycle:

1. `backend-prepare` verifies the sealed frontend and creates a private
   transaction-consistent SQLite snapshot, including committed WAL state.
2. `backend-solve` solves only the selected keyframe images.
3. `backend-register` uses COLMAP's image registrator to recover all
   non-keyframe left and right images.
4. `backend-quality` checks registration, rig completeness, geometry, and
   provenance.
5. Optionally, `backend-refine-rtk` clones the completed backend, removes
   temporal holdout priors, runs position-prior BA, and checks visual quality,
   stereo scale/rig integrity, and blocks absent from its optimizer factors.
6. `backend-export` publishes a new pose artifact only after the selected
   mapper/refinement path passes every gate.

The corresponding Python API is `MapperConfig`,
`prepare_mapper_backend()`, `run_mapper_solve()`,
`run_image_registration()`, `run_quality_summary()`, and
`export_pose_artifact()` in `rtk_splat.backends.mapper`.

The optional API is `RtkRefinementConfig`, `prepare_rtk_refinement()`,
`run_rtk_refinement_solve()`, and `run_rtk_refinement_quality()` in
`rtk_splat.backends.rtk_refinement`. It writes under
`refinement_artifacts/`, never inside a frontend/backend/segment. COLMAP's
stock prior mapper uses an internal Sim(3) initialization before restoring the
database rig scale, so this is not claimed as strict SE(3)-only BA. Exact
camera, rig, baseline, trajectory-scale, registration, reprojection, and
held-out gates enforce the usable metric contract.

Position residuals are covariance-whitened in both supported loss modes.
`cauchy` is the production-compatible default; `trivial` disables robust
downweighting while retaining those covariances and is intended for controlled
diagnostics. Sealed plans created before the loss field existed are normalized
in memory to `cauchy`; they are never rewritten. Every non-default arm requires
a distinct artifact name.

The sidecar also has two explicit initialization modes. `continuation` is the
historical default and supplies the completed visual model to COLMAP.
`fresh` omits only that `--input_path` pair, so calibration-block RTK priors
participate while a new incremental track graph is constructed. Fresh and
continuation artifacts are always separately named. Continuation retains its
relative track gate; fresh reconstruction uses absolute mean-track-length and
observations-per-image gates because the two track graphs are not identical.

This holdout establishes whether the added RTK refinement factors generalize
across time. It is not an end-to-end GNSS-independent benchmark: the sealed
frontend may already use the full RTK track for keyframes, headings, revisits,
and pair planning. Reports and pose provenance state that scope explicitly.

Resume markers are stage-specific and content-verified. A marker is not trusted
when its command, input hashes, calibration, or output inventory differs.

The Global backend exposes the bounded profile that produced the historical
18.1-minute control: three BA iterations, a 60,000-track ceiling, a target of
1,000 retained tracks per view, final retriangulation disabled, and CPU global
positioning/BA. These are canonical `mapper:` options; backend plans from the
obsolete `global_mapper:` schema are rejected rather than silently resumed.
The generated COLMAP command uses the exact Global Mapper option names.

Every real Global solve runs in its own process group with configurable nice,
memory, and disk floors. Per-attempt logs and periodic CSV/JSON resource
records are retained. Persistent low memory, low disk space, or interruption
terminates the complete process group and publishes no pose. Incremental
Mapper remains an optional fallback and keeps its previous execution path.

Read-only replay of the accepted headland model under the current temporal
holdout semantics measured 0.09891653 m median, 0.12088889 m inlier p95, and
1.0 inlier fraction. The reproduction profile gates these at 0.12 m, 0.15 m,
and 0.95 respectively; the replay validates the gate choice, not a fresh
mapper run.

## Exactly how RTK is used

| Stage | RTK role |
|---|---|
| Frontend planning | Translation, heading when available, revisits, and pair planning |
| COLMAP database | Filtered position priors retained as input evidence |
| Global/incremental visual solve | Visual feature geometry and rig constraints |
| Optional RTK pose solve | Covariance-whitened camera-position factors from calibration blocks only; either refine a supplied model or constrain fresh incremental mapping; factor-holdout rows are physically absent |
| Metric export | Fixed-scale SE(3) fit on calibration time blocks; absolute and covariance-normalized RTK diagnostics on blocks excluded from that fit |
| Diagnostic | A calibration-block Sim(3) reports scale drift but is never applied |

The integrated Global Mapper does **not** optimize covariance-weighted RTK
factors in its bundle adjustment. Preserving database priors is useful
provenance and initialization evidence, but it must not be described as an
RTK-constrained Global BA method. The optional sidecar is a separate
position-only method, in continuation or fresh incremental-mapping mode;
dual-antenna inputs improve upstream camera-centre construction but are not
consumed as an explicit heading factor by COLMAP.

Fixed scale is essential because the calibrated stereo baseline and depth are
metric. Applying a free Sim(3) scale to the trajectory while leaving stereo
depth unchanged would make the two geometry sources inconsistent.

The normalized export diagnostic evaluates each held-out 3-D camera-centre
residual with its stored 3x3 covariance and chi-square statistics. Independent
metre and support caps always remain in force, so declaring a large covariance
cannot make an arbitrarily inaccurate map pass. Chi-square checks are
authoritative only when upstream provenance establishes a calibrated effective
camera-centre covariance; static/approximated or incomplete covariance keeps
them diagnostic and records the missing uncertainty terms.

## Current command surface

The installed interface is one explicit command per stage:

```text
validate
ingest
depth
frontend-build
frontend-features
frontend-rig
frontend-priors
frontend-match
backend-prepare
backend-solve
backend-register
backend-quality
backend-refine-rtk
backend-export
cloud
train
```

Every command requires `--config`. Artifact names, expected frame counts,
frontend profile, keyframe preset, backend, and run name can be overridden
explicitly. There is no `all` command and no normal training path silently
launches COLMAP, calibration, or an experimental sidecar.

`scripts/runs/citrusfarm_05_13d_uturn.sh` is a reproduction launcher, not a
second workflow API. It invokes the explicit commands above in order, requires
a new internal-disk work directory unless verified resume is requested, and
does not start `tmux`.

`scripts/runs/refine_existing_backend.sh` runs the optional refinement and then
attempts refined export against an already completed named backend. It never
replays bags, depth, features, matching, the source mapper, cloud, or GS. It is
therefore not the launcher for a diagnostic arm that may legitimately fail.

`scripts/experiments/citrusfarm_rtk_loss_b.sh` is the fail-closed cached loss
A/B. It reuses the completed Cauchy control, runs only one new quadratic
candidate, verifies that their inputs, factor holdouts, and acceptance gates
match and their recorded COLMAP commands differ by exactly the robust-loss
flag, then writes a paired report. A single-run lock protects the candidate
workspace. The launcher pins the current executable path and hash, while the
report states that A's historical runtime-binary hash was not sealed. It never
calls pose export, cloud construction, training, or PLY generation.

`scripts/experiments/citrusfarm_pose_prior_fresh_l2.sh` is the sealed
initialization A/B. It reuses the completed quadratic continuation arm and the
same cached source, factor split, rig, features, and matches. Its fresh command
must equal the control after removing exactly `--input_path <input-model>`.
`preflight` performs no solve; `run` performs only fresh mapping, its quality
audit, and the paired report. It never exports or starts GS.

## Required real-data evaluation

The first cross-dataset Citrus run completed through visual reconstruction and
the optional RTK continuation experiments, then correctly stopped. The Cauchy
arm improved held-out median from 0.266 m to 0.249 m and support from 59.2% to
67.2%. The later quadratic arm reached 0.228 m and 95.3% support, but still
missed the 0.15 m median and 0.30 m inlier-p95 gates. A from-start quadratic
pose-prior mapping arm is prepared but has not run. Only a complete stereo registration with
acceptable held-out fixed-scale RTK behavior may proceed to cloud construction
and the 65,000-iteration GS evaluation. The headland score is not a
cross-dataset PSNR threshold.

The remaining headland efficiency A/B begins from one validated contract-v2
segment:

1. compare `gpu` with `cpu_reference` features using the `all` preset;
2. run pose-only `all`, `dense`, `balanced`, and `sparse` arms with identical
   feature evidence and mapper settings;
3. require 100% left/right registration and no georeferencing regression;
4. use cheap pose and rendering proxies to choose candidates; and
5. run GS only for the baseline and selected candidate.

The acceptance target is a material runtime reduction with no more than
0.2--0.3 dB masked-PSNR loss. Until that experiment is completed, Phase 4 is
an implemented hypothesis, not a measured improvement.
