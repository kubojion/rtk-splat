# Configurations

Configuration has three authored layers and one measured runtime layer:

```text
configs/
  profiles/<name>.yaml
  robots/<name>.yaml
  sequences/<name>.yaml
  reproductions/
  benchmarks/
  environments/
```

- `profiles/` stores versioned method/quality policy: features, matching,
  mapper and training policy, plus bounded runtime-derivation rules. It cannot
  own bags, topics or sensor geometry.
- `robots/` stores facts measured or selected once for a platform: adapter,
  topics, camera rig, image geometry, antenna frames, camera-to-antenna
  transform and uncertainty, baseline, pose source, and stable adapter clock/
  chunk-contract policy. It cannot own output paths or training policy.
- `sequences/` stores recording inputs and a documented, bounded set of run
  overrides: `profile: <name>`, `robot: <name>`, recording paths, output
  work directory, time window, sampling mode, artifact names, scene depth
  bounds, topology/keyframe settings, bag/clock validation gates, and
  independent acceptance gates. It cannot redefine topics, sensor geometry,
  feature extraction, or loss terms.
- Runtime derivation owns values which are neither method constants nor robot
  or recording facts: frame stride/spacing, stereo maximum range, initial
  cloud capacity, iteration count, and Gaussian capacity. The cloud limit is
  derived below the final VRAM-bound Gaussian cap so MCMC retains growth
  headroom. An authored value is either `auto` or an
  explicit override. Each stage logs its formula, measurements, bounds, source
  origin and chosen value.
- `reproductions/` freezes machine-specific settings for accepted project
  experiments. It is not a source of generic defaults.
- `benchmarks/` contains external-dataset or method-comparison configurations.
- `environments/` records non-Python tool environments.

Unknown keys and keys placed in the wrong layer fail before a stage starts.
`reproductions/` is the deliberate exception: a frozen historical run is a
complete monolithic record and may own every supported key.

## Planned release-facing simplification

The current schema is explicit and auditable, but the public new-dataset path
is still too verbose. Its simplification is tracked in `../TODO.md` and is not
implemented yet. The intended user-authored surface is:

- robot facts that must actually be measured: intrinsics, stereo transform,
  camera-to-antenna prior and uncertainty, timestamps/topics, GNSS status and
  covariance semantics;
- sequence facts: input location, optional time/window selection, output root,
  and quality profile; and
- rare explicit overrides, each labelled as an override.

Sampling density, usable stereo range, iteration count, Gaussian capacity, and
tile training-view workload belong to sealed runtime derivation. A new recording must
not require guessing them, while a scientific reproduction must retain the
fully resolved values and formula inputs. Until that interface is implemented
and tested, the existing strict schema remains authoritative.

`quality_v1.yaml` now also owns the generic `tiles:` planning policy. Its
automatic frame capacity is derived from the training iteration/presentation
budget, while support-cell size, bounded depthless-frame tolerance, minimum
core training support, and visibility halo policy are sealed in each TilePlan.
A sequence may name a TilePlan but cannot redefine the planning method. Do not
put row numbers, per-tile time windows, or dataset-specific boundaries in a
profile. See [the tiled-scene method](../docs/methods/TILED_SCENE.md).

`profiles/headland_quality_accepted_v1.yaml` is the deliberately frozen first
full-field transfer profile. It inherits `quality_v1` but keeps the measured
4M initialization cloud, 65k presentations, 2.5M Gaussian cap, and 0.20 m
scale cap that reproduced the accepted headland. It is not a claim that these
limits are optimal for a 3090. A higher-capacity policy needs a matched
one-tile A/B and a new versioned profile rather than an in-place edit.

The same profile keeps heading freshness at 150 ms but permits a sparse,
auditable receiver dropout: at most 0.5% of selected frames, no more than two
consecutively, and never farther than 0.5 s from the nearest raw heading. Such
rows are stored with `association_valid=false` and cannot become heading
evidence. The generic `quality_v1` default remains fail-closed at zero invalid
heading associations.

`sequences/field1_0703_full_77min.yaml` names artifacts and the complete local
recording only. The local/server procedure is documented in
[`SERVER_RUN.md`](../SERVER_RUN.md); server commands override the segment and
work roots while retaining the same authored configuration identity.

`configs/robots/zed_dual_rtk.example.yaml` and
`configs/sequences/headland.example.yaml` show the intended split. Copy and
measure a new robot profile; do not hide a changed camera mount or antenna
offset in a sequence file.

`robots/rosario_v2_dual_m2_d435.yaml` and
`sequences/rosario_v2_sequence5_ppk_140_250.yaml` are the cross-dataset ROS 1
example. The robot file owns topics, calibration priors, and predeclared
real-sensor acceptance gates. The sequence owns only its input bags, offline
PPK choice, bounded window, output names, and independent georeferencing caps.
Its online-GNSS ablation is a separate sequence and is expected to fail rather
than silently replace the configured PPK evidence.

## Profile, robot and sequence merge

`load_config()` resolves `profile` and `robot` by name, then deep-merges in the
fixed order profile -> robot -> sequence. Nested mappings are merged and lists
are replaced rather than appended. Ownership validation happens before the
merge, so precedence cannot hide a sensor fact in a sequence. No input mapping
is mutated.

For files under `configs/sequences/`, the enclosing `configs/` directory is
used automatically:

```python
from rtk_splat.workflows.configio import load_config

cfg = load_config("configs/sequences/headland.example.yaml")
```

For a config elsewhere, pass `config_root=...`; otherwise the loader searches
near the config for a `robots/` directory. References are names, not arbitrary
paths, and missing files or inheritance cycles fail clearly. Files under
`reproductions/` and `benchmarks/` remain self-contained monolithic mappings.

Every resolved config requires `paths.workdir`. Generic paths such as
`paths.bags` and `paths.segment` are normalized to expanded paths. An adapter
owns and normalizes any dataset-specific paths it declares, such as a ROS
message-definition directory.

## Contract-v2 adapter use

The generic ingestion boundary selects an adapter from the resolved robot
profile and publishes a new, immutable contract-v2 segment:

```python
from rtk_splat.adapters.registry import publish_from_config
from rtk_splat.workflows.configio import load_config

cfg = load_config("configs/sequences/headland.example.yaml")
segment = publish_from_config(cfg, "/new/output/segment")
```

The destination must not already exist. Adapter output records exact integer
nanosecond timestamps, coordinate and timebase semantics, camera/antenna frame
semantics, calibration, per-frame position covariance and status, optional
dual-antenna or IMU evidence, and explicit image/depth capabilities. Dataset
differences belong in adapters and profiles, not in the mapping core.

The installed workflow entry point is:

```bash
rtk-splat <stage> --config configs/sequences/<name>.yaml
```

For a source checkout, the equivalent module is
`python -m rtk_splat.workflows.cli`. No machine-specific configuration is selected
implicitly.

Only implemented options belong in active configs. SGBM is the current built-in
computed-depth backend; an adapter may instead publish recorded metric depth in
contract-v2 NPZ files. Optional diagnostic calibration blocks are consumed only
by their explicit diagnostic stage and are not enabled during ordinary
training.

## Resolved runtime record

Successful CLI stages accumulate a fail-closed ledger at
`<workdir>/config_artifacts/resolved_config.json`. It stores the complete
merged authored configuration, every source file and SHA-256, explicit CLI
overrides, and stage derivations. A later invocation may hydrate old
derivations as provenance, but never applies their chosen values; a stage that
needs one measures and derives it again. Reusing a work directory after a
configuration source changes is rejected.

Ingested segments, derived-depth segments, sealed frontends, and GS runs copy
the complete effective configuration and derivation evidence into their own
immutable provenance. Artifacts therefore remain auditable without depending
on mutable ledger state.
