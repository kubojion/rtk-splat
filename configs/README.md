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
  transform and uncertainty, baseline, and pose source. It cannot own output
  paths or training policy.
- `sequences/` stores recording inputs and a documented, bounded set of run
  overrides: `profile: <name>`, `robot: <name>`, bag or dataset paths, output
  work directory, time window, sampling mode, artifact names, scene depth
  bounds, topology/keyframe settings, and independent acceptance gates. It
  cannot redefine topics, sensor geometry, feature extraction, or loss terms.
- Runtime derivation owns values which are neither method constants nor robot
  or recording facts: frame stride/spacing, stereo maximum range, iteration
  count, and Gaussian capacity. An authored value is either `auto` or an
  explicit override. Each stage logs its formula, measurements, bounds, source
  origin and chosen value.
- `reproductions/` freezes machine-specific settings for accepted project
  experiments. It is not a source of generic defaults.
- `benchmarks/` contains external-dataset or method-comparison configurations.
- `environments/` records non-Python tool environments.

Unknown keys and keys placed in the wrong layer fail before a stage starts.
`reproductions/` is the deliberate exception: a frozen historical run is a
complete monolithic record and may own every supported key.

`configs/robots/zed_dual_rtk.example.yaml` and
`configs/sequences/headland.example.yaml` show the intended split. Copy and
measure a new robot profile; do not hide a changed camera mount or antenna
offset in a sequence file.

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
