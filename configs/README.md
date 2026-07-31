# Configurations

Configuration separates stable robot facts from per-recording facts:

```text
configs/
  robots/<name>.yaml
  sequences/<name>.yaml
  reproductions/
  benchmarks/
  environments/
```

- `robots/` stores facts measured or selected once for a platform: adapter,
  topics, camera rig, image geometry, antenna frames, camera-to-antenna
  transform and uncertainty, baseline, and pose source.
- `sequences/` stores per-run inputs only: `robot: <name>`, bag or dataset
  paths, output work directory, optional time window and sampling settings,
  split policy, and run name.
- `reproductions/` freezes machine-specific settings for accepted project
  experiments. It is not a source of generic defaults.
- `benchmarks/` contains external-dataset or method-comparison configurations.
- `environments/` records non-Python tool environments.

`configs/robots/zed_dual_rtk.example.yaml` and
`configs/sequences/headland.example.yaml` show the intended split. Copy and
measure a new robot profile; do not hide a changed camera mount or antenna
offset in a sequence file.

## Robot and sequence merge

`load_config()` resolves a sequence's `robot` value to
`robots/<name>.yaml`, recursively loads that profile, and deep-merges the
sequence over it. Nested mappings are merged; a sequence value overrides the
profile value at the same key, and lists are replaced rather than appended.
Neither input mapping is mutated.

For files under `configs/sequences/`, the enclosing `configs/` directory is
used automatically:

```python
from rtk_splat.core.configio import load_config

cfg = load_config("configs/sequences/headland.example.yaml")
```

For a config elsewhere, pass `config_root=...`; otherwise the loader searches
near the config for a `robots/` directory. Robot references are profile names,
not arbitrary paths, and missing profiles or profile cycles fail with a clear
error. A monolithic config without `robot` remains a single mapping and is not
merged.

Every resolved config requires `paths.workdir`. Generic paths such as
`paths.bags` and `paths.segment` are normalized to expanded paths. An adapter
owns and normalizes any dataset-specific paths it declares, such as a ROS
message-definition directory.

## Contract-v2 adapter use

The generic ingestion boundary selects an adapter from the resolved robot
profile and publishes a new, immutable contract-v2 segment:

```python
from rtk_splat.adapters.registry import publish_from_config
from rtk_splat.core.configio import load_config

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
