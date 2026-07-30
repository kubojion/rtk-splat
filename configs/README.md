# Configurations

- `reproductions/` records the project's field, headland, and hangar
  experiments. These contain local paths and are not generic defaults.
- `benchmarks/` contains external dataset or method comparisons.
- `environments/` records non-Python tool environments.

The CLI requires `--config`; it never selects one of these machine-specific
files implicitly.

Only implemented options belong in active configs. `depth.backend: sgbm` is
currently the sole built-in depth choice. External adapters may provide depth
NPZ artifacts directly.

The `calibration` block in the headland stereo-BA configuration is an explicit
diagnostic sidecar. Normal `all`, cloud, and training stages ignore it.
