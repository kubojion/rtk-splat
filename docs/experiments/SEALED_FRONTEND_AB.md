# Sealed Frontend Experiments

These launchers prepare the two controlled headland experiments required before
changing the production defaults:

- `scripts/experiments/feature_profile_ab.sh`: GPU SIFT versus the accepted
  CPU-reference SIFT settings, with all 1,344 frames in both solves.
- `scripts/experiments/adaptive_keyframes_ab.sh`: dense, balanced, and sparse
  solve subsets using GPU features.

They run in the foreground without tmux, stop on the first failure, log every
stage, and never move from pose-only evaluation to GS automatically.

## Input and output safety

Use the normalized immutable contract-v2 headland segment: 1,344 stereo pairs,
1,176 train frames, 168 fixed validation frames, and no test frames. Pass it
explicitly with `--segment` (or set `RTK_SPLAT_SEGMENT`); the scripts validate
those numbers before writing. Every workflow command receives that exact
segment, while `--workdir` changes only the new arm-specific outputs.

The first `pose` invocation requires an experiment root that does not exist.
Later phases, or a deliberate interrupted-stage resume, require
`--resume-existing`. The root contains an `experiment.json` binding it to the
config hash, segment path, experiment kind, and arm list. A mismatched root is
rejected. Outputs are also rejected inside the source segment or git checkout.
They are rejected inside the configuration's original workdir as well, so a
new experiment cannot modify the accepted historical run.
The flag permits inspection of that root; individual stages still reject
partial output they cannot verify safely. Nothing is deleted automatically.

Example:

```bash
SCRIPT=scripts/experiments/feature_profile_ab.sh
CONFIG=configs/reproductions/headland_stereo_ba.yaml
SEGMENT=/home/jion_kubo/agromap4d_work/field_turn_contract_v2_normalized/segment
ROOT=/new/disk/path/feature-profile-$(date -u +%Y%m%dT%H%M%SZ)

bash "$SCRIPT" plan
bash "$SCRIPT" pose --config "$CONFIG" --segment "$SEGMENT" \
  --experiment-root "$ROOT" \
  --python /path/to/python --colmap /path/to/colmap
```

The pose command performs, independently for every arm:

```text
frontend-build -> frontend-features -> frontend-rig -> frontend-priors
-> frontend-match -> backend-prepare -> backend-solve
-> backend-register -> backend-quality -> backend-export
```

Global Mapper is primary. Its database is a verified private snapshot of the
sealed frontend database. For adaptive arms, only selected stereo keyframes
enter the solve; `image_registrator` must then restore all non-keyframes. Pose
export requires exact left-and-right registration, so every arm finishes with
1,344 timestamped left-camera poses. Keyframe selection never changes the
train/validation manifest. The launcher also verifies the selected preset
thresholds, one mandatory stereo pair per frame, all-image feature coverage,
and the exact GPU/CPU feature flags. A config that overrides the controlled
preset values is rejected instead of silently changing the experiment.

## Manual gates

After pose-only completion, inspect the backend and pose reports for every arm.
Do not acknowledge the next phase unless all of these hold:

- 100% registration of selected frames, all 1,344 frames, and all fixed
  evaluation frames;
- identical frame/evaluation timestamps across arms;
- no mean-reprojection-error regression;
- no held-out RTK residual or fixed-scale georeferencing regression;
- diagnostic similarity scale is reported but never applied; and
- the GPU/CPU experiment includes pose delta against the accepted solve.

Then explicitly run all 15k proxies:

```bash
bash "$SCRIPT" proxy --config "$CONFIG" --segment "$SEGMENT" \
  --experiment-root "$ROOT" \
  --resume-existing --accept-pose-gates \
  --python /path/to/python --colmap /path/to/colmap
```

Each arm gets a pose-matched cloud and a distinct non-overwriting run name. The
launcher verifies from `run_provenance.json` that every arm used exactly 15,000
training-image presentations, the same available training views and immutable
input hashes, and 1,344 identical pose timestamps. All 1,344 frames retain
poses; GS supervision continues to use the unchanged leakage-safe
1,176/168 train/validation split rather than the SfM keyframe subset.

Review the proxy metrics before choosing a winner:

- colour-corrected masked PSNR loss no greater than 0.2–0.3 dB versus the
  accepted reference;
- no material SSIM or LPIPS regression;
- unchanged evaluation timestamps and input checksums;
- no georeferencing regression; and
- at least 3x measured frontend runtime/storage reduction for a speed claim.

If an arm improves quality but misses the 3x efficiency gate, report it as a
quality result, not as a speed result. Do not infer speed from keyframe count.

Run exactly one full winner:

```bash
bash "$SCRIPT" full --config "$CONFIG" --segment "$SEGMENT" \
  --experiment-root "$ROOT" \
  --resume-existing --winner gpu --accept-proxy-gates \
  --python /path/to/python --colmap /path/to/colmap
```

For the adaptive launcher, the winner is `dense`, `balanced`, or `sparse`.
The full phase first re-verifies all completed proxies and then runs only the
named winner for exactly 65,000 presentations.

## Arms and measured timing facts

The feature-profile experiment changes only the feature extraction profile:

| Arm | Solve frames | Feature settings |
| --- | ---: | --- |
| `gpu` | 1,344 | GPU SIFT; affine shape and DSP omitted |
| `cpu` | 1,344 | CPU reference; affine shape and DSP enabled |

The adaptive experiment uses GPU features for every image. The measured
candidate solve counts on this headland sequence were:

| Arm | Translation | Rotation | Max elapsed | Reference count |
| --- | ---: | ---: | ---: | ---: |
| `dense` | 0.08 m | 1 degree | 1.5 s | 459 |
| `balanced` | 0.10 m | 2 degrees | 1.5 s | 347 |
| `sparse` | 0.15 m | 3 degrees | 1.5 s | 270 |

Turns, revisits, GNSS state changes, and quality gates are first-class inputs,
so the sealed artifact's actual count is authoritative.

Only measured task facts are printed by the launchers:

- CPU-reference feature extraction: 62.7 min
- GPU feature extraction: about 6 min
- all-frame sequential matching: 18.7 min
- all-frame Global Mapper: 18.1 min
- incremental mapper: 581.0 min, retained only as a fallback/reference
- one 15k proxy: about 1.2 h
- one 65k run: about 245 min

Adaptive matching, non-keyframe registration, and mapper timings are unknown
until this A/B runs. Logs record actual elapsed seconds per stage.

## Retired v1 launchers

The former `scripts/reproduce/headland_stereo_ba.sh`,
`headland_global_mapper.sh`, and `headland_global_gs_overnight.sh` were removed.
They were experiment-specific v1 launchers, called the deleted
`rtk_splat.cli` compatibility module, and could write into historical
workdirs. Their accepted
scientific context remains in `docs/experiments/HEADLAND_STEREO_BA.md`, the
golden manifest, and the measured facts above; they are not valid contract-v2
commands.
