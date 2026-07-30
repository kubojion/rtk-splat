# RTK-Splat Progress

Last updated: 2026-07-30

This is the current source of truth for this repository. Superseded plans are
kept under `docs/archive/`.

## Current reference result

The reference experiment is the 450-second headland sequence containing the end
of row 5, a U-turn, and the entry to row 1.

- 1,344 calibrated stereo pairs at approximately 3 Hz.
- 0.119846 m measured rectified stereo baseline.
- 65,000 GS iterations and a 2.5 million Gaussian cap.
- The raw-RTK and stereo-BA arms used the same SGBM depth, split, and GS
  settings. Pose backend and pose-derived initialization cloud were the
  material changes.

| Metric | `tile_turn2`, RTK | `tile_turn3_stereo_ba` | Change |
|---|---:|---:|---:|
| Masked PSNR | 20.6097 | **24.4449** | **+3.8352 dB** |
| Corrected masked PSNR | 21.1519 | **25.8480** | **+4.6960 dB** |
| SSIM | 0.3209 | **0.5944** | +0.2736 |
| Corrected LPIPS | 0.5333 | **0.3187** | −0.2146 |
| Test-pose-aligned masked PSNR | 20.9356 | **24.4383** | — |

Test-time pose alignment provides no gain after stereo BA, supporting the
conclusion that the refined trajectory is already locally consistent for this
reconstruction.

## Pose reconstruction evidence

- Registered left frames: **1,344 / 1,344**.
- Registered total images: **2,688 / 2,688**.
- Visual-to-RTK alignment residual:
  - median: **5.11 cm**
  - p95: **10.72 cm**
  - maximum: **13.80 cm**
- Exporter Sim(3) scale: **0.985794**.
- Mean COLMAP reprojection error: **0.818 px**.
- COLMAP mapper runtime: approximately **581 minutes**.
- Complete feature, matching, mapping, and GS experiment: **15 h 16 min**.

The 5.11 cm value is a disagreement between the aligned visual camera trajectory
and the RTK-derived camera trajectory. It is not an independent measurement of
RTK accuracy and cannot alone distinguish RTK noise, rough TF, clock offset, or
visual deformation.

The run began while the stereo implementation was uncommitted. Git commit
`3f86826c151ffffab8932e6211bbe933108a9983` is the first matching post-run
source snapshot, not a falsely claimed clean run-time checkout. Exact hashes
and artifact statistics are in
[headland_stereo_ba.json](docs/experiments/golden/headland_stereo_ba.json).

## Metric-integrity audit

The strict audit is non-destructive and is not consumed by training.

Trusted corrections:

- Optical-camera lever Z: **−0.03862885 m**.
- Antenna-baseline tangent mode 0: **+0.01502952 rad (+0.8611°)**.

Retained at their rough priors because the data did not support correction:

- Lever X and Y.
- Second baseline tangent mode.
- Clock offset.
- Rotation about the physical antenna baseline, which one baseline cannot
  observe structurally.

Held-out aggregate:

| Metric | Prior | Retained correction |
|---|---:|---:|
| Antenna median | 5.65 cm | **3.71 cm** |
| Antenna RMS | 5.70 cm | **4.58 cm** |
| Antenna p95 | **7.63 cm** | 7.84 cm |
| Baseline-angle median | 0.877° | **0.243°** |

The p95 position residual worsened by 2.8%, within the configured audit gate.
That tradeoff remains visible. These are calibration recommendations, not
enabled training parameters.

## Phase 1 repository cleanup

Completed in the current working tree:

- Removed tracked bytecode and generated caches; added ignore rules.
- Added package metadata and one discoverable test command.
- Grouped reproduction, benchmark, and environment configurations.
- Replaced six root-level run scripts with one guarded, parameterized
  reproduction script; five obsolete scripts were removed.
- Made the long-run check path genuinely read-only.
- Fingerprint new COLMAP resume markers so changed commands or calibration do
  not silently reuse stale completed steps.
- Added an exact golden-artifact verifier and nondeterministic acceptance gates.
- Made `--config` explicit; there is no local-machine default.
- Fixed the external TUM trajectory source's incomplete track structure.
- Stopped silently accepting unimplemented depth backend names.
- Store both rectified camera calibrations in new canonical segments so future
  COLMAP pose backends need not reopen a ROS bag.
- Corrected ENU CRS propagation into new segment metadata.
- Preserved the metric-integrity implementation and tests without enabling it
  in normal mapping.

No COLMAP, GS, input bag, or existing work artifact was modified or rerun.

## Genericity status

The intended stable core boundary is now documented and partially enforced:

```text
dataset adapter
    → canonical stereo segment
    → named pose backend
    → pose-matched cloud
    → GS mapper/evaluator
```

Verified adapters:

- Project ROS 2 ZED/u-blox recordings.
- AgriGS external-folder format.

Not yet verified:

- CitrusFarm.
- Arbitrary ROS topic layouts.
- Stereo cameras split across separate bag files.
- A bag-free COLMAP preparation using a newly generated portable segment.

The last item is supported in code for new metadata but still needs an
end-to-end adapter test before being claimed.

## Known scientific limitations

- Validation views participated in SfM pose estimation.
- The headland sequence lacks independent camera-pose ground truth.
- One successful scene is not enough to claim generalization or SOTA.
- Current right imagery contributes to stereo depth and BA, but right-camera GS
  supervision is not validated.
- Full-field chunking, submap consistency, and tile merging are not implemented.
- Metric stereo should ultimately use a fixed-scale SE(3) georegistration model;
  the reference exporter still records its legacy bounded Sim(3) alignment.

## Next implementation phase

1. Add a non-destructive COLMAP Global Mapper backend.
2. Reuse the existing headland feature/match database and write a new pose
   artifact.
3. Compare registration, reprojection, tracks, stereo rigidity, fixed-scale RTK
   alignment, runtime, and memory before running GS.
4. Run a shortened GS screen only if pose gates pass.
5. Repeat the winning backend on the hangar-to-field sequence.
6. Implement and contract-test one external dataset adapter, with CitrusFarm as
   the leading candidate.

The first speed target is to preserve quality within 0.3 dB while materially
reducing pose-estimation time. A further 3.8 dB pose-only gain is not assumed.
