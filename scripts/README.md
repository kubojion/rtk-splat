# Scripts

The stable public interface is the installed `rtk-splat` CLI and Python API.
Files here are reproducibility or development orchestration; they are not the
generic mapping contract.

- `experiments/` contains controlled A/B launchers and negative probes.
- `runs/` contains dataset- and machine-specific reproduction launchers.
- `tools/` contains small artifact-inspection helpers.

`experiments/headland_two_tile_overnight.sh` is the current cached Option A
validation launcher. It never reads the 77-minute bag: it trains one same-code
monolithic control and two sequential tiles, then publishes a controlled
whole-scene and 1 m seam-band verdict. The 2026-08-10 run completed in
11 h 29 min 46 s and passed all 13 scene checks; the merged scene improved by
0.305/0.380 dB masked/corrected masked PSNR over the same-code control. Use
`preflight` before a fresh `run`, or `status` to inspect the completed workdir.
Exact within-training resume is not yet available.

This launcher is a two-tile reproduction, not the full-field server launcher.
It contains cached artifact identities and requires a monolithic control, so it
must not be pointed at the 77-minute recording.

`runs/field1_0703_full_local_prepare.sh` is the guarded local bag-to-portable-
segment preparation for the full recording. It runs no COLMAP or GS stage.
`runs/field1_0703_full_server.sh` starts from that verified portable segment,
solves the complete pose/automatic TilePlan, then trains an arbitrary number
of tiles with immutable per-tile retry attempts and production scene
publication. The completed server run used a separately labelled diagnostic
continuation after geodetic rejection, automatically planned 32 tiles, trained
all of them, and published the full layered diagnostic scene. These remain
recording/site launchers; the generic implementation lives in the package CLI.
Do not rerun them against the existing immutable workdir. See
[`SERVER_RUN.md`](../SERVER_RUN.md) and the
[milestone record](../docs/milestones/FULL_FIELD_DIAGNOSTIC_V1.md).

`rtk-splat-scene-export` is the generic portable-viewer path for a verified
layered scene. It streams the already opacity-pruned, uniquely core-owned tile
PLY bodies into one atomically published, hash-bound PLY bundle. It preserves a
diagnostic source label and does not claim to reproduce per-view layered
blending.

`tools/bootstrap_server_env.sh` creates and verifies the pinned Python/gsplat
and COLMAP Conda prefixes entirely inside a private `/data/.../rtk-splat`
project root. It never installs system software or changes base Conda. The
separate `tools/server_preflight.py` then validates the real GPU/CUDA/COLMAP
paths and portable segment before a production launcher may write artifacts.

Personal absolute paths, delayed-start wrappers, power/sleep scheduling, and
one-off experiment names must not become the public workflow. Historical
launchers may retain exact paths as reproduction evidence; reusable Python
package code may not.

New generic behavior belongs in `src/rtk_splat/` with tests. A script may call
that behavior, but must not become the only implementation of a mapping stage.
