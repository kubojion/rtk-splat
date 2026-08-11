# Full-field RTX 3090 run

Last updated: 2026-08-10

This is the operational checklist for the 77-minute field build. It is not a
generic API specification. The stable method remains the `rtk-splat` CLI; the
two launchers below are guarded reproductions for this recording and server.

## Current state

- [x] Two-tile controlled headland validation passed whole-scene and seam
  gates.
- [x] Full ROS2 ingest is memory-bounded; selected JPEG payloads are spooled
  and moved into the immutable segment instead of being retained in RAM.
- [x] The finalized 120-second real-bag probe published 316 frames in
  29 min 18 s at 453 MB peak RSS on USB 2. Its one 190 ms heading association
  remained invalid and passed the bounded dropout policy.
- [x] Sparse dual-heading dropouts remain explicitly invalid at the original
  150 ms freshness limit. The full-field profile permits at most 0.5% invalid
  rows, no run longer than two selected frames, and no stale residual above
  0.5 s; larger or clustered outages abort ingest.
- [x] A self-contained segment materializer and terminal SHA-256 verifier are
  implemented.
- [x] Production scene publication accepts an arbitrary number of verified
  tiles and does not require a full-field monolithic GS reference.
- [x] The first full-field profile freezes the accepted 65k / 2.5M-Gaussian /
  4M-cloud settings. Higher capacity is a later one-tile A/B, not a silent
  change to the first transfer test.
- [x] The server workflow restarts only an interrupted tile under a new
  immutable attempt name and preserves completed tiles.
- [ ] Reconnect Buffalo at USB 3 speed and run local ingest/depth.
- [ ] Create and push (or bundle) one clean tested Git snapshot.
- [ ] Transfer and verify the portable segment on the server.
- [ ] Inventory the server and run the full pose/TilePlan preparation stage.
- [ ] Complete one RTX 3090 tile smoke test, inspect it, then run the remaining
  tiles and publish the production scene.

Exact optimizer-level resume is deliberately not claimed. The CUDA gsplat
MCMC path is not bitwise deterministic across process restarts and the current
checkpoint omits optimizer/scheduler state. Once tile training has begun, a
failure loses only the active tile, expected to be about 1.5--3.5 hours on the
3090, rather than every completed tile. Upstream interruption behavior is
listed separately below.

## Directory layout

Keep source code, immutable data, and generated work separate:

```text
/data/jkobo/code/rtk-splat
/data/jkobo/datasets/field1_0703_full77/segment
/data/jkobo/runs/field1_0703_full77_v1
```

The raw bag and the local no-depth source segment are not needed on the
server. Transfer only the portable SGBM segment and the exact code snapshot.
Do not copy the laptop's `config_artifacts/` into the server work directory;
the server creates its own ledger with server-local source paths.

## 1. Local ingest and SGBM depth

First connect Buffalo through USB 3. Check:

```bash
lsusb -t
```

The Mass Storage entry should show `5000M` or `10000M`, not `480M`. The
observed USB 2 read rate is 35.7 MB/s, and the MCAP must be scanned more than
once. USB speed is therefore important for ingest. SGBM itself is CPU/NVMe
bound and does not use the GPU.

From the repository:

```bash
cd /home/jion_kubo/agrorob_ws/src/AgroMap-4D/rtk_splat

bash scripts/runs/field1_0703_full_local_prepare.sh preflight
bash scripts/runs/field1_0703_full_local_prepare.sh run
```

If a completed stage already has a matching receipt:

```bash
bash scripts/runs/field1_0703_full_local_prepare.sh run --resume-existing
```

The script runs, in order:

1. bag to immutable rectified stereo/RTK segment;
2. SGBM depth into a new derived segment;
3. portable materialization with no symlinks; and
4. a full byte rehash against `transfer_manifest.json`.

It intentionally does not hard-gate the predicted frame count. The raw topic
inventory suggests about 13,800 selected pairs, but the published count is the
source of truth. Expected unique output is about 40 GB (37 GiB).

Estimated local time:

| Connection | Ingest | SGBM | Total |
|---|---:|---:|---:|
| Current USB 2 | 2 h 10--40 min | 30--55 min | about 2.7--4 h |
| USB 3 | 25--60 min | 30--55 min | about 1.2--2.5 h |

The total also allows roughly 10--25 minutes to materialize and hash the
portable directory. These are estimates; the terminal stage logs record the
actual time and peak memory.

The final transfer source is:

```text
/home/jion_kubo/agromap4d_work/field1_0703_full_77min/transfer/field1-0703-full77-portable-v1
```

## 2. Freeze and move the code

Do not clone `origin` yet unless the current tested tree has been committed:
the working tree contains the new tile, production-publication, streaming, and
server code that is not all present in the remote branch.

Preferred path:

1. review `git status` and exclude probe/personal artifacts;
2. run the complete test suite;
3. create a named branch and one tested snapshot commit;
4. push the branch; and
5. clone exactly that branch/commit on the server.

Data never belongs in Git. If server-side GitHub authentication is inconvenient,
create a `git bundle` from the committed branch, transfer it with `rsync`, and
clone the bundle on the server. A bundle is reproducible; copying a dirty source
directory is not.

After connecting to the server, the GitHub path is conceptually:

```bash
mkdir -p /data/jkobo/code
cd /data/jkobo/code
git clone --branch <tested-branch> <repository-url> rtk-splat
cd rtk-splat
git rev-parse HEAD
git status --short       # must be empty
```

## 3. Install the server environments

A repository copy alone is not runnable. Use two user-owned Conda environments
for the first build; no tested Docker image exists yet, and adding Docker now
would introduce NVIDIA-runtime, COLMAP, and storage-mount differences from the
accepted environment.

```bash
cd /data/jkobo/code/rtk-splat

conda create -n rtk-splat-server python=3.10 pip -y
conda activate rtk-splat-server
python -m pip install -r configs/environments/rtk-splat-3090-cu121.txt
python -m pip install -e . --no-deps

conda env create -f configs/environments/colmap-4.1.1-cuda.yml
```

The tested GS stack is Python 3.10, PyTorch 2.4.1+cu121, torchvision
0.19.1+cu121, and gsplat 1.5.3+pt24cu121. Cache LPIPS weights while outbound
internet is available:

```bash
conda activate rtk-splat-server
python - <<'PY'
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
print("LPIPS VGG cache ready")
PY

sha256sum ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
# expected:
# 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0
```

Run the isolated suite once on the server:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider
```

The plugin-autoload override prevents unrelated sourced ROS pytest plugins
from contaminating this project environment. The pinned server requirements
include the pure-Python `rosbags` package only so the complete synthetic
adapter test suite can run; the server mapping stages start from the portable
segment and do not read ROS bags or require a ROS installation.

The pinned wheel route follows gsplat's documented precompiled-wheel option,
and COLMAP stays in a separate environment because its CUDA support is a
build/runtime property rather than a Python package detail. References:
[gsplat installation](https://github.com/nerfstudio-project/gsplat#installation)
and [COLMAP installation](https://colmap.github.io/install.html).

## 4. Transfer and verify the segment

Keep the VPN endpoint and account outside the repository. Define a local alias
in `~/.ssh/config` (substitute the real values there):

```sshconfig
Host rtk3090
  HostName <server-address>
  User <server-user>
  IdentitiesOnly yes
  PubkeyAuthentication no
  PreferredAuthentications password
  ServerAliveInterval 30
  ServerAliveCountMax 6
```

Start the VPN separately, then create the destination through that alias and
transfer the portable directory. Do not use `-z`: JPEG and NPZ data are already
compressed.

```bash
ssh rtk3090 'mkdir -p /data/jkobo/datasets/field1_0703_full77/segment'

rsync -a --partial --append-verify --info=progress2 \
  -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6' \
  /home/jion_kubo/agromap4d_work/field1_0703_full_77min/transfer/field1-0703-full77-portable-v1/ \
  rtk3090:/data/jkobo/datasets/field1_0703_full77/segment/
```

Rerun the same command after an interruption. Approximate 40 GB transfer time
before VPN overhead is 53 min at 100 Mbps, 1 h 46 min at 50 Mbps, 4 h 25 min at
20 Mbps, or 8 h 51 min at 10 Mbps.

On the server, the launcher preflight fully rehashes the terminal transfer
manifest and validates the segment contract. Do not proceed merely because
the directory size looks plausible.

## 5. Inventory and run the server

The first preflight records the CPU model/count and checks the exact
Python/gsplat/COLMAP stack, a real gsplat CUDA rasterization, a real COLMAP GPU
SIFT extraction, RTX 3090 memory, the VGG checksum, at least 120 GiB physical
RAM by default, at least 500 GiB free, and every segment byte:

```bash
cd /data/jkobo/code/rtk-splat
conda activate rtk-splat-server
export COLMAP_BIN="$HOME/miniconda3/envs/colmap-rtk/bin/colmap"

bash scripts/runs/field1_0703_full_server.sh preflight
```

The preferred RAM target is 128 GB until the full Global Mapper is measured.
The 40 TB capacity is ample, but capacity is not I/O speed. Inspect the
preflight `findmnt` result. If `/data/jkobo` is HDD, network storage, or a slow
array, place the frontend database/backend work on local NVMe and archive the
finished artifacts to `/data/jkobo` afterward.

Run in three reviewable phases:

```bash
# Full frontend, Global pose, fixed-scale ENU export, automatic TilePlan.
bash scripts/runs/field1_0703_full_server.sh prepare

# First complete 2.5M-Gaussian tile. The launcher chooses the planned tile
# with the largest training/validation workload.
bash scripts/runs/field1_0703_full_server.sh smoke

# Reuse the smoke tile, train every remaining tile, publish the scene.
bash scripts/runs/field1_0703_full_server.sh run
```

Recovery on the server is automatic: verified sealed stages and completed
tiles are reused. An interrupted tile is preserved and retried under a new
attempt name. An interrupted feature extraction is not resumable in place,
and a Global Mapper process restart repeats that solve; if feature extraction
is interrupted, preserve the work directory and start a new versioned
workdir. The launcher never silently deletes partial evidence.

`prepare` is the main uncertainty: the full all-frame visual solve is roughly
27,600 images and has not previously been measured. It must register all
required frames and pass visual/fixed-scale RTK gates before tiling starts.
The planner chooses tile count from measured depth visibility and workload; it
does not assume six rows or a fixed interval.

The first launcher intentionally stops if fixed-scale georeferencing fails; it
does not silently turn a failed metric map into a production result. The
expensive sealed frontend/backend solve remains reusable in that case. Review
the residual report first, then create a separately named diagnostic pose/plan
continuation if a render is still wanted—do not restart COLMAP and do not loosen
the production gate in place.

For a job independent of an SSH/VS Code session, use `nohup` (no tmux). Keep
the shell log outside the initially empty workdir because the launcher seals
that workdir's identity itself:

```bash
mkdir -p /data/jkobo/run_logs
nohup bash scripts/runs/field1_0703_full_server.sh prepare \
  > /data/jkobo/run_logs/field1_full77_v1_prepare.log 2>&1 < /dev/null &
echo $!
```

Monitor without changing state:

```bash
bash scripts/runs/field1_0703_full_server.sh status
tail -f /data/jkobo/run_logs/field1_full77_v1_prepare.log
nvidia-smi
```

Expected server time is deliberately broad until the smoke completes:

- full frontend/matching/Global pose: roughly 6--24+ hours;
- automatic plan: minutes to perhaps an hour, dominated by depth rehash/support;
- about 10--15 tiles before measured halo duplication;
- each 2.5M/65k tile: provisional 1.5--3 hours on the 3090; and
- total: approximately 2--4 days if the Global solve passes.

Reserve 0.5--1 TB of fast working storage. The first profile intentionally
keeps the accepted 2.5M cap. After the smoke, a separate 4M-cap A/B may test
whether the 3090's extra VRAM improves quality enough to justify a new profile.

## 6. Codex on the server

The cleanest workflow is VS Code Remote-SSH for editing/navigation and Codex
CLI inside the remote integrated terminal. Codex must run in the remote repo
to see the server files, logs, dependencies, and GPU.

Install the official CLI on the server:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex
```

Complete the first-time sign-in on that machine and confirm outbound HTTPS is
allowed. Open `/data/jkobo/code/rtk-splat` through the `rtk3090` SSH alias, then
run Codex from that remote directory. Do not make a multi-day training process
depend on Codex or VS Code remaining connected; use the deterministic launcher
and let Codex inspect sealed status/logs afterward.

Official references: [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) and
[Codex IDE extension](https://learn.chatgpt.com/docs/codex/ide).

## Completion gates

Do not call the full map successful until all are true:

- every source stereo pair required by the plan is registered;
- fixed-scale ENU pose export passes held-out RTK/covariance gates;
- every planned tile has one verified completed attempt;
- every core contributes Gaussians and ownership is exhaustive/nonduplicated;
- every source validation view and the metric seam bands are evaluated;
- the production scene terminal manifest rehashes; and
- georeferencing status remains production/PASSED/eligible. If it does not,
  the production launcher stops. A later explicitly named diagnostic
  continuation may render the cached visual solve, but it must carry no metric
  georeferencing claim.
