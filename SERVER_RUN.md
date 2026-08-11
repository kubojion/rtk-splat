# Full-field 24 GB RTX server run

Last updated: 2026-08-11

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
- [x] Run local ingest/depth and publish the portable segment.
- [ ] Commit and push the clean tested server-preparation snapshot. The
  repository is public, but remote `main` is currently behind the local tree.
- [ ] Finish transferring and verify the portable segment on the server.
- [x] Inventory the server: RTX 4090 24 GB, driver 580.95.05, 125 GiB
  reported RAM, 24 CPU threads, and about 17 TiB free below `/data`.
- [ ] Install and verify the two isolated project-local Conda environments.
- [ ] Pass the complete server preflight and run the full pose/TilePlan
  preparation stage.
- [ ] Complete one RTX 4090 tile smoke test, inspect it, then run the remaining
  tiles and publish the production scene.

Exact optimizer-level resume is deliberately not claimed. The CUDA gsplat
MCMC path is not bitwise deterministic across process restarts and the current
checkpoint omits optimizer/scheduler state. Once tile training has begun, a
failure loses only the active tile, expected to be about 1.5--3.5 hours on the
4090, rather than every completed tile. Upstream interruption behavior is
listed separately below.

## Directory layout

Keep source code, immutable data, and generated work separate:

```text
/data/jkobo/rtk-splat/code/rtk-splat
/data/jkobo/rtk-splat/datasets/field1_0703_full77/segment
/data/jkobo/rtk-splat/envs/{rtk-splat-server,colmap-rtk}
/data/jkobo/rtk-splat/runs/field1_0703_full77_v1
/data/jkobo/rtk-splat/logs
```

The server account is `imoroz`; `jkubo` is not the SSH account. The existing
`/data/jkubo` directory belongs to someone else and must not be touched. The
allocated parent `/data/jkobo` is not directly writable, so initialize only
the exact project subtree once:

```bash
sudo mkdir -p /data/jkobo/rtk-splat
sudo chown -R --no-dereference imoroz:imoroz /data/jkobo/rtk-splat
sudo chmod 0750 /data/jkobo/rtk-splat
```

If a dataset copy was started with `sudo`, wait for it to finish before the
recursive `chown`, then never use `sudo` inside this subtree again. Do not
change ownership or permissions on `/data/jkobo`, `/data/jkubo`, `/data`, the
Conda base installation, or any other project.

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
inventory suggested about 13,800 selected pairs; the published 10,227-frame
count is the source of truth.

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

The completed 2026-08-10 build contains **10,227** stereo/depth frames over
4,621.30 s, **30,687** content-sealed files and **31,546,138,468 bytes**
(29.38 GiB). It has no symlinks. On the observed USB 2 connection, ingest took
2 h 03 min, SGBM took 44 min 50 s, and portable publication took 1 min 32 s;
all exited zero without swap. Its terminal inventory identity is:

```text
2d10e9f39dee8e89125b4644b269c06d3344611d7b341fb271e45dcea17b843c
```

## 2. Clone the tested code

The repository is public, so clone over HTTPS without adding a personal GitHub
credential to the server. Data never belongs in Git.

```bash
mkdir -p /data/jkobo/rtk-splat/code
cd /data/jkobo/rtk-splat/code
git clone https://github.com/kubojion/rtk-splat.git rtk-splat
cd rtk-splat
git rev-parse HEAD
git status --short       # must be empty
```

Record and use the exact tested commit. Do not run a production stage from a
dirty checkout; the launcher enforces this.

Before cloning, verify that the remote actually contains the local full-field
commit. “Repository is public” and “all commits were pushed” are separate
facts:

```bash
git status -sb
git rev-parse HEAD
git ls-remote --heads origin main
```

The last two hashes must agree after the server-preparation changes are
committed and pushed.

## 3. Install the server environments

A repository copy alone is not runnable. Use two user-owned Conda environments
for the first build; no tested Docker image exists yet, and adding Docker now
would introduce NVIDIA-runtime, COLMAP, and storage-mount differences from the
accepted environment.

Use the checked bootstrap. It installs prefix-based environments and package
caches below `/data/jkobo/rtk-splat`; it never uses `sudo`, changes base Conda,
installs a system CUDA toolkit/driver, writes `/usr`, or changes shell startup
files:

```bash
cd /data/jkobo/rtk-splat/code/rtk-splat

bash scripts/tools/bootstrap_server_env.sh plan
bash scripts/tools/bootstrap_server_env.sh install
bash scripts/tools/bootstrap_server_env.sh verify

conda activate /data/jkobo/rtk-splat/envs/rtk-splat-server
export COLMAP_BIN=/data/jkobo/rtk-splat/envs/colmap-rtk/bin/colmap
export RTK_SPLAT_SERVER_ROOT=/data/jkobo/rtk-splat
```

Allow roughly 20--60 minutes for the first install, mostly Conda solving,
package downloads and the LPIPS VGG download. `verify` itself should take only
a few minutes. Rerunning `install` is idempotent for a valid existing prefix;
it does not reinstall system software.

The tested GS stack is Python 3.10, PyTorch 2.4.1+cu121, torchvision
0.19.1+cu121, and gsplat 1.5.3+pt24cu121. Cache LPIPS weights while outbound
internet is available; the bootstrap performs and verifies this. The server's
580.95.05 driver can run the packaged CUDA runtimes. `nvcc` is not required:
PyTorch carries CUDA 12.1 runtime libraries, and COLMAP is isolated in its own
Conda CUDA build.

```bash
conda activate /data/jkobo/rtk-splat/envs/rtk-splat-server
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
Host rtk4090
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
ssh rtk4090 'mkdir -p /data/jkobo/rtk-splat/datasets/field1_0703_full77/segment'

rsync -a --partial --append-verify --info=progress2 \
  -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6' \
  /home/jion_kubo/agromap4d_work/field1_0703_full_77min/transfer/field1-0703-full77-portable-v1/ \
  rtk4090:/data/jkobo/rtk-splat/datasets/field1_0703_full77/segment/
```

Rerun the same command after an interruption. For the actual 31.55 GB artifact,
approximate transfer time before VPN overhead is 42 min at 100 Mbps, 1 h 24 min
at 50 Mbps, 3 h 30 min at 20 Mbps, or 7 h at 10 Mbps.

On the server, the launcher preflight fully rehashes the terminal transfer
manifest and validates the segment contract. Do not proceed merely because
the directory size looks plausible.

## 5. Inventory and run the server

The first preflight records the CPU model/count and checks the exact
Python/gsplat/COLMAP stack, a real gsplat CUDA rasterization, a real COLMAP GPU
SIFT extraction, supported RTX 3090/4090 memory, the VGG checksum, at least
120 GiB physical RAM by default, at least 500 GiB free, and every segment byte:

```bash
cd /data/jkobo/rtk-splat/code/rtk-splat
conda activate /data/jkobo/rtk-splat/envs/rtk-splat-server
export RTK_SPLAT_SERVER_ROOT=/data/jkobo/rtk-splat
export COLMAP_BIN=/data/jkobo/rtk-splat/envs/colmap-rtk/bin/colmap

bash scripts/runs/field1_0703_full_server.sh preflight
```

The preferred RAM target is 128 GB until the full Global Mapper is measured.
The measured 19 TiB data-volume capacity is ample, but capacity is not I/O
speed. Inspect the preflight `findmnt` result. The inventory shows three NVMe
devices, but the mount check remains authoritative. If it is unexpectedly slow,
measure before starting the full database rather than moving paths ad hoc.

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

`prepare` is the main uncertainty: the full all-frame visual solve is exactly
20,454 images (10,227 stereo frames) and has not previously been measured. It
must register all required frames and pass visual/fixed-scale RTK gates before
tiling starts.
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
mkdir -p /data/jkobo/rtk-splat/logs
nohup bash scripts/runs/field1_0703_full_server.sh prepare \
  > /data/jkobo/rtk-splat/logs/field1_full77_v1_prepare.log 2>&1 < /dev/null &
echo $!
```

Monitor without changing state:

```bash
bash scripts/runs/field1_0703_full_server.sh status
tail -f /data/jkobo/rtk-splat/logs/field1_full77_v1_prepare.log
nvidia-smi
```

Expected server time is deliberately broad until the smoke completes:

- full frontend/matching/Global pose: roughly 6--24+ hours;
- automatic plan: minutes to perhaps an hour, dominated by depth rehash/support;
- roughly 8--12 tiles before measured halo duplication (the final count is
  data-derived);
- each 2.5M/65k tile: provisionally no more than the old 3090 estimate of
  1.5--3 hours; measure the actual 4090 result in `smoke`; and
- total: approximately 2--4 days if the Global solve passes.

Reserve 0.5--1 TB of fast working storage. The first profile intentionally
keeps the accepted 2.5M cap. After the smoke, a separate 4M-cap A/B may test
whether the 4090's compute headroom improves quality enough to justify a new
profile.

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
allowed. Open `/data/jkobo/rtk-splat/code/rtk-splat` through the `rtk4090` SSH
alias, then run Codex from that remote directory. Do not make a multi-day
training process depend on Codex or VS Code remaining connected; use the
deterministic launcher and let Codex inspect sealed status/logs afterward.

A copy-paste safety/operation prompt for the server agent is in
[SERVER_CODEX_HANDOFF.md](docs/operations/SERVER_CODEX_HANDOFF.md).

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
