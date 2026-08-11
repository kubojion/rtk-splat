# Codex handoff for the full-field server

Use this prompt after starting Codex from the clean server checkout. It keeps
the agent inside RTK-Splat's private project tree and makes the first server
actions read-only.

```text
Repository: /data/jkobo/rtk-splat/code/rtk-splat
Private project root: /data/jkobo/rtk-splat

Read SERVER_RUN.md, TODO.md, PROGRESS.md, and
scripts/runs/field1_0703_full_server.sh completely before acting.

Goal: validate and then execute the guarded 77-minute full-field pipeline on
this server. Do not redesign the method, loosen a quality/georeferencing gate,
or change the frozen stride-5 / 65k / 2.5M first-run settings.

Safety boundaries:
- Work only inside /data/jkobo/rtk-splat and this Git checkout.
- Do not use sudo, apt, Docker, system Python, system CUDA, or modify Conda
  base, shell startup files, GPU drivers, services, or another user's files.
- Do not kill a process unless it is positively identified as this run and I
  explicitly authorize it. Check nvidia-smi for other users before GPU work.
- Never edit the transferred segment or an already published artifact.
- Preserve failed/partial attempts; use the launcher's immutable retry policy.
- Keep the Git checkout clean. Stop if HEAD, configuration, environment,
  segment hashes, or server inventory disagree with the sealed run identity.
- Do not start a long stage merely because preflight passed. Report the
  evidence and wait for my approval between preflight, prepare, smoke, and run.
- Update TODO.md at the end of each completed request.

First, perform read-only checks only:
1. git status --short and git rev-parse HEAD;
2. inspect ownership and free space below /data/jkobo/rtk-splat;
3. run scripts/tools/bootstrap_server_env.sh verify;
4. activate /data/jkobo/rtk-splat/envs/rtk-splat-server and export
   COLMAP_BIN=/data/jkobo/rtk-splat/envs/colmap-rtk/bin/colmap;
5. inspect transfer progress/status, without changing it;
6. after transfer completes, run field1_0703_full_server.sh preflight;
7. report exact GPU, driver, RAM, mount, segment frame count/hash, package
   versions, COLMAP CUDA smoke result, and any blocker. Do not start prepare.

Long stages must use the checked launcher and an external nohup log under
/data/jkobo/rtk-splat/logs. Codex/VS Code connectivity must not own the job.
```

Codex should run in a terminal on the server—either directly over SSH or in a
VS Code Remote-SSH integrated terminal—so it sees the server filesystem and
GPU. The production job itself remains a launcher/nohup process independent of
Codex and the editor session.
