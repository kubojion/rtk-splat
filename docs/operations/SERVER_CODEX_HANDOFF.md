# Archived server-agent handoff

Status: completed and archived 2026-08-25.

This handoff originally bootstrapped the first 77-minute server run. Do not
paste its historical run commands into the completed project root: the source
segment, pose attempts, TilePlan, tile runs, scene, and review exports are
immutable evidence and refuse overwrite.

The run reached these terminal states:

- all 10,227 selected stereo frames entered a complete diagnostic pose;
- the data-driven planner produced 32 tiles;
- all 32 tiles trained on the RTX 4090;
- the layered scene passed structural, visual, ownership, inventory, and seam
  checks; and
- production georeferencing was rejected by unchanged independent RTK gates.

Use `../../SERVER_RUN.md` only as a historical reproduction record. Current
metrics, artifact paths, and limitations are in
`../milestones/FULL_FIELD_DIAGNOSTIC_V1.md`. The provenance-linked retention
set is in `ARTIFACT_RETENTION.md`.

## Template for a genuinely fresh reproduction

Before any write, an operator or agent must:

1. use a new isolated Git worktree at an exact reviewed commit;
2. use a new workdir and new artifact names;
3. verify the immutable segment, environment, GPU, COLMAP, and free-space
   receipts without modifying them;
4. run the recording-specific launcher's preflight and report its evidence;
5. start long stages independently of the chat/editor session;
6. preserve every partial or failed attempt; and
7. stop fail-closed on a production gate rather than relabelling diagnostic
   output.

Dataset paths belong only in the recording configuration or launcher. Generic
changes belong under `src/rtk_splat/` with focused, tampering, and complete-suite
tests. Do not weaken held-out RTK, fixed-scale, calibration, baseline, visual,
track, ownership, or seam gates during reproduction.
