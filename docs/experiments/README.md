# Experiment records

These files record measured evidence; they are not an active task list. Use
`../../TODO.md` for current work.

## Accepted references

- `HEADLAND_STEREO_BA.md` and `golden/headland_stereo_ba.json`: accepted
  incremental stereo-BA headland reference.
- `golden/headland_global_mapper.json`: accepted historical reduced-Global
  control.

## Modern reproduction and pending efficiency test

- `SEALED_FRONTEND_AB.md`: completed GPU/CPU all-frame pose comparison and GPU
  65k GS reproduction; adaptive frame-density arms remain pending.
- `migrations/headland_contract_v2.json`: immutable v1-to-v2 data migration
  receipt, not a rendering result.

## Controlled tiled-scene evidence

- `../methods/TILED_SCENE.md`: completed 2026-08-10 same-code monolithic versus
  two-tile headland A/B. The core-owned scene passes all source, ownership,
  whole-view, and exact 1 m seam-band gates and improves masked/corrected masked
  PSNR by 0.305/0.380 dB. It is accepted rendering/seam evidence but remains
  provisional and metric-claim ineligible because its historical source pose
  is `legacy_unassessed` under the current evidence schema.

## Transfer and diagnostic evidence

- `ROSARIO_V2_SEQUENCE5_PILOT.md`: accepted IR/depth pose and GS plus
  metric-ineligible colour diagnostics and a rejected monocular pose probe.
- CitrusFarm evidence is summarized in `../../PROGRESS.md`; it includes
  rejected full-window georeferencing controls and a bounded diagnostic
  retrace.

Never compare PSNR between different datasets as though it were one benchmark.
Every result must retain its sensor inputs, split, pose status, georeferencing
eligibility, configuration hashes, and failure/diagnostic labels.
