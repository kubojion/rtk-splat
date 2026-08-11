# Headland Stereo-BA Reference Experiment

Run window: 2026-07-29 18:20 to 2026-07-30 09:36 CEST.

This is the first high-fidelity reference run. It reused a 1,344-pair canonical
headland segment, reconstructed a calibrated stereo rig with COLMAP, aligned
the visual trajectory to the immutable RTK camera centers, rebuilt the
pose-specific initialization cloud, and trained a new GS map.

Key outcomes:

- 1,344/1,344 left frames and 2,688/2,688 total images registered.
- 0.818 px mean COLMAP reprojection error.
- 5.11 cm median visual-to-RTK trajectory residual after alignment.
- +3.84 dB masked PSNR over the raw-RTK pose arm.
- 24.445 dB masked PSNR, 25.848 dB corrected masked PSNR, 0.594 SSIM,
  and 0.319 corrected LPIPS.

The mapper itself took about 581 minutes. Total elapsed time was 15 h 16 min.
The feature/match database can be reused for future pose-backend comparisons;
it is not necessary to rerun feature extraction for every controlled sidecar
experiment.

Scientific caveat: held-out GS images participated in SfM pose estimation.
This result measures reconstruction quality, not frozen-map localization.

The complete provenance, hashes, structural COLMAP statistics, environment, and
acceptance gates are in
[golden/headland_stereo_ba.json](golden/headland_stereo_ba.json).

The later contract-v2 all-frame GPU frontend + Global + 65k GS control
reproduced this reference at 24.3639 dB masked and 25.7326 dB corrected masked
PSNR (-0.0810/-0.1153 dB). That validates the modern all-frame path; adaptive
frame-density arms remain pending. See [SEALED_FRONTEND_AB.md](SEALED_FRONTEND_AB.md).
