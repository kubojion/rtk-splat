# RTK-Splat: Results Audit, State of the Art, and Top-Level Paper Roadmap

**Research snapshot:** 2026-07-29  
**Local implementation audited:** commit `643b5d237767beb59afe7605188e546c370d95cf`  
**Literature index audited:** [3D-Vision-World/awesome-NeRF-and-3DGS-SLAM](https://github.com/3D-Vision-World/awesome-NeRF-and-3DGS-SLAM) at commit `2433e00afc6b16eb763186931c201e9e82653e5b` (2026-07-28)  
**Scope:** read-only analysis of the implementation, experiment artifacts under `/home/jion_kubo/agromap4d_work`, primary papers, and public source repositories. No implementation was changed. This report is the only new project file.

This report supersedes the broad RTK-Splat novelty wording in the older `study/RELATED_WORK.md` and `study/IMPLEMENTATION_REFERENCES.md`. Several relevant works appeared or became discoverable in 2026, so the earlier statement that georeferenced or RTK-constrained Gaussian Splatting itself was novel is no longer safe.

---

## 1. Executive verdict

### 1.1 Is the current result good?

It is a **credible and useful prototype result**, but it is not yet evidence of a publishable state-of-the-art method.

The latest `field_turn/tile_turn2` run shows that the pipeline can:

- Build a globally coherent, metric, georeferenced Gaussian map from dual-RTK-derived poses and offline stereo.
- Process a 40 m turning field sequence with 1,344 stereo pairs.
- Reach 20.61 dB on its current depth-masked, interleaved validation frames.
- Finish within the available GPU budget with a compact exported map.
- Remain stable enough that the training-view and held-out-frame scores are almost identical.

Those are meaningful engineering achievements. The result is not a trivial failure.

However, it currently cannot support a top-paper quality claim because:

- There is no independent test traversal in this result.
- A validation camera is only 2.87 cm, on median, from its nearest training camera.
- The main score is evaluated only where the system's own SGBM depth succeeds.
- Full-frame PSNR is 12.19 dB, SSIM is 0.321, and LPIPS is 0.536.
- The render is visibly soft, contains sky/horizon artifacts, and bakes a camera-attached object into the world.
- There is no independent geometry reference, raw global trajectory metric, failure-rate benchmark, or common baseline protocol.
- Increasing the Gaussian cap by 67% and iterations by 44% improved masked PSNR by only 0.226 dB.

The current evidence says **the representation has reached a pose/depth/observation-model ceiling**, not that it needs more Gaussians.

### 1.2 Is the project still novel?

The broad claim is not novel:

> “We make georeferenced 3D Gaussian Splatting using RTK/GNSS.”

That territory is already occupied by GS-GVINS, LD3DGS-SLAM, GeoRefGS, RTK-constrained UAV GS, an ISRS 2026 RTK-GNSS workflow, and commercial RTK-to-GS pipelines.

The following individual ideas are also already occupied:

- Outdoor passive-stereo GS-SLAM.
- Learned stereo depth for outdoor GS.
- Uncertainty-aware GS-SLAM.
- Hessian/eigenspectrum observability reasoning.
- Ray-constrained Gaussian depth adjustment.
- Surface-aligned or disk-like Gaussians.
- Chunked/out-of-core Gaussian maps.
- Geometry-only Gaussian maps.
- Keyframe ownership for coherent map correction.

A defensible contribution remains possible as a **specific integrated method**:

> **Observability-routed dual-RTK and rigid-stereo Gaussian mapping for long, slow, low-parallax agricultural operation, with calibrated state/depth uncertainty, failure-aware map admission, and bounded geospatial tiles.**

The contribution cannot be the sensor list. It must be the way the system reasons about:

1. What the full three-dimensional dual-RTK baseline observes.
2. What it does not observe.
3. When stereo, terrain geometry, or optional gravity should fill only weak state directions.
4. When pose/depth uncertainty makes a frame unsafe to write into the persistent map.
5. How that uncertainty propagates to geospatial Gaussian locations without corrupting their rendering footprint.
6. How the system stays bounded and recoverable over hectare-scale fields and RTK outages.

This package appears meaningfully different from the closest work found, but any eventual “first” wording needs another formal literature/patent search immediately before submission.

### 1.3 Does adding an IMU remove the novelty?

No. An IMU does not automatically remove novelty. GS-GVINS is not simply “any GS method that touches an IMU”; it is a tightly integrated monocular GNSS-visual-inertial navigation system using raw GNSS/INS/visual/GS factors.

The stronger design for this project is:

- **Core profile:** dual RTK + stereo, no IMU required.
- **Optional profile:** add a calibrated gravity/gyro factor only when observability analysis says it is useful.
- Report both profiles as an ablation.

This preserves the important scientific claim that the system works without inertial navigation, while testing whether gravity provides useful roll and outage support.

At slow constant speed, accelerometer-derived translation is indeed weak and vibration-prone. That does **not** make all IMU information useless. Gravity can constrain roll/pitch even at zero speed, and a gyroscope can bridge short angular gaps. The correct response is to calibrate, gate, and ablate it—not to make it mandatory and not to discard it on intuition alone.

### 1.4 The most important next experiment

Do **not** start another 2.5-million-Gaussian, 65k-iteration budget sweep.

First run an equal-compute diagnostic ladder:

1. Current poses.
2. Smoothed covariance-weighted RTK trajectory.
3. Full 3-D dual-RTK baseline factor.
4. Full RTK + stereo temporal relative-motion factors.
5. The same plus ground-normal roll.
6. The same plus calibrated gravity only.
7. Oracle/reference poses.

Measure training-view sharpness as well as independent-view quality. If the oracle-pose run substantially improves even training PSNR and fine detail, the dominant bottleneck is confirmed before months are spent changing the mapper.

---

## 2. What the current system actually does

### 2.1 Effective dataflow

```text
RTK fix + RELPOS baseline
       │
       ├── absolute antenna position → local ENU translation
       └── RELPOS north/east only → yaw
                                      │
left/right rectified JPEGs             │
       │                               ▼
       └── OpenCV SGBM → depth ──► fixed-roll/fixed-pitch camera poses
                                      │
                                      ▼
                    back-project all valid left depths
                                      │
                                      ▼
                      voxel first-hit + random cap
                                      │
                                      ▼
                    large Gaussian initialization
                                      │
                                      ▼
             left RGB + left depth GS optimization
               optional right RGB / pose experiment
                                      │
                                      ▼
                 interleaved held-out-frame metrics
```

The code is compact and understandable for a prototype. The main mapping loop is in [`rtk_splat/train.py`](rtk_splat/train.py), pose construction in [`rtk_splat/poses.py`](rtk_splat/poses.py), bag extraction in [`rtk_splat/bagio.py`](rtk_splat/bagio.py), and SGBM in [`rtk_splat/depth.py`](rtk_splat/depth.py).

### 2.2 Are both stereo cameras currently used?

The answer has two parts.

| Operation | Left camera | Right camera | Current status |
|---|---:|---:|---|
| Offline disparity/depth | Yes | Yes | Both images are passed to SGBM. |
| Gaussian color supervision in the main runs | Yes | No | Main mapper primarily trains on left RGB. |
| Metric depth supervision | Yes | Indirectly | Depth is computed from the left-right pair but supervised in the left frame only. |
| Camera intrinsics | Left `K` | Assumed equivalent | Right `K`, both `P` matrices, and the recorded rectified transform are not fully used. |
| Right-view depth mask | Left-specific | Left mask reused | Geometrically invalid near disparity and occlusion boundaries. |
| Right RGB experiment | Yes | Yes | Confounded with pose changes and unequal image-presentation budget. |
| Stereo uncertainty | No | No | No right-left consistency, occlusion class, confidence, or disparity variance. |

Therefore, the project **uses both images to estimate depth**, but it does **not yet exploit the rigid stereo rig as two fully calibrated photometric observations of one shared body pose**.

The earlier right-camera experiment cannot establish that right RGB does not help. It:

- Reused the left depth-valid mask in the right view.
- Assumed matching intrinsics.
- Had no right-frame depth or occlusion reasoning.
- Kept a fixed iteration count, so each image received fewer presentations.
- Combined stereo supervision with a pose-model change.

At the field's median depth, disparities are not negligible relative to the mask dilation. A right mask copied from the left can be wrong by tens of pixels.

### 2.3 Can depth be reconstructed from this bag?

Yes. Rectified synchronized left/right images are recorded, so disparity and metric depth can be recomputed offline:

\[
z = \frac{fB}{d}.
\]

Possible offline backends include:

- SGBM with left-right and temporal consistency.
- IGEV-Stereo.
- FoundationStereo or Fast-FoundationStereo.
- DEFOM-Stereo.
- MonSter or another outdoor-generalized stereo model.

What cannot be reconstructed exactly is the ZED SDK's original native depth product. The ZED depth engine may use proprietary matching, temporal filtering, confidence, sensor metadata, and parameters not preserved in the bag. Offline stereo can be better or worse, but it is a new derived product, not recovery of the missing SDK output.

The recorded JPEG compression also limits the best achievable stereo precision relative to native lossless images. It does not make reconstruction impossible, but future data collection should store lossless or lightly compressed rectified stereo plus the exact calibration and exposure metadata.

---

## 3. Local results audit

### 3.1 Latest `field_turn` result

| Property | `tile_turn2` |
|---|---:|
| Path / duration | 40.06 m / approximately 450 s |
| Stereo pairs | 1,344 |
| Train / validation / test | 1,176 / 168 / **0** |
| Median motion between frames | 3.02 cm |
| Median validation-to-nearest-training distance | **2.87 cm** |
| Raw SGBM-valid area | 66.4% |
| Dilated supervised area | 72.0% |
| Raw fused depth samples | 44.77 million |
| Initial cloud | capped at 4.0 million |
| Scene extent | 40.3 × 43.4 × 10.4 m |
| Gaussian cap | 2.5 million |
| Iterations | 65,000 |
| Training time | approximately 3 h 49 min |
| Segment / checkpoint / export | 3.7 GB / 220 MB / 65 MB |
| Exported Gaussians | 734,850, or 29.4% of cap |
| Full-frame PSNR | **12.194 dB** |
| Depth-masked PSNR | **20.610 dB** |
| Color-corrected masked PSNR | 21.152 dB |
| Test-time pose-aligned masked PSNR | 20.936 dB |
| SSIM | **0.321** |
| LPIPS | **0.536** |
| Sampled training masked PSNR | 20.634 dB |

The representative artifact is:

`/home/jion_kubo/agromap4d_work/field_turn/runs/tile_turn2/renders/eval_65000.jpg`

Its major visible failure modes are:

- Unsupported black sky.
- Large sky/horizon blobs.
- Soft crop and soil texture.
- A camera/robot-attached foreground object smeared into world coordinates.
- Loss of thin leaves, stems, and high-frequency ground texture.

### 3.2 What the training-versus-validation result means

Sampled training masked PSNR is 20.634 dB and validation masked PSNR is 20.610 dB. This is not the usual pattern of a model sharply fitting training views and failing to generalize.

It indicates a shared ceiling affecting both sets:

- Inconsistent local camera poses.
- Stereo depth noise and hard-valid weighting.
- Wind-driven or ego-attached content.
- Exposure/illumination inconsistency.
- Weak initialization geometry.
- An optimization schedule that begins at the Gaussian cap.

The current validation set is still useful as a regression check. It is not an independent novel-view benchmark because adjacent training cameras are only centimetres away.

### 3.3 Capacity ablation

| Run | Cap | Iterations | Full PSNR | Masked PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| `tile_turn1` | 1.5 M | 45k | 11.991 | 20.384 | 0.310 | 0.550 |
| `tile_turn2` | 2.5 M | 65k | 12.194 | 20.610 | 0.321 | 0.536 |
| Change | +66.7% | +44.4% | +0.203 | **+0.226** | +0.011 | -0.014 |

This is a poor quality-per-resource trade. Approximately 71% of the second run's Gaussian budget does not survive export. The system should begin with fewer high-confidence surface elements and grow only where supported.

There is a useful optimization clue: masked PSNR increases sharply after the MCMC relocation phase ends. A stable finishing stage improves the result more than continued aggressive relocation.

### 3.4 Cross-experiment summary

Numbers below are local evaluator outputs and are not automatically comparable to paper tables from other systems.

| Dataset/run | Full PSNR | Masked PSNR | Masked CC | Masked aligned | SSIM | LPIPS | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Straight `e2_masked` | 12.050 | 21.787 | — | — | 0.327 | — | Stronger than turn under easy near-duplicate protocol. |
| Straight `e3_tilt` | 12.006 | 21.764 | — | — | 0.326 | — | Current optional tilt path shows no gain; not a valid general IMU verdict. |
| Straight `e4_refined` | 11.896 | 20.383 | — | 22.099 | 0.305 | — | Raw pose refinement hurts; oracle test-time alignment diagnoses pose sensitivity. |
| Straight `e5_overnight` | 11.870 | 21.196 | 22.660 | 21.510 | 0.337 | — | Large color-correction gap. |
| Straight `e6_lr01` | 12.238 | 21.369 | 22.680 | 21.690 | 0.345 | 0.450 | Best later straight-run perceptual result. |
| Straight `e7_clean` | 12.081 | 21.209 | 22.675 | 21.522 | 0.342 | 0.450 | Similar ceiling; more tuning did not dominate. |
| Hangar `tile_hangar3` | 14.472 | 23.477 | 23.797 | 23.732 | 0.435 | — | Rigid control scene is much easier. |
| Hangar `tile_hangar4` | 14.599 | 23.468 | 23.763 | 23.722 | 0.434 | 0.547 | Stable plateau. |
| Field turn `tile_turn1` | 11.991 | 20.384 | 21.010 | 20.672 | 0.310 | 0.550 | Turning/dynamic field degradation. |
| Field turn `tile_turn2` | 12.194 | 20.610 | 21.152 | 20.936 | 0.321 | 0.536 | More capacity gives a small gain. |
| AgriGS adapter cam 1 | 12.088 | 12.165 | 16.901 | 12.499 | 0.380 | 0.573 | Reverse-view adapter stress test, not AgriGS full system. |
| AgriGS adapter cam 2 | 12.071 | 12.524 | 16.869 | 12.759 | 0.404 | 0.619 | Same caveat. |
| AgriGS adapter cam 2 pose-opt | 11.686 | 12.214 | 16.865 | 12.845 | 0.382 | 0.593 | Current pose optimization does not help raw quality. |

The approximately 2.86 dB gap between the latest hangar and field-turn masked scores supports the hypothesis that field dynamics, lighting, stereo quality, and trajectory consistency—not only rasterizer capacity—are limiting performance.

The AgriGS adapter runs use its published poses and one camera at a time. AgriGS's complete system instead uses LiDAR odometry, three ZED cameras, dense camera point clouds, and its own mapping/evaluation protocol. These local adapter numbers must never be presented as a direct win or loss against the AgriGS paper.

### 3.5 Why slow motion is a special estimation problem

The median maximum-axis RTK position variance in the field result is approximately:

\[
0.000460\ \mathrm{m}^2,
\]

or about 2.15 cm standard deviation. Median frame-to-frame motion is only 3.02 cm.

This means:

- RTK is excellent for absolute position and low-frequency global gauge.
- Raw frame-to-frame RTK displacement has poor signal-to-noise at this sampling density.
- Small independent pose fluctuations blur multi-view reconstruction even while the global path looks correct.

The stored heading accuracy is approximately 0.41°. At \(f_x \approx 1119\), that is on the order of eight pixels of angular uncertainty. With a camera lever arm around 3.18 m, it can also create approximately 2.3 cm of lateral camera-position uncertainty.

This leads to the correct sensor decomposition:

```text
Dual RTK: absolute, low-frequency position and baseline-supported attitude
Stereo:   smooth local relative motion, metric depth, and missing attitude information
IMU:      optional gravity and short angular bridge; not mandatory position propagation
```

Slow motion is not a reason to ignore vision. It creates many redundant views and allows temporal stereo/multi-view filtering, but it also makes duplicate-frame ingestion and RTK jitter particularly damaging.

---

## 4. Current implementation audit

### 4.1 What is already good

- WGS84-to-ENU conversion uses a proper geodesy library.
- Initialization leakage between train and validation is explicitly guarded.
- Splits are stored in a manifest.
- Carrier state is gated.
- Yaw smoothing has explicit support.
- Pose corrections are bounded and gauge projected.
- MCMC has an explicit stop/settle phase.
- Mean learning-rate decay is present.
- The AgriGS adapter is isolated rather than mixed into the main bag reader.
- The prototype is small: roughly 1,800 Python source lines.

These qualities are worth preserving. The next implementation should remain explicit and compact.

### 4.2 Pose and calibration gaps

[`rtk_splat/bagio.py`](rtk_splat/bagio.py) uses only the north/east RELPOS components to construct yaw. It discards:

- The vertical baseline component.
- `acc_n`, `acc_e`, and `acc_d`.
- Heading accuracy.
- Baseline consistency.
- Most validity/quality information.
- The physical endpoint formulation of both antennas.

[`rtk_splat/poses.py`](rtk_splat/poses.py) hardcodes roll to zero and applies fixed mount pitch. One rigid baseline vector determines two rotational degrees of freedom, not all three. If that baseline is approximately body-forward, its full 3-D direction provides yaw and pitch; rotation about the baseline remains unobservable and corresponds approximately to roll.

The existing optional IMU tilt path:

- Converts a fused ZED quaternion directly to roll/pitch.
- Has no calibrated IMU-to-body rotation.
- Does not use covariance or innovation.
- Has no vibration gate.
- Removes a median offset but not dynamic axis mixing.

Its no-gain result does not prove that calibrated gravity is useless.

The extraction artifacts do not retain enough estimation evidence:

- Exact left/right timestamps and their residual.
- Per-frame RTK endpoints/baseline/covariance/status.
- Per-frame pose covariance.
- Calibration version/hash.
- IMU innovations.
- Exposure metadata.

The camera model uses only the left `K` matrix and a hardcoded baseline. The right intrinsics, projection matrices, and recorded rectified stereo transform should be authoritative and versioned.

### 4.3 Stereo gaps

The configured depth backend is not a real interface: the CLI constructs SGBM regardless of the backend label.

SGBM currently produces a depth value and hard valid bit, but no:

- Left-right consistency.
- Occlusion label.
- Match confidence.
- Disparity variance.
- Temporal consistency.
- Calibration of expected metric depth error.

Depth uncertainty grows rapidly with distance:

\[
\sigma_z \approx \frac{fB}{d^2}\sigma_d.
\]

With a roughly 12 cm baseline, a one-pixel disparity error at long range becomes a large metric depth error. Treating every “valid” pixel equally is physically unjustified.

Cross-bag stereo handling also does not match the implied interface: the implementation opens the bag containing the left topic and searches for both topics inside it. Actual left-right timing error is bucketed rather than retained and reported.

### 4.4 Initialization and mapping gaps

The current initializer:

- Concatenates approximately 44.8 million raw points before reducing.
- Uses a hash without collision validation.
- Keeps the first voxel hit instead of robustly fusing observations.
- Discards later view count, color diversity, and uncertainty.
- Randomly caps the cloud.
- Contains some extreme depth outliers.

Training then:

- Starts at or near the maximum Gaussian population.
- Gives all initial splats equal scale.
- Uses random quaternions instead of local normals.
- Spends MCMC effort moving an oversized weak initialization.
- Uses one image per iteration even though `batch_size` is configured.
- Reads JPEG and compressed NPZ data from disk every iteration.
- Saves only the final model, without full optimizer/MCMC/exposure state.

The mapper lacks:

- Persistent observation/support counts.
- A candidate/confirmed/transient lifecycle.
- Per-splat geospatial uncertainty.
- Contribution- and geometry-aware pruning.
- Stable ownership for later trajectory corrections.
- Fixed geospatial tile persistence with optimizer state.
- Sky, ego, or dynamic-vegetation handling.

### 4.5 Evaluation and reproducibility gaps

The main masked metric is defined by SGBM success. This:

- Selects texture-rich, easy-stereo regions.
- Excludes much of the sky and distant field.
- Makes the evaluation mask method-dependent.
- Prevents a fair comparison with a method using a different depth backend.

Per-image color correction fits an affine transform using the held-out ground-truth image. Test-time pose alignment optimizes on the same ground-truth pixels that are then scored. These are useful oracle diagnostics, but not acceptable headline metrics.

Other correctness/reproducibility issues include:

- `batch_size` is unused.
- `min_straight_s` is unused.
- CRS is saved as null due to a metadata path mismatch.
- `TrajectoryFile` constructs an incomplete `RtkTrack`.
- Torch is not seeded.
- Run folders omit resolved config, commit, environment, dataset checksum, calibration hash, and seed.
- There are no periodic resumable checkpoints.
- There are no unit tests, integration tests, CI, dependency lock, package metadata, project README, or license in this subproject.
- `__pycache__` files are tracked.
- Several large YAML files duplicate most settings.

These are not merely cosmetic. A top-level result must be traceable, deterministic, restartable, and evaluator-independent.

---

## 5. Adversarial novelty audit

### 5.1 Direct novelty threats

| Work | What it already establishes | Consequence for this project |
|---|---|---|
| [GS-GVINS](https://arxiv.org/abs/2502.10975), IEEE Access 2025 | Tightly integrated raw GNSS/RTK, visual, inertial, feature, motion, and GS photometric factors for outdoor navigation; motion-aware GS pruning. | No “first GNSS + GS,” “first RTK-aided GS navigation,” or generic tight-fusion claim. |
| [LD3DGS-SLAM](https://doi.org/10.1109/JIOT.2025.3638876), IoTJ 2026 | Long-distance monocular GS-SLAM with GNSS-aided graph localization and rendered-view support during GNSS denial; UAV tests over long distances. | No “first long-distance GNSS-aided GS-SLAM.” Public material does not establish its antenna configuration, so do not invent that distinction. |
| [GeoRefGS](https://doi.org/10.3390/drones10030195), Drones 2026 | Georeferencing embedded into GS training through a learnable similarity transform and geographic loss. Reports sub-5.4 cm distance error in a simulated UAV environment. | No “first georeferenced GS” or “first geographic loss.” Its simulation-only validation leaves a real agricultural robustness gap. |
| [Kim et al., RTK-GPS-constrained GS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7009801), June 2026 | RTK camera positions directly constrain per-camera translation/rotation with multi-view reprojection on three urban UAV datasets; reports GCP RMSE 0.286–0.495 m. | No “first RTK-constrained camera-pose refinement in GS.” |
| [ISRS 2026 A3-3](https://www.rssj.or.jp/isrs2026/program_final.html) | RTK-GNSS used to determine absolute camera orientation for smartphone 3DGS initialization in cadastral reconstruction. | Even an RTK-to-GS workflow claim is too broad. |
| [AgriGS-SLAM](https://arxiv.org/abs/2510.26358) | Agricultural/orchard Gaussian mapping using LiDAR odometry, three ZED cameras, and depth/point clouds, with reverse-direction validation. | No “first agricultural GS-SLAM.” Compete on fewer sensors, absolute georeferencing, geometry, reliability, and scale. |
| [BGS-SLAM](https://arxiv.org/abs/2507.23677) | Outdoor passive binocular GS-SLAM with deep stereo, weighted depth, normal, smoothness, and sky handling. | No “first outdoor stereo GS,” “first deep-stereo GS,” or generic stereo-only novelty. |
| [LSG-SLAM](https://arxiv.org/abs/2505.09915) | Large-scale stereo GS-SLAM with IGEV depth, multimodal tracking, submaps, loops, and structure refinement. | A direct sensor-matched baseline; submaps or learned stereo alone are not novel. |
| [Photo-SLAM](https://arxiv.org/abs/2311.16728) | Mature real-time monocular/stereo/RGB-D feature SLAM plus Gaussian mapping. | Stereo support and decoupled feature tracking are established. |
| [AERGS-SLAM](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_AERGS-SLAM_Auto-Exposure-Robust_Stereo_3D_Gaussian_Splatting_SLAM_CVPR_2026_paper.html) | Stereo GS-SLAM robust to auto-exposure through illumination-robust localization, coarse-to-fine optimization, and a camera exposure network. | Exposure robustness alone is not enough; a simpler physically constrained radiometric model is preferable. |
| [Spectral GS-SLAM](https://arxiv.org/abs/2606.21258) | Hessian eigenspectrum analysis and degeneracy-subspace information injection for GS tracking. | No generic “first observability-aware/spectral GS” claim. The new part must be dual-RTK/stereo/gravity routing plus map-admission consequences. |
| [SGAD-SLAM](https://arxiv.org/abs/2603.21055) | Gaussians adjusted only along camera rays using a learned scalar depth offset. | Ray-constrained splat adjustment itself is not novel. |
| [VarSplat](https://arxiv.org/abs/2603.09673) | Learned per-splat appearance variance rendered and used in tracking, registration, and loop decisions. | “Uncertainty-aware GS” is too broad. Distinguish sensor, pose, world-location, render-footprint, and static uncertainty. |
| [DiskChunGS](https://arxiv.org/abs/2511.23030) | Chunked large-scale GS with GPU/CPU/disk lifecycle and persisted optimizer state. | Chunking or out-of-core storage is not novel by itself. |
| [GeoGS-SLAM: geometry-only](https://arxiv.org/abs/2607.07452) | Geometry-only disk Gaussians, local-plane initialization, multi-view geometry, and coherent global correction. | No geometry-only, local-plane, disk, or coherent-correction claim alone. |
| [GeoGS-SLAM: geometric priors](https://arxiv.org/abs/2607.11184) | Monocular online GS with learned depth/pose/confidence priors, joint optimization, loops, and owner-based correction. | Learned priors and keyframe ownership alone are occupied. |

### 5.2 Claims that should not appear in a paper

Avoid:

- “The first georeferenced 3D Gaussian Splatting method.”
- “The first RTK/GNSS Gaussian map.”
- “The first outdoor stereo Gaussian SLAM.”
- “The first uncertainty-aware Gaussian SLAM.”
- “The first observability-aware Gaussian SLAM.”
- “The first ray-constrained Gaussian mapper.”
- “The first surface-aligned or geometry-only GS.”
- “The first scalable/chunked Gaussian mapper.”
- “State of the art” based on unmatched masks, sensors, datasets, or author-reported paper numbers.

Also avoid calling the present pipeline “purely camera and RTK” if a ZED fused orientation is enabled in a run. Every experiment should declare the exact signals used.

### 5.3 A potentially defensible claim

After a final pre-submission search, a cautious formulation could be:

> We present a georeferenced Gaussian field mapper that jointly exploits raw three-dimensional dual-RTK baseline observations and synchronized metric stereo through online observability routing. The estimator protects well-observed RTK directions, fills weak attitude and local-motion directions using stereo and optional gravity, and propagates the resulting uncertainty into map admission and bounded geospatial tile management.

If evidence supports it, the application-specific extension is:

> The system targets long, slow, low-parallax agricultural trajectories with repetitive rows, wind-driven vegetation, RTK fix/float/outage transitions, and hectare-scale bounded operation.

The strongest eventual contribution statement is a measured capability, not a priority claim:

> Under equal sensors and compute, the method improves raw global trajectory accuracy, independent-view fidelity, metric surface accuracy, failure rate, and map cost over the strongest public stereo and agricultural baselines.

---

## 6. Code and architecture audit of the strongest systems

Public repositories were inspected at fixed commits where available. A paper description and a public implementation are not assumed to be identical.

### 6.1 Summary table

| System | Inputs / role | Public code status | Best idea to take | Main reason not to copy wholesale |
|---|---|---|---|---|
| [AgriGS-SLAM](https://github.com/AIRLab-POLIMI/agri-gs-slam/tree/336b1db0d61f2672f13e709e9c95ad102fe8b77d) | LiDAR odometry + 3 ZED RGB-D/point clouds; orchard GS | Public v1.0.0 | Independent reverse traversal, synchronized shared map, local active splats | Paper/code mismatches, LiDAR dependency, nondeterministic scheduler, weak evaluator and map correction |
| GS-GVINS | Monocular + raw GNSS + IMU + GS navigation | Exact GS-GVINS code not found | Covariance-weighted physical factors and GNSS-state handling | Requires raw GNSS/IMU and solves a different navigation problem; cannot claim exact reproduction without code |
| BGS-SLAM | Passive stereo outdoor GS | Paper found; public implementation not found | Deep stereo, selective weighted depth, normals, smoothness, sky mask | Not executable as a direct baseline; reported 1.37 s mapping per frame |
| [AERGS-SLAM](https://github.com/zzy-2021/AERGS-SLAM/tree/f0d8782d48210db8e73b83b05358dc5fe37695f9) | Stereo exposure-robust GS-SLAM | Public, one main commit | Decoupled tracking/mapping and age-aware coarse-to-fine optimization | Large C++ stack, committed build artifacts/vendor code, hardcoded parameters, no useful tests |
| [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM/tree/2b8cfdba02d7010ea18a5c02d3fc227253804fee) | Large-scale stereo GS-SLAM | Public | IGEV, feature fallback, verified loops, isotropic-to-anisotropic refinement | Manual multi-stage scripts, hardcoded paths, no propagated stereo covariance |
| [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM/tree/f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac) | Mature feature SLAM + Gaussian map | Public | ORB-SLAM3 frontend, asynchronous mapper, pyramid training | Heavy/old C++ dependency stack; right camera mainly provides depth, not symmetric RGB supervision |
| [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS/tree/f179f372bcde0f2f20a9ce8fe4369c045c621b67) | Large-scale stereo/RGB-D/mono map | Public | Stable chunk IDs, active-set paging, persisted Adam state, external-pose mode | General 3-D chunks are more complex than field tiles; depth confidence and mapping loss remain weak |
| [SGAD-SLAM](https://github.com/MachinePerceptionLab/SGAD-SLAM/tree/2dca26dcef242edf07c3c361be390d3ca254aa43) | RGB-D indoor mapping | Public | Ray-only depth adjustment and simple geometry/appearance split | One Gaussian per selected pixel scales poorly; indoor RGB-D assumptions |
| [VarSplat](https://github.com/anhthuan1999/varsplat/tree/65e3623b9e0e602be9176e41bc914bde8bf48411) | Uncertainty-aware RGB-D SLAM | Public | Rendered uncertainty used across tracking/registration/loops | Appearance variance is not calibrated geospatial uncertainty; large GPU and sequence tuning |
| Spectral GS-SLAM | Degeneracy-aware GS tracking | Project/paper; code link unavailable | Weak-subspace fusion rather than unconditional sensor mixing | Spectral observability is already its contribution; must extend it materially |
| [TVG-SLAM](https://github.com/MagicTZ/TVG-SLAM/tree/6f84b0b94b198b93e39e6497920814c662408374) | Tri-view robust GS-SLAM | Repository contains README/assets only | Multi-view trust and delayed map admission | No auditable implementation; heavy learned matching would be unnecessary as an always-on path |
| [KiloGS-SLAM](https://github.com/3dagentworld/KiloGS-SLAM/tree/f1007086de316f0c8082e6a04a01273abeed10d6) | Kilometer-scale monocular GS | README-only at audit | Condition-triggered fallback, map lifecycle, information-based insertion | Code unavailable and foundation-model stack is excessive for reliable RTK-stereo core |
| [Pocket-SLAM](https://github.com/UMN-ZhaoLab/Pocket-SLAM/tree/f7712335c3820d0e7d2a6cf34f5c33bea81f6c6d) | Memory-pruned monocular GS-SLAM | Public | Screen-contribution-aware spatial budget | Its published repo example improves memory/FPS while retaining very large ATE; not a localization target |
| [RTG-SLAM](https://github.com/MisEty/RTG-SLAM/tree/49dada148551961e2dee38040a4f760fa1b0f01d) | Efficient RGB-D mapper | Public | Candidate/stable pools, observation support, residual-driven growth | Indoor reliable-depth assumptions and no global/geospatial uncertainty |
| [HI-SLAM2](https://github.com/Willyzw/HI-SLAM2/tree/76c833c7d8ed474f0f3ba18056c1803e032a537f) | Globally corrected monocular GS | Public | Explicit Gaussian owner keyframe for coherent corrections | Heavy DROID/learned-normal pipeline and monocular domain assumptions |
| [WildGS-SLAM](https://github.com/GradientSpaces/WildGS-SLAM/tree/be187eabbe6862cef3cfe87031ee2e64ad8c4cec) | Dynamic monocular GS-SLAM | Public | Uncertainty participates in tracking, mapping, BA, and filtering | Complex pretrained semantic stack; uncertainty can explain away model errors |
| [MAGiC-SLAM](https://github.com/VladimirYugay/MAGiC-SLAM/tree/3ea4e0d1d652a08baef06a6d5f4ec4bc8ee73687) | Multi-agent submap GS | Public | Clean submap boundaries and registration/pose-graph flow | GT initialization, fixed submaps, equal graph weights, multi-GPU assumptions |
| [SplatReg](https://github.com/Archerkattri/splatreg/tree/c8f1c6de1680204ff15392f519947efab2426155) | Splat-to-splat registration library | Public | Typed API, analytic Jacobians, covariance/ambiguity, CI and extensive tests | Not a complete SLAM system; opacity-as-confidence is unsafe for vegetation |

### 6.2 AgriGS-SLAM: fair use and important code findings

AgriGS's public frontend is not camera tracking. Ouster LiDAR DLO/GICP provides the poses; ZED RGB and precomputed point clouds initialize and train the map. Fixposition/RTK data are loaded but do not georeference the public map.

Valuable ideas:

- Clockwise training and counter-clockwise validation.
- One shared map supervised by synchronized cameras.
- Local GPU working set.
- Mapping work bounded between physical keyframes.
- The intended idea of depth uncertainty.

Important public code/paper differences found:

- The default depth/KL flag does not activate the described probabilistic loss.
- If the alternative depth flag is manually enabled, the implementation is a categorical softmax KL, not a per-ray Gaussian depth likelihood.
- Camera point-cloud GICP refinement exists in C++ but is not called by the Python pipeline.
- The described second-order motion model is first-order in code.
- A memory threshold is read but not used as an LRU trigger.
- Loop-closure YAML keys and consumed keys differ; loops are disabled by the internal default.
- Graph corrections update stored poses but do not coherently correct the existing Gaussian map.
- Current-camera 4×4 matrices can be optimized outside SE(3) and can break the rigid camera rig.
- Queue sizes and per-keyframe optimization work depend on thread timing.
- One backward pass is followed by multiple Adam steps using the same gradient.
- Broad exceptions are frequently swallowed.
- The evaluator masks pixels according to nonzero prediction, which can reward incomplete coverage.

Benchmark AgriGS in four separate modes:

1. Exact official public reproduction.
2. Its output under a common evaluator.
3. Oracle mapper comparison using identical poses and depth.
4. Native sensor/cost Pareto: LiDAR + three RGB-D cameras versus dual RTK + one stereo pair.

Do not silently repair public AgriGS code and call the result “official.” A corrected variant must be labeled separately.

### 6.3 Stereo systems: what they reveal

#### BGS-SLAM

BGS-SLAM is a central prior because it already combines:

- Outdoor passive stereo.
- A deep stereo network.
- ORB-SLAM-based pose tracking.
- Selective weighted depth supervision.
- Normal and smoothness losses.
- Sky segmentation.

Its paper reports that the selected multi-domain stereo model materially affects both rendering and geometry. This supports benchmarking several stereo backends, but also means “replace SGBM with IGEV” is not a method contribution by itself.

The reported system is not real time, and no public code was found. Treat it as a paper-only comparison or request code/results from the authors.

#### LSG-SLAM

LSG-SLAM is the strongest directly runnable stereo inspiration:

- IGEV metric depth.
- SuperPoint/LightGlue/PnP fallback.
- Place retrieval followed by geometric loop verification.
- Continuous submaps.
- Online constrained/isotropic Gaussians followed by anisotropic structure refinement.

Its EuRoC conversion script is dangerous to run on source data because it can delete unmatched images. Create a non-destructive adapter in a separate converted dataset. Its scripts also hardcode calibration and paths, so exact adapters and commit hashes must be documented.

#### Photo-SLAM

Photo-SLAM is a mature full-system stereo baseline. Its robust ORB-SLAM3 frontend and decoupled mapper are useful. Its “stereo” mapper, however, primarily uses the right image to derive depth; Gaussian appearance optimization remains left-focused. This recurring limitation creates a concrete opportunity for **true symmetric rigid-stereo rendering**.

#### AERGS-SLAM

AERGS cleanly separates stereo AirSLAM tracking from GS mapping. It introduces:

- Illumination-robust tracking features.
- New-keyframe low-resolution and mature-keyframe high-resolution optimization.
- Canonical scene radiance plus a camera exposure network.

Borrow the schedule and radiometric separation. Do not begin with a flexible exposure MLP. A monotonic response curve, per-frame exposure/white balance, and vignetting are more interpretable and easier to calibrate.

### 6.4 Scale, lifecycle, correction, and uncertainty

#### DiskChunGS

DiskChunGS is the best direct storage reference:

- Deterministic spatial chunk identifiers.
- Bounded active GPU population.
- CPU/disk keyframe and chunk lifecycle.
- Persisted Gaussian parameters, stable IDs, Adam moments, and optimizer steps.
- External-pose mode.

Its stereo confidence is based largely on inverse-depth edges rather than a calibrated disparity posterior. Mapping depth loss is still ordinary L1. Its right RGB is not a second photometric observation. Borrow the storage model, not its measurement model.

For fields, fixed 2.5-D ENU tiles with halos are simpler than arbitrary 3-D chunks.

#### RTG-SLAM

RTG-SLAM's candidate/stable Gaussian lifecycle is highly relevant:

- New Gaussians are quarantined.
- Observation count provides persistence evidence.
- Residuals drive targeted growth.
- Persistent disagreement can release or remove geometry.
- Depth normals provide surface-aligned initialization.

Upgrade observation count to a calibrated support record containing depth covariance, temporal separation, angular spread, and static probability.

#### HI-SLAM2

The owner-keyframe field is a strong consistency pattern. When a trajectory or pose graph changes, each Gaussian can receive the physical correction of the keyframe/submap that created it. This is safer than transforming whatever happens to be visible after a loop.

For RTK-Splat, ownership should be:

- Stable Gaussian ID.
- Creator keyframe or tile.
- Supporting keyframes.
- Current tile.
- Last trajectory solution version.

#### VarSplat and WildGS-SLAM

These make generic uncertainty wording unsafe. The opportunity is to separate quantities that other systems often conflate:

1. Stereo sensor covariance.
2. Camera trajectory covariance.
3. Gaussian world-location covariance.
4. Gaussian render covariance/physical footprint.
5. Appearance variance.
6. Static/transient probability.

In particular, uncertain location must not be represented by simply making the rendered Gaussian larger. That turns epistemic uncertainty into false physical thickness.

#### Spectral GS-SLAM

Spectral GS-SLAM already decomposes a tracking Hessian and injects supplemental feature information only in degenerate directions. The transferable principle is:

> Fuse a source where it adds information, not merely because it is available.

The new method must extend this principle to:

- Full dual-RTK endpoint/baseline factors.
- Stereo temporal and geometric factors.
- Optional gravity.
- Time offset and selected calibration states.
- Map-write eligibility.
- Per-splat geospatial uncertainty.

### 6.5 Code-quality lessons

The best software-design reference in the surveyed set is SplatReg:

- Typed public API.
- Explicit SE(3)/Sim(3) semantics.
- Analytic Jacobians.
- Robust optimization.
- Covariance and ambiguity outputs.
- CI and substantial tests.
- Deterministic overlap handling.

MAGiC-SLAM has the cleanest high-level submap boundaries.

The local RTK-Splat prototype is smaller and easier to understand than most surveyed SLAM systems. That is an advantage. The goal should be a small set of explicit modules with strict records and invariants—not a fork of a 20,000-line system.

“Redundant” should describe independent measurements and safe fallbacks, not duplicate code, state, or configuration.

---

## 7. Proposed method: observability-routed dual-RTK stereo GS

The provisional method name in this report is **ORDS-GS**:

> **Observability-Routed Dual-RTK Stereo Gaussian Mapping**

This is a working description, not a final paper name.

### 7.1 Inputs and outputs

Required input:

- Synchronized, rectified stereo RGB.
- Calibrated left/right intrinsics and rigid extrinsics.
- Both RTK antenna position/baseline observations or the richest available equivalent.
- RTK covariance, fix state, and quality flags.
- Camera-to-body and antenna lever arms.

Optional input:

- Calibrated IMU gravity and gyro.
- Wheel/nonholonomic motion prior.
- Occasional surveyed control points.

Output:

- A camera/body trajectory in ENU/ECEF with covariance and sensor-health state.
- A tiled Gaussian map in a documented CRS.
- Per-Gaussian world-location uncertainty, render covariance, support, and static probability.
- An explicit record of degraded intervals and rejected/quarantined map updates.

### 7.2 State and physical RTK factors

For keyframe \(k\), estimate body pose:

\[
T_{WB_k} = (R_{WB_k}, t_{WB_k}).
\]

Potential shared states include:

- Camera extrinsic correction.
- RTK lever-arm correction.
- Camera-to-RTK clock offset.
- Stereo scale/baseline correction within calibration priors.

For antenna \(a\) with body-frame lever arm \(l_a\), use the measured world position directly:

\[
r_{p,k} =
p^{\mathrm{meas}}_{W,a,k}
-
\left(t_{WB_k} + R_{WB_k}l_a\right).
\]

For the measured baseline vector between antennas \(a\) and \(b\):

\[
r_{b,k} =
b^{\mathrm{meas}}_{W,k}
-
R_{WB_k}(l_b-l_a).
\]

Weight both residuals with the actual covariance and RTK status. This is superior to converting north/east RELPOS to a scalar yaw before estimation.

The full baseline direction observes two rotational degrees of freedom. Rotation about that vector remains unobservable from a single baseline. That final direction should be supplied by stereo geometry, a terrain/ground normal, or optional gravity—not hardcoded to zero.

### 7.3 Smooth local trajectory

Do not optimize 1,344 unrelated pose corrections. Use:

- Keyframe poses with interpolation, or
- A continuous-time SE(3) spline.

This allows:

- RTK to preserve the absolute gauge.
- Stereo to constrain high-frequency relative motion.
- A shared time offset to be estimated.
- Camera frames between RTK updates to be interpolated consistently.
- Smoothness to be a physically explicit prior rather than post-hoc yaw filtering.

A simple offline three-pass workflow is sufficient:

1. Solve RTK/stereo/optional-gravity trajectory and calibration.
2. Build a coarse Gaussian map.
3. Refine only the trajectory spline and shared calibration within their priors, then rebuild/finalize the map.

### 7.4 Observability router

Linearize the active factors into a local normal matrix:

\[
H = J^\top WJ.
\]

Compute:

\[
H = V\Lambda V^\top.
\]

Define a weak-state projector:

\[
P_{\mathrm{weak}}
=
\sum_{\lambda_i < \tau} v_i v_i^\top.
\]

The exact implementation need not literally multiply every residual by this projector. Equivalent square-root information routing, state freezing, or prior adaptation may be numerically cleaner. The essential behavior is:

- Protect strongly observed RTK position/baseline directions.
- Add stereo information in locally weak motion/attitude directions.
- Add optional gravity only where it contributes.
- Freeze or strongly regularize time/extrinsic states when excitation is insufficient.
- Detect degeneracy rather than silently converging to arbitrary roll or calibration.

Unlike Spectral GS-SLAM, the router should also govern whether a frame may change persistent geometry.

### 7.5 Failure-aware map admission

Every candidate mapping frame receives a health decision based on:

- RTK fix/float/outage state and innovation.
- Baseline length/direction consistency.
- Stereo confidence and left-right consistency.
- Temporal pose information.
- Predicted pose covariance at image time.
- Exposure/blur/ego/dynamic coverage.

Possible actions:

- **Commit:** full persistent map update.
- **Quarantine:** create candidate splats that require later confirmation.
- **Appearance only:** update exposure/color, not geometry.
- **Local submap:** continue during RTK outage with growing covariance.
- **Skip:** use pose propagation but do not contaminate the map.
- **Close safely:** stop the active submap when all sources are degraded.

This connects estimator uncertainty to map reliability, which is more meaningful than reporting covariance after the fact.

### 7.6 Probabilistic stereo depth

For disparity \(d\), focal length \(f\), and baseline \(B\):

\[
z = \frac{fB}{d},
\qquad
\sigma_z \approx \frac{fB}{d^2}\sigma_d.
\]

Each backend should return:

```text
disparity
depth
validity
occlusion class
confidence / disparity variance
left-right consistency
temporal consistency
backend health
```

Use a calibrated robust per-ray likelihood:

\[
\mathcal{L}_z =
\rho\left(
\frac{(\mu_r-z_s)^2}{\sigma_r^2+\sigma_s^2}
+
\log(\sigma_r^2+\sigma_s^2)
\right),
\]

where:

- \(z_s,\sigma_s\) are stereo depth and uncertainty.
- \(\mu_r,\sigma_r\) are rendered ray depth and uncertainty.
- \(\rho\) is a robust loss.
- Occluded or left-right-inconsistent pixels are excluded.

The logarithmic variance term prevents the system from winning by declaring every measurement uncertain.

### 7.7 Ray-constrained structural refinement

SGAD-SLAM already establishes ray-only Gaussian movement. A useful adaptation is:

\[
\mu_i = c_k + \alpha_i(p_i-c_k),
\]

or an inverse-depth offset along the initialization ray.

Use this only during an early geometry stage. It prevents photometric gradients from sliding new Gaussians tangentially across repetitive crop texture before multi-view support exists. Later, confirmed Gaussians may unlock a limited surface-tangent correction.

The contribution would be the combination with calibrated stereo/trajectory covariance and persistence—not the ray constraint by itself.

### 7.8 Separate Gaussian uncertainties

Each Gaussian should contain:

```text
mean_world
render_covariance
geospatial_covariance
opacity
appearance
appearance_variance (optional)
static_probability
support_count
support_time_span
viewing_angle_span
creator_keyframe / tile
last_seen
lifecycle_state
```

Definitions:

- **Render covariance:** physical/visual surface footprint used by rasterization.
- **Geospatial covariance:** uncertainty of the Gaussian's world location, propagated from trajectory and stereo.
- **Static probability:** likelihood that the observed structure persists in world coordinates.
- **Appearance variance:** unexplained view/lighting variation, if modeled.

This separation supports geospatial confidence maps, safe fusion, selective pruning, and honest uncertainty calibration.

### 7.9 True rigid-stereo supervision

For each timestamp:

- Estimate one body pose.
- Derive left and right camera poses from fixed/bounded extrinsics.
- Render both cameras together.
- Use camera-specific intrinsics, masks, exposure, and vignetting.
- Use right-specific depth/occlusion when relevant.
- Enforce measured and rendered stereo consistency only in mutually visible regions.
- Count equal image presentations when comparing mono-left and stereo training.

The right camera should not have an independently optimized 6-DoF pose.

This is a promising differentiator because several nominal stereo GS systems use the right camera mainly to create left-frame depth rather than as a second rigid radiance observation.

### 7.10 Geometry-first lifecycle

Use:

```text
candidate → supported → mature → retired
                  └────→ transient
```

Candidate creation:

- High-confidence stereo pixels.
- Under-rendered regions.
- Large persistent color/depth residual.
- Useful image gradient or depth gradient.
- Spatially balanced per-tile budget.

Promotion:

- At least two temporally separated observations.
- Consistent depth within covariance.
- Sufficient viewing-angle or baseline diversity.
- Static probability above threshold.

Retirement:

- Repeated geometric inconsistency.
- No visibility/contribution over a defined distance.
- Redundant co-planar neighbor with equivalent appearance.
- Persistent association with ego or wind-driven transient content.

Ground can be promoted quickly under strong stereo/normal support. Foliage should require stronger temporal evidence.

### 7.11 Two-stage optimization

Recommended schedule:

1. **Coarse geometry:** fewer high-confidence points, isotropic or normal-aligned, ray-constrained depth, low-resolution images.
2. **Support/growth:** promote confirmed geometry and add only under-covered regions.
3. **Structure refinement:** unlock surface-aligned anisotropy; refine normals and multi-view depth.
4. **Stable appearance:** stop aggressive relocation/densification; optimize canonical color and exposure at higher resolution.
5. **Offline finalization:** revisit selected historical/reverse views and consolidate redundant splats.

This combines the most useful ideas from RTG-SLAM, LSG-SLAM, AERGS-SLAM, and the observed local post-MCMC quality jump.

### 7.12 Outdoor agricultural observation model

At minimum:

- Fixed ego/robot mask.
- Sky/environment layer or explicit sky exclusion from geometry.
- Temporal static score for wind-driven vegetation.
- Canonical Gaussian color.
- Per-frame exposure and white balance.
- Monotonic camera response curve and vignetting, if data support them.
- Robust low-frequency illumination residual for cloud shadows.

The current brightness-versus-PSNR correlation is strongly negative, so radiometry is not optional for high-quality field maps.

### 7.13 Bounded geospatial tiles

Use fixed ENU tiles, for example:

- 10–20 m core size.
- 1–2 m deterministic halo.
- Double-precision global anchor.
- Float local coordinates for rasterization.
- Stable 64-bit Gaussian IDs.
- Tile-owned optimizer and lifecycle state.
- Frustum/trajectory-driven active set.
- Atomic checkpoint and version metadata.

When the trajectory changes:

- Apply creator-keyframe or submap corrections.
- Reassign elements deterministically if they cross tile boundaries.
- Reconcile halos without opacity-only deduplication.

Report map size per hectare and Gaussians per square metre, not only a global cap.

---

## 8. A simple, reliable code architecture

The method can remain small:

```text
rtk_splat/
  config.py
  records.py

  io/
    bag_reader.py
    synchronizer.py
    calibration.py

  estimation/
    trajectory.py
    factors_rtk.py
    factors_stereo.py
    factors_gravity.py
    observability.py
    health.py

  stereo/
    backend.py
    sgbm.py
    learned.py
    confidence.py

  mapping/
    gaussian_state.py
    initializer.py
    optimizer.py
    lifecycle.py
    tile_store.py

  evaluation/
    trajectory.py
    geometry.py
    rendering.py
    uncertainty.py
    resources.py

  artifacts.py
  cli.py
```

### 8.1 Core typed records

`FramePacket`:

```text
frame_id
t_left, t_right, time_residual
left/right image references
left/right calibration
RTK antenna observations
RTK baseline/covariance/status
optional IMU/gravity
blur/exposure/quality metadata
```

`TrajectoryState`:

```text
timestamp
T_W_B
6x6 covariance
source information summary
observability eigenvalues
health state
calibration version
```

`DepthObservation`:

```text
disparity/depth
variance
validity
occlusion
left-right consistency
temporal consistency
backend/version
```

`GaussianState`:

```text
render parameters
geospatial covariance
support/lifecycle/static state
ownership/tile identity
```

### 8.2 Non-negotiable invariants

- `T_A_B` always means “transform coordinates from B into A.”
- One body pose exists per timestamp.
- Left/right extrinsics remain rigid unless a shared, tightly bounded calibration variable is enabled.
- Global coordinates are double precision; raster coordinates are tile-local float.
- Unknown and unused configuration keys are errors.
- All time differences are stored, not hidden by nearest-neighbor pairing.
- Core research execution is synchronous and deterministic.
- Deployment queues, if added, are bounded.
- A poor observation may be skipped; it must not silently update persistent geometry.
- Every output is attributable to code, config, seed, calibration, and data.

### 8.3 Reproducibility requirements

Every run directory should contain:

- Git commit and dirty-worktree status.
- Fully resolved configuration.
- Dataset/bag fingerprint.
- Calibration file and hash.
- Random seeds.
- Package/CUDA/GPU environment.
- Frame manifest and exact splits.
- Per-stage timing, peak VRAM, disk I/O, and power if available.
- Periodic atomic checkpoints including optimizer, strategy, exposure, trajectory, and RNG state.
- Health/rejection log.

Minimum tests:

- ENU/ECEF and transform round trips.
- Lever-arm behavior under yaw, slope, and turn.
- Synthetic full-baseline attitude factors.
- Rank/observability behavior for flat, straight, and turning motion.
- IMU-to-body axis/sign calibration.
- Cross-bag stereo pairing and timestamp gates.
- Right-camera reprojection and occlusion.
- Constant-depth and slanted-plane stereo.
- Disparity-to-depth covariance propagation.
- Collision-free voxel/tile indexing.
- Gaussian ownership correction.
- Checkpoint/resume equivalence.
- Evaluator mask independence.
- A deterministic small synthetic end-to-end sequence.

### 8.4 Practical redundancy

Use:

- Learned stereo primary, SGBM fallback behind one interface.
- Dual RTK and stereo cross-checking.
- Optional gravity when calibrated and healthy.
- Local stereo continuation during RTK float/outage.
- RTK anchoring after recovery.
- Map-write rejection when both sources are weak.

Avoid:

- Two independent pose representations.
- Separate left/right trajectory states.
- Duplicate configuration trees.
- Broad exception swallowing.
- A plugin framework before two real backends exist.
- An always-on foundation model where a lightweight physical factor is sufficient.

---

## 9. Benchmarking existing algorithms fairly

### 9.1 Four benchmark tracks

#### Track A: oracle mapper

Purpose: isolate Gaussian map quality from trajectory quality.

Give every mapper:

- Identical camera poses.
- Identical calibration.
- Identical images.
- Identical stereo depth and confidence where supported.
- Identical train/test views.
- Equal render-step or wall-clock budgets.

Compare:

- Current RTK-Splat mapper.
- Proposed mapper.
- DiskChunGS mapping path.
- Photo-SLAM mapping path where separable.
- LSG-SLAM mapping/refinement.
- RTG-SLAM/SGAD/VarSplat adapted to identical metric depth.
- AgriGS official and corrected mapper variants where feasible.

#### Track B: sensor-matched full system

Required sensors:

- One stereo pair.
- Dual RTK.
- No LiDAR.
- No required IMU.

Compare:

- Current RTK-Splat.
- Proposed core method.
- RTK trajectory + ORB-SLAM3 stereo constraints + common mapper.
- LSG-SLAM/Photo-SLAM/DiskChunGS stereo, with an external georeferencing stage reported separately.
- BGS-SLAM if code becomes available.

#### Track C: optional gravity/IMU

Compare:

- Core dual-RTK + stereo.
- Core + gravity only.
- Core + gravity/gyro.
- A conventional visual-inertial/GNSS system if compatible raw data exist.

The no-IMU core remains the main method. This track quantifies when inertial support helps or hurts.

#### Track D: native-sensor and cost/quality Pareto

Run methods in their intended sensor configurations:

- AgriGS: LiDAR + three RGB-D cameras.
- GS-GVINS: GNSS + monocular + IMU, if reproducible.
- Proposed system: dual RTK + stereo, optional gravity declared.

Report sensor cost, calibration burden, compute, energy, reliability, map quality, and global accuracy. This does not claim a controlled algorithm-only comparison; it evaluates practical systems.

### 9.2 Baseline feasibility

| Baseline | Direct execution now? | Adapter effort | Correct interpretation |
|---|---:|---:|---|
| Current RTK-Splat | Yes | None | Internal baseline. |
| ORB-SLAM3 stereo | Yes, public | Medium | Strong trajectory baseline; attach common fixed-pose GS mapper. |
| Photo-SLAM stereo | Public | Medium/high | Mature full system; older dependency stack. |
| LSG-SLAM | Public | High | Best sensor-matched learned-stereo/large-scale baseline. Use a non-destructive data converter. |
| DiskChunGS | Public | High | Best out-of-core/external-pose baseline. |
| AgriGS-SLAM | Public | High | Native sensor baseline and oracle-mapper comparison; public code has important caveats. |
| AERGS-SLAM | Public | High | Exposure-robust stereo baseline; substantial build/adapter work. |
| SGAD-SLAM | Public | Medium | Oracle depth/mapper component comparison, not native outdoor SLAM. |
| VarSplat | Public | Medium/high | Uncertainty comparison in oracle-depth or adapted track. |
| RTG-SLAM | Public | Medium | Lifecycle/map-efficiency component baseline. |
| BGS-SLAM | No public code found | Blocked on authors/reimplementation | Paper-only numbers are not a direct common-data result. |
| Spectral GS-SLAM | No usable public code found | Blocked/reimplementation | Use conceptual ablation or request code. |
| KiloGS-SLAM | README-only at audit | Blocked | Paper/reference only. |
| TVG-SLAM | README/assets only at audit | Blocked | Paper/reference only. |
| GS-GVINS | Exact code not found | Blocked/high | Do not claim direct reproduction. Request author code/data or label a reimplementation. |
| LD3DGS-SLAM | No public code found | Blocked | Related-work comparison, not executable baseline. |

### 9.3 Can GS-GVINS be benchmarked directly?

Not yet on the present evidence.

It is also important to define what “surpass GS-GVINS quality” means. The GS-GVINS paper evaluates **navigation APE**, not PSNR/SSIM/LPIPS or surface reconstruction accuracy. Its open-sky results are:

| Sequence | GS-GVINS translation / rotation APE | GICI translation / rotation APE |
|---|---:|---:|
| Open-sky A | 0.073 m / 3.748° | 0.071 m / 3.293° |
| Open-sky B | 0.084 m / 3.154° | 0.081 m / 3.028° |

In those two open-sky sequences, GICI is slightly better, and the paper explains that RTK dominates while visual/inertial additions have limited influence. GS-GVINS's larger gains occur in degraded suburban/urban GNSS. Its published table therefore provides a useful localization target, but not a Gaussian-map quality target.

The present RTK-Splat runs have no independent trajectory ground truth, so they cannot yet be declared better or worse than 7–8 cm open-sky APE. Conversely, because GS-GVINS does not report a common quantitative rendering/geometry table, a future RTK-Splat map can only demonstrate better visual or geometric quality through a new common-data evaluation—not by comparing its PSNR to a nonexistent GS-GVINS PSNR.

The paper also reports a large research compute setup (four partitions of an NVIDIA A100 MIG configuration, 40 GB VRAM allocation context, 70 GB RAM). A successful dual-RTK/stereo system with centimetric localization, stronger measured map geometry, and bounded operation on substantially smaller hardware would be a meaningful practical advantage.

Obstacles:

- Exact GS-GVINS source code was not found.
- Its estimator expects raw GNSS/RTK, IMU, visual features, and GS factors.
- The current bag appears to preserve processed fix/RELPOS and ZED fused orientation, not necessarily the raw pseudorange, carrier phase, ephemeris, and calibrated IMU stream its underlying framework expects.
- Its paper emphasizes localization, not a common PSNR/geometry protocol.

Valid options:

1. Ask the authors for source, configuration, raw datasets, and evaluator.
2. Run their reported public dataset if an executable release is provided.
3. Compare against open GVINS/GICI independently and label it accurately.
4. Implement the paper from its equations and call it a reimplementation, not GS-GVINS official.
5. Compare published results only in a clearly separated table with sensor/dataset caveats.

### 9.4 Common dataset adapter

Create a read-only canonical sequence format:

```text
sequence/
  calibration.yaml
  frames.csv
  images/left/
  images/right/
  rtk/observations.csv
  imu/optional.csv
  reference/trajectory.csv
  reference/geometry.*
  masks/
  splits/
```

`frames.csv` should include:

- Frame ID and exact timestamps.
- Image paths.
- Left-right time residual.
- RTK association residual.
- Sensor status.
- Blur/exposure statistics.
- Split and traversal ID.

Each baseline receives a generated adapter directory. Raw data are never edited or deleted.

### 9.5 Required data collection

At least:

- Separate outbound/training and reverse/inbound test traversals.
- Multiple fields, crop types, row orientations, and slopes.
- Straight segments, headland turns, and revisit loops.
- Slow and normal operating speeds.
- Calm and windy conditions.
- Stable and changing illumination.
- Controlled RTK fixed, float, outage, false-fix/recovery intervals.
- A rigid control scene.
- Repeated days if long-term robustness is claimed.

Ground truth:

- Total station or surveyed checkpoints for absolute pose/geometry.
- TLS or high-quality LiDAR for surface geometry.
- Surveyed row centerlines and control targets.
- Independent reference trajectory where possible.

Without independent geometry and trajectory reference, a geospatial-accuracy paper is not supportable.

### 9.6 Metrics

#### Trajectory and attitude

- Absolute ENU/ECEF translation RMSE **without alignment**.
- Aligned ATE as a secondary diagnostic.
- RPE by time and travelled distance.
- Roll, pitch, and yaw error.
- RTK outage drift rate.
- Recovery jump and recovery time.
- Failure/intervention count per kilometre or hectare.

#### Rendering

- Raw full-frame PSNR, SSIM, and LPIPS.
- Fixed-mask versions using one dataset mask shared by all methods.
- Coverage percentage.
- Near-path versus off-path bins.
- Reverse-direction independent traversal.
- No per-image ground-truth color fit in the headline.
- No test-time pose optimization on scored pixels in the headline.

#### Geometry and geospatial accuracy

- Accuracy/completeness Chamfer distances.
- F-score at multiple centimetric thresholds.
- Depth RMSE/MAE by distance.
- Normal consistency.
- Ground-height and canopy-height error.
- Row-centerline error.
- GCP/checkpoint 3-D RMSE.
- Map deformation across tile boundaries and RTK recovery.

#### Uncertainty and reliability

- Negative log likelihood.
- Calibration/coverage curves.
- Selective-risk curve: quality when low-confidence observations are rejected.
- Failure-detection AUROC/AUPRC.
- Fraction of frames committed, quarantined, appearance-only, or skipped.
- Accuracy conditional on RTK fixed/float/outage.

#### Resources

- End-to-end FPS and stage latency p50/p95.
- Peak VRAM and RAM.
- Disk I/O.
- Energy.
- Map bytes per hectare.
- Gaussians per square metre.
- Checkpoint/recovery time.

### 9.7 Experimental controls

- At least three seeds where stochastic optimization affects output.
- Fixed train/test traversals.
- Fixed calibration and masks.
- Equal image presentations, not merely equal iterations.
- Both equal-compute and fixed-epoch results.
- Same image resolution unless a method fundamentally requires otherwise.
- Separate author-reported and locally reproduced numbers.
- Record failures rather than truncating them silently.
- Report confidence intervals across sequences, not only per-frame variance.

### 9.8 Ablation ladder

| ID | Change | Question answered |
|---|---|---|
| A0 | Current pipeline | Baseline. |
| A1 | Preserve timestamps, full RTK vector/covariance/status, complete calibration | Does correct evidence handling matter before algorithm changes? |
| A2 | Covariance-weighted RTK spline | Is high-frequency RTK jitter the main blur source? |
| A3 | Full 3-D baseline factor | Does pitch/attitude improve without IMU? |
| A4 | Stereo temporal relative pose | Does local visual motion sharpen the map while RTK preserves global gauge? |
| A5 | Observability routing, no IMU | Does weak-subspace fusion beat unconditional priors? |
| A6 | Calibrated gravity factor | When does optional IMU help? |
| A7 | SGBM left-right/temporal confidence | How much is hard-valid depth hurting geometry? |
| A8 | Learned stereo with calibrated confidence | Does a stronger backend improve real field geometry? |
| A9 | Ray-constrained probabilistic depth | Does physically constrained refinement reduce floaters/blur? |
| A10 | True rigid-stereo RGB supervision | Does the second radiance view improve geometry and off-path rendering? |
| A11 | Candidate/persistent/transient lifecycle | Does quarantined map admission improve reliability under wind/outages? |
| A12 | Sky/ego/radiometry model | How much quality loss is not geometry? |
| A13 | ENU tiles with persisted state | Does quality stay constant while scale grows? |
| A14 | Offline anisotropic finalization | What quality is available beyond the online map? |

### 9.9 Success criteria before claiming SOTA

A credible target—not a current claim—is:

- Raw global translation RMSE below 5 cm in RTK-fixed open field.
- Attitude error below 0.5° where reference is available.
- Zero unrecovered failures across full field traversals.
- At least 1 dB raw reverse-traversal PSNR improvement over the strongest sensor-matched runnable baseline.
- At least 20% improvement in independent geometry error.
- Calibrated uncertainty that reliably rejects bad map writes.
- Bounded VRAM and approximately linear bytes/hectare.
- Better sensor-cost/quality Pareto than AgriGS's LiDAR + three-camera configuration.

The exact thresholds should be frozen before the final benchmark and not selected after seeing all results.

---

## 10. Reliability state machine

| RTK | Stereo | Optional IMU | Estimation behavior | Mapping behavior |
|---|---|---|---|---|
| Fixed/healthy | Good | Any | RTK absolute gauge + stereo local motion | Full persistent update |
| Fixed/healthy | Poor | Healthy or absent | RTK pose, weak visual correction | Skip geometry; optional appearance only |
| Float/degraded | Good | Healthy or absent | Continue local metric submap; covariance grows | Candidate/quarantine updates |
| Outage | Good | Healthy | Stereo local odometry + gravity/gyro bridge | Local submap, no absolute promotion |
| Fixed but inconsistent | Good | Any | Robustly downweight/reject false RTK fix | Preserve local submap pending recovery |
| Any | Good | Vibration/high innovation | Disable inertial factor | Continue RTK/stereo |
| Degraded | Degraded | Any | Declare loss of reliable observability | Close submap safely; do not write map |
| Recovered fixed | Good | Any | Robustly estimate alignment to pending submap | Correct/anchor ownership groups, then promote |

This state machine should be derived from continuous covariance/innovation measures, not only hard carrier flags. The table describes actions; the observability router supplies the quantitative decision.

---

## 11. Prioritized execution roadmap

### P0: Make every result trustworthy

1. Freeze a canonical dataset format and common evaluator.
2. Save exact timestamps, synchronization residuals, full calibration, RTK observations/covariance/status, and optional IMU records.
3. Add deterministic seeds, resolved config, commit, environment, and data hashes.
4. Add periodic resumable checkpoints.
5. Collect an independent reverse traversal and independent geometry/pose reference.
6. Replace method-dependent headline masks with fixed masks plus coverage.

**Gate:** no new method claim until a run can be exactly reproduced and scored independently.

### P1: Diagnose and solve trajectory sharpness

1. Run current-versus-oracle pose mapping at equal compute.
2. Implement the covariance-weighted RTK spline.
3. Use the full 3-D baseline vector.
4. Add stereo temporal relative motion.
5. Add observability/rank diagnostics.
6. Add terrain-normal roll.
7. Add calibrated gravity-only ablation.

**Gate:** if oracle poses do not materially improve training-view sharpness, shift priority from trajectory to depth/dynamics/radiometry. If they do, do not expand the mapper until the trajectory gap closes.

### P2: Establish trustworthy stereo geometry

1. Implement a real backend interface.
2. Add SGBM right-left consistency, occlusion, and temporal agreement.
3. Benchmark SGBM, IGEV, Foundation/Fast-FoundationStereo, and one outdoor-generalized model.
4. Calibrate disparity/depth uncertainty against independent geometry.
5. Use lossless stereo in new recordings.
6. Add robust per-ray depth likelihood.

**Gate:** select the backend using geometry, calibration, robustness, latency, and license—not PSNR alone.

### P3: Improve the Gaussian method

1. Initialize 300k–500k high-confidence spatially balanced points, not the cap.
2. Align initial covariance with robust local normals.
3. Add ray-constrained geometry refinement.
4. Add true synchronized left/right RGB supervision.
5. Add candidate/supported/mature/transient lifecycle.
6. Separate geospatial and render covariance.
7. Add ego/sky/static/radiometric handling.
8. Use a stable high-resolution finishing phase.

**Gate:** each feature must improve at least one independent metric without hiding a regression in coverage, geometry, or resources.

### P4: Make it hectare-scale and failure-safe

1. Fixed ENU tiles with halos.
2. Stable IDs and persisted optimizer state.
3. Information-based keyframes.
4. Ownership-based correction after trajectory updates.
5. RTK outage/recovery experiments.
6. Memory, energy, and bytes/hectare measurement.

**Gate:** demonstrate bounded resource use and no permanent map corruption under controlled failures.

### P5: Baselines and paper

1. Build safe adapters for ORB-SLAM3, Photo-SLAM, LSG-SLAM, and DiskChunGS.
2. Reproduce AgriGS exactly, then under the common evaluator.
3. Contact BGS-SLAM, GS-GVINS, LD3DGS-SLAM, and Spectral GS-SLAM authors for code/data.
4. Run oracle-mapper and native-sensor tracks.
5. Run the full ablation ladder over multiple fields and conditions.
6. Freeze claims only after all tables are complete.

---

## 12. Paper strategy

### 12.1 Is it paper-worthy now?

Not yet.

The implementation is a strong prototype, and the field result is useful diagnostic evidence. It currently lacks the independent data, baselines, geometry truth, reliability experiments, and method contribution needed for a top robotics or vision paper.

### 12.2 What can reach a top robotics venue?

A strong ICRA/IROS/RA-L-style paper can be built around:

- A physically grounded dual-RTK/stereo estimator.
- Explicit observability and failure-state reasoning.
- Reliable map admission under RTK/stereo degradation.
- Bounded hectare-scale mapping.
- A demanding agricultural dataset/protocol.
- Strong absolute trajectory, geometry, rendering, and resource results.
- Fewer sensors and lower cost than LiDAR-heavy agricultural systems.

The systems contribution and evaluation can be the centre of the paper.

### 12.3 What is needed for a top vision venue?

For CVPR/ICCV/ECCV level, sensor fusion and an agricultural application are probably insufficient by themselves. The paper needs a representation or learning/optimization contribution that generalizes beyond agriculture, for example:

- Calibrated separation of geospatial location covariance and rendering covariance.
- Observability-conditioned Gaussian map admission.
- A probabilistic ray-constrained stereo Gaussian objective.
- Rigid-stereo multi-view supervision with uncertainty and lifecycle.
- A representation that improves both geometry and novel-view rendering across agricultural and standard outdoor datasets.

The method should be evaluated on KITTI/KITTI-360/EuRoC or another standard outdoor/stereo benchmark as well as the agricultural data.

### 12.4 Candidate contribution set

A coherent paper should have three contributions, not ten unrelated modules:

1. **Estimator:** raw dual-RTK baseline and metric stereo are fused through state observability, with optional gravity and explicit failure modes.
2. **Map:** trajectory/depth uncertainty governs ray-constrained Gaussian creation and persistence, while geospatial and rendering covariances remain distinct.
3. **System/benchmark:** bounded ENU tiles and a multi-condition agricultural evaluation demonstrate centimetric absolute mapping, independent-view quality, and graceful RTK degradation with fewer sensors.

Radiometry, dynamics, stereo backends, and tile storage support these contributions; they should not each be claimed as independent novelty.

### 12.5 Possible title directions

- **Observability-Routed Dual-RTK Stereo Gaussian Mapping for Agricultural Fields**
- **GeoField-GS: Failure-Aware Georeferenced Gaussian Mapping from Dual RTK and Stereo**
- **From Centimetric Poses to Trustworthy Splats: Uncertainty-Routed Gaussian Field Mapping**
- **AgroGeoGS: Bounded Georeferenced Gaussian Mapping under Low Parallax and RTK Degradation**

Names must be checked for prior use before adoption.

### 12.6 The paper's central story

The most compelling story is:

> Open-sky agricultural robots already carry accurate RTK, but raw framewise RTK is not a sharp local camera trajectory when the machine moves only centimetres between images. A full dual-antenna baseline provides strong absolute information yet leaves a rotational null direction; stereo provides metric geometry and smooth local motion but fails in repetitive, textureless, or dynamic vegetation. We explicitly route these complementary observations by what is currently observable and allow uncertain frames to localize without contaminating the persistent Gaussian map. This produces a centimetric, bounded, photorealistic field map with graceful sensor degradation and substantially fewer sensors than LiDAR-heavy agricultural systems.

That is a method story. “We passed RTK poses into gsplat” is not.

---

## 13. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| A new dual-RTK/stereo GS paper appears before submission | Medium | High | Avoid priority-dependent framing; lead with measured properties and final search. |
| Oracle poses do not improve field sharpness | Medium | High | Diagnose depth, dynamics, rolling/exposure, and representation using rigid controls. |
| Learned stereo fails on fine vegetation | High | High | Calibrate multiple backends, use classical fallback and temporal multi-view confirmation. |
| JPEG data cap stereo quality | High for current bags | Medium | Quantify it; record lossless stereo in new datasets. |
| Optional IMU is vibration-dominated | Medium | Medium | Calibrate, monitor innovation, gravity-only mode, automatic factor disable. |
| RTK covariance is optimistic or false-fix occurs | Medium | High | Innovation tests, baseline consistency, switchable robust factors, outage protocol. |
| Wind prevents one static photorealistic map | High | High | Static/transient separation, calm/windy subsets, report selective coverage honestly. |
| Baseline builds consume project time | High | Medium | Start with ORB-SLAM3, Photo-SLAM, LSG-SLAM, DiskChunGS; request author help early. |
| Licensing prevents code reuse | Medium | High | Treat GPL/noncommercial systems as external executables; keep core implementation license-clean. |
| Tiling introduces seams or duplicate geometry | Medium | Medium | Deterministic halos, ownership, cross-tile tests, common geometry evaluation. |
| Top-vision novelty judged too application-specific | Medium | High | Make uncertainty/map-admission objective general and test on standard outdoor data. |

---

## 14. Exact next twelve actions

1. Freeze this result set and preserve `tile_turn2` as the current reference.
2. Define the common raw evaluator before touching the mapper.
3. Collect or identify one independent reverse field traversal and one geometry/pose reference.
4. Extend extraction artifacts conceptually to preserve full timing, RTK vector/covariance/status, and calibration; do not train until the data contract is fixed.
5. Run the oracle-pose versus current-pose mapper test.
6. Build the covariance-weighted full-baseline trajectory prototype.
7. Add stereo temporal relative-pose constraints and inspect observability eigenvalues.
8. Re-run the mapper with identical depth and compute to isolate trajectory improvement.
9. Implement SGBM left-right/occlusion/temporal confidence and calibrate it against reference geometry.
10. Compare one learned stereo backend at equal geometry/latency conditions.
11. Run a correctly calibrated, equal-image-presentation rigid-stereo RGB experiment.
12. Only after these diagnostics, implement lifecycle/uncertainty/tiles and begin public-baseline adapters.

The first publishable milestone is not “a prettier render.” It is a controlled table showing which part of the pose-depth-map chain explains the current ceiling.

---

## 15. Primary sources and audited code snapshots

### Survey and closest application work

- [Awesome NeRF and 3DGS SLAM index](https://github.com/3D-Vision-World/awesome-NeRF-and-3DGS-SLAM)
- [AgriGS-SLAM paper](https://arxiv.org/abs/2510.26358)
- [AgriGS-SLAM code, audited commit](https://github.com/AIRLab-POLIMI/agri-gs-slam/tree/336b1db0d61f2672f13e709e9c95ad102fe8b77d)

### GNSS, RTK, and georeferencing

- [GS-GVINS](https://arxiv.org/abs/2502.10975)
- [LD3DGS-SLAM DOI](https://doi.org/10.1109/JIOT.2025.3638876)
- [GeoRefGS](https://doi.org/10.3390/drones10030195)
- [RTK-GPS-constrained camera pose refinement for georeferenced GS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7009801)
- [ISRS 2026 program, A3-3 RTK-GNSS with 3DGS](https://www.rssj.or.jp/isrs2026/program_final.html)
- [GVINS public implementation](https://github.com/HKUST-Aerial-Robotics/GVINS)

### Stereo and outdoor GS-SLAM

- [BGS-SLAM](https://arxiv.org/abs/2507.23677)
- [LSG-SLAM paper](https://arxiv.org/abs/2505.09915)
- [LSG-SLAM code, audited commit](https://github.com/lsg-slam/LSG-SLAM/tree/2b8cfdba02d7010ea18a5c02d3fc227253804fee)
- [Photo-SLAM paper](https://arxiv.org/abs/2311.16728)
- [Photo-SLAM code, audited commit](https://github.com/HuajianUP/Photo-SLAM/tree/f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac)
- [AERGS-SLAM paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_AERGS-SLAM_Auto-Exposure-Robust_Stereo_3D_Gaussian_Splatting_SLAM_CVPR_2026_paper.html)
- [AERGS-SLAM code, audited commit](https://github.com/zzy-2021/AERGS-SLAM/tree/f0d8782d48210db8e73b83b05358dc5fe37695f9)

### Scale, uncertainty, geometry, and correction

- [DiskChunGS paper](https://arxiv.org/abs/2511.23030)
- [DiskChunGS code, audited commit](https://github.com/leggedrobotics/DiskChunGS/tree/f179f372bcde0f2f20a9ce8fe4369c045c621b67)
- [SGAD-SLAM paper](https://arxiv.org/abs/2603.21055)
- [SGAD-SLAM code, audited commit](https://github.com/MachinePerceptionLab/SGAD-SLAM/tree/2dca26dcef242edf07c3c361be390d3ca254aa43)
- [VarSplat paper](https://arxiv.org/abs/2603.09673)
- [VarSplat code, audited commit](https://github.com/anhthuan1999/varsplat/tree/65e3623b9e0e602be9176e41bc914bde8bf48411)
- [Spectral GS-SLAM](https://arxiv.org/abs/2606.21258)
- [TVG-SLAM paper](https://arxiv.org/abs/2506.23207)
- [TVG-SLAM repository state audited](https://github.com/MagicTZ/TVG-SLAM/tree/6f84b0b94b198b93e39e6497920814c662408374)
- [KiloGS-SLAM](https://arxiv.org/abs/2606.30436)
- [Pocket-SLAM paper](https://arxiv.org/abs/2606.24796)
- [Pocket-SLAM code, audited commit](https://github.com/UMN-ZhaoLab/Pocket-SLAM/tree/f7712335c3820d0e7d2a6cf34f5c33bea81f6c6d)
- [RTG-SLAM code, audited commit](https://github.com/MisEty/RTG-SLAM/tree/49dada148551961e2dee38040a4f760fa1b0f01d)
- [HI-SLAM2 code, audited commit](https://github.com/Willyzw/HI-SLAM2/tree/76c833c7d8ed474f0f3ba18056c1803e032a537f)
- [WildGS-SLAM code, audited commit](https://github.com/GradientSpaces/WildGS-SLAM/tree/be187eabbe6862cef3cfe87031ee2e64ad8c4cec)
- [MAGiC-SLAM code, audited commit](https://github.com/VladimirYugay/MAGiC-SLAM/tree/3ea4e0d1d652a08baef06a6d5f4ec4bc8ee73687)
- [SplatReg code, audited commit](https://github.com/Archerkattri/splatreg/tree/c8f1c6de1680204ff15392f519947efab2426155)
- [GeoGS-SLAM: geometry-only](https://arxiv.org/abs/2607.07452)
- [GeoGS-SLAM: geometric priors](https://arxiv.org/abs/2607.11184)

### Stereo backends

- [IGEV-Stereo](https://github.com/gangweix/IGEV)
- [FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
- [DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo)

### Research limitations

- Public code can differ from the version used for a paper's tables.
- Several 2026 systems have no code or only placeholder repositories.
- Author-reported metrics from different datasets, masks, sensors, poses, and resolutions are not directly comparable.
- The literature search is extensive but cannot prove absence. Novelty wording must be rechecked at submission time.
- No implementation benchmark was run as part of this read-only audit.
