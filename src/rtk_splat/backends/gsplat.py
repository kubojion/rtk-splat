"""Train one Gaussian-Splatting tile from posed frames + init cloud.

gsplat MCMC strategy with a hard Gaussian cap (the VRAM knob).

Views: LEFT camera frames, plus (optionally) the RIGHT eye of each rectified
pair as extra training views -- one rigid pose per pair. Held-out evaluation
is always LEFT-only so metrics stay comparable across configurations.

Refinement (optional, config `train.pose_opt` / `train.exposure_opt`):
- bounded per-pair SE(3) deltas polish photometric consistency without ever
  leaving the trust radius of the given (RTK) poses;
- per-pair affine exposure absorbs lighting drift; it modifies the loss, not
  the map.

Losses (inside the supervision mask): L1 + ssim_lambda*(1-SSIM)
+ depth_lambda*|log D_render - log D_stereo| (left views only)
+ MCMC opacity/scale regularizers. Anti-needle scale/anisotropy projections
after each step.

Metrics: psnr (full frame), psnr_masked (supervised region -- the headline),
psnr_near (bottom half). The final evaluation additionally reports
psnr_masked_aligned: held-out views re-scored after BARF-style test-time pose
alignment, the honest metric when training refines poses.

Model selection: every evaluation may replace an atomic, optimizer-independent
checkpoint when held-out color-corrected masked PSNR strictly improves. Ties
keep the earlier checkpoint. At normal completion the selected Gaussians are
restored, evaluated once more, and used for both params.pt and the exported PLY.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from gsplat.exporter import export_splats
from gsplat.strategy import MCMCStrategy
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

from .gsplat_cameras import (
    ExposureAdjust,
    PoseAdjust,
    align_eval_pose,
    right_c2w,
)
from rtk_splat.backends.pose_evidence import (
    cloud_georeferencing_evidence,
    pose_georeferencing_evidence,
    require_render_permission,
    splat_output_name,
)
from rtk_splat.core.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    pose_artifact_name,
    pose_fingerprint,
    verify_cloud_matches_poses,
    verify_cloud_tile_binding,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.core.runtime_resolution import configuration_evidence

_LPIPS = None
_GAUSSIAN_PARAMETER_NAMES = (
    "means", "quats", "scales", "opacities", "sh0", "shN"
)
_BEST_METRIC_NAME = "psnr_masked_cc"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_sha256(values) -> str:
    payload = json.dumps(
        [int(value) for value in values], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value) -> None:
    """Publish JSON by one same-filesystem rename."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_better_model(candidate: float, incumbent: float | None) -> bool:
    """Strict comparison makes exact ties select the earlier evaluation."""
    candidate = float(candidate)
    if math.isnan(candidate):
        return False
    return incumbent is None or candidate > incumbent


def _checkpoint_state(params, pose_adj=None) -> dict[str, torch.Tensor]:
    """CPU model tensors only: deliberately excludes every optimizer state."""
    state = {
        name: params[name].detach().cpu().clone()
        for name in _GAUSSIAN_PARAMETER_NAMES
    }
    if pose_adj is not None:
        state["pose_deltas_raw"] = pose_adj.raw.weight.detach().cpu().clone()
    return state


def _atomic_save_checkpoint(path: Path, params, pose_adj=None) -> None:
    """Atomically replace the recoverable best-so-far model checkpoint."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(_checkpoint_state(params, pose_adj), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint_gaussians(path: Path, device) -> dict[str, torch.Tensor]:
    """Load only Gaussian tensors; optional camera diagnostics stay on disk."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing = [name for name in _GAUSSIAN_PARAMETER_NAMES if name not in state]
    if missing:
        raise RuntimeError(f"best model checkpoint is missing tensors: {missing}")
    return {name: state[name].to(device) for name in _GAUSSIAN_PARAMETER_NAMES}


def _step_mcmc_before_cutoff(
    strategy,
    params,
    optimizers,
    state,
    step: int,
    info,
    lr: float,
    refine_stop_iter: int,
) -> bool:
    """Do not enter third-party MCMC once the declared cutoff is reached.

    gsplat's MCMCStrategy disables relocation at its cutoff but injects
    positional noise unconditionally. Keeping the call outside the cutoff is
    therefore necessary for a real noise-free settling tail.
    """
    if step >= refine_stop_iter:
        return False
    strategy.step_post_backward(
        params, optimizers, state, step, info, lr=lr
    )
    return True


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in sorted(vars(value).items())
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _training_config_snapshot(cfg) -> tuple[dict, str]:
    snapshot = {
        name: _jsonable(getattr(cfg, name))
        for name in ("pose", "depth", "cloud", "train")
    }
    if hasattr(cfg, "runtime_resolution"):
        snapshot["runtime_resolution"] = _jsonable(cfg.runtime_resolution)
    encoded = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":")).encode()
    return snapshot, hashlib.sha256(encoded).hexdigest()


def _lpips_model(device):
    """Lazy singleton; VGG weights are cached locally by torchvision."""
    global _LPIPS
    if _LPIPS is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        _LPIPS = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True).to(device).eval()
    return _LPIPS


@torch.no_grad()
def _masked_lpips(rgb, rgb_gt, sup, device):
    """Perceptual distance on the supervised region: zero out unsupervised
    pixels identically in both images, crop to the mask bounding box."""
    ys, xs = torch.where(sup)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sup3 = sup[..., None]
    a = (rgb * sup3)[y0:y1, x0:x1].permute(2, 0, 1)[None]
    b = (rgb_gt * sup3)[y0:y1, x0:x1].permute(2, 0, 1)[None]
    return float(_lpips_model(device)(a.clamp(0, 1), b.clamp(0, 1)))


def _load_frame(
    seg_dir: Path,
    frames: dict[str, np.ndarray],
    idx: int,
    dilate_px: int,
    device,
    side="L",
    context_mask: np.ndarray | None = None,
):
    """(rgb, depth, valid, supervise_mask) for one view of pair `idx`.

    The mask comes from LEFT stereo validity; for right views it is reused
    (boundary error ~ disparity << mask scale -- it only excludes large
    sky/far regions). Depth tensors are meaningful for left views only.
    """
    image_field = "left_image_path" if side == "L" else "right_image_path"
    if image_field not in frames:
        raise RuntimeError(f"segment has no {image_field} for {side}-camera view")
    image_path = seg_dir / str(frames[image_field][idx])
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"cannot read training image: {image_path}")
    rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float() / 255.0
    if "depth_path" not in frames or not str(frames["depth_path"][idx]):
        raise RuntimeError(
            "GS training needs recorded or computed depth declared in frames.npz"
        )
    depth_path = seg_dir / str(frames["depth_path"][idx])
    with np.load(depth_path, allow_pickle=False) as archive:
        valid_np = np.asarray(archive["valid"], dtype=bool)
        depth_np = np.asarray(archive["depth"], dtype=np.float32)
    if context_mask is not None:
        context = np.asarray(context_mask, dtype=bool)
        if context.shape != valid_np.shape:
            raise RuntimeError(
                f"tile context mask shape {context.shape} != depth shape "
                f"{valid_np.shape} for frame {idx}"
            )
        valid_np &= context
    if not valid_np.any():
        raise RuntimeError(f"frame {idx} has no valid depth inside tile context")
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    supervise_np = cv2.dilate(valid_np.astype(np.uint8), kernel).astype(bool)
    depth = torch.from_numpy(depth_np)
    return (rgb.to(device), depth.to(device),
            torch.from_numpy(valid_np).to(device),
            torch.from_numpy(supervise_np).to(device))


def init_params(cloud_npz: Path, cfg, device) -> torch.nn.ParameterDict:
    data = np.load(cloud_npz)
    pts = torch.from_numpy(data["xyz"]).float()
    rgb = torch.from_numpy(data["rgb"]).float() / 255.0
    # MCMC never grows past cap_max but also never shrinks below the init
    # count, so the init must respect the cap (it is the VRAM budget).
    if len(pts) > cfg.train.max_gaussians:
        keep = torch.randperm(len(pts))[: cfg.train.max_gaussians]
        pts, rgb = pts[keep], rgb[keep]
    n = len(pts)
    sh_dim = (cfg.train.sh_degree + 1) ** 2
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "quats": torch.nn.Parameter(torch.rand(n, 4) * 0.02 + torch.tensor([1.0, 0, 0, 0])),
        "scales": torch.nn.Parameter(
            torch.full((n, 3), math.log(cfg.cloud.voxel_m * cfg.train.init_scale_mult))),
        "opacities": torch.nn.Parameter(
            torch.full((n,), torch.logit(torch.tensor(cfg.train.init_opacity)).item())),
        "sh0": torch.nn.Parameter(((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)),
        "shN": torch.nn.Parameter(torch.zeros(n, sh_dim - 1, 3)),
    }).to(device)


def make_optimizers(params, cfg) -> dict:
    lr = cfg.train.lr
    return {name: torch.optim.Adam([params[name]], lr=float(getattr(lr, name)), eps=1e-15)
            for name in ("means", "scales", "quats", "opacities", "sh0", "shN")}


def render(params, viewmat, k_mat, width, height, sh_degree,
           rasterize_mode="classic"):
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    return rasterization(
        params["means"],
        torch.nn.functional.normalize(params["quats"], dim=-1),
        torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]),
        colors,
        viewmat[None], k_mat[None], width, height,
        sh_degree=sh_degree, render_mode="RGB+ED", packed=True,
        rasterize_mode=rasterize_mode,
    )


def _color_correct(rgb, rgb_gt, mask, max_px=200000):
    """Per-image least-squares affine color fit render->gt on masked pixels
    (Zip-NeRF / gsplat cc-metric style). Isolates exposure from geometry."""
    x = rgb[mask].reshape(-1, 3)
    y = rgb_gt[mask].reshape(-1, 3)
    if len(x) > max_px:
        # Uniform row-major coverage is deterministic across repeated scene
        # evaluations and avoids allowing an unseeded metric sampler to decide
        # whether a sealed quality gate passes.
        idx = torch.linspace(
            0, len(x) - 1, max_px, device=x.device
        ).round().long()
        x, y = x[idx], y[idx]
    a = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
    m = torch.linalg.lstsq(a, y).solution
    full = torch.cat([rgb.reshape(-1, 3),
                      torch.ones_like(rgb.reshape(-1, 3)[:, :1])], dim=1)
    return (full @ m).reshape(rgb.shape).clamp(0, 1)


@torch.no_grad()
def _project_scales(params, cfg):
    """Hard bounds after each step: absolute scale range + anisotropy."""
    s = params["scales"]
    s.clamp_(math.log(cfg.train.min_scale_m), math.log(cfg.train.max_scale_m))
    mean = s.mean(dim=1, keepdim=True)
    half = 0.5 * math.log(cfg.train.max_anisotropy)
    s.copy_(torch.minimum(torch.maximum(s, mean - half), mean + half))


def _masked_psnr(rgb, rgb_gt, mask):
    if not mask.any():
        return float("nan")
    mse = ((rgb - rgb_gt) ** 2)[mask].mean()
    return float(-10.0 * torch.log10(mse))


def train_tile(
    seg_dir: Path,
    run_dir: Path,
    cfg,
    device="cuda",
    *,
    allow_failed_georeferencing_for_render: bool = False,
    initial_cloud: str | Path | None = None,
    train_frame_ids=None,
    eval_frame_ids=None,
    tile_plan: dict | None = None,
    context_masks=None,
):
    seed = int(getattr(cfg.train, "seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    segment = SegmentReader(seg_dir).validate()
    frames = segment.frames
    left_camera = segment.calibration["cameras"]["left"]
    intr = np.asarray(left_camera["K"], dtype=np.float64)
    width, height = int(left_camera["width"]), int(left_camera["height"])
    k_mat = torch.tensor([[intr[0, 0], 0, intr[0, 2]],
                          [0, intr[1, 1], intr[1, 2]],
                          [0, 0, 1]], dtype=torch.float32, device=device)
    georeferencing = pose_georeferencing_evidence(seg_dir, cfg)
    require_render_permission(
        georeferencing,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    viewmats_np, _ = load_pose_artifact(
        seg_dir,
        cfg,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    viewmats = torch.tensor(viewmats_np,
                            dtype=torch.float32, device=device)
    c2ws = torch.linalg.inv(viewmats)

    n_pairs = len(viewmats)
    from rtk_splat.core.manifest import load_manifest
    manifest = load_manifest(seg_dir)
    tile_mode = tile_plan is not None
    supplied_selection = train_frame_ids is not None or eval_frame_ids is not None
    if tile_mode != supplied_selection or tile_mode != (context_masks is not None):
        raise ValueError(
            "tile training requires tile_plan, train/eval IDs, and context "
            "masks together"
        )
    train_pairs = (
        list(manifest["train"])
        if train_frame_ids is None
        else [int(value) for value in train_frame_ids]
    )
    eval_ids = (
        list(manifest["val"])
        if eval_frame_ids is None
        else [int(value) for value in eval_frame_ids]
    )
    for label, selected, source in (
        ("train", train_pairs, manifest["train"]),
        ("val", eval_ids, manifest["val"]),
    ):
        if (
            not selected
            or len(selected) != len(set(selected))
            or not set(selected) <= set(int(value) for value in source)
        ):
            raise ValueError(
                f"{label} frame selection must be nonempty unique members of "
                f"manifest.{label}"
            )
    if tile_mode:
        expected = {
            "train_selection_sha256": _selection_sha256(train_pairs),
            "val_selection_sha256": _selection_sha256(eval_ids),
            "n_train": len(train_pairs),
            "n_val": len(eval_ids),
            "pose_fingerprint": pose_fingerprint(viewmats_np),
        }
        if any(tile_plan.get(key) != value for key, value in expected.items()):
            raise ValueError("tile-plan frame or pose binding is inconsistent")
    train_views = [(i, "L") for i in train_pairs]
    if cfg.train.use_right_camera:
        if not segment.meta["capabilities"]["stereo"]:
            raise RuntimeError("right-camera supervision requires a stereo segment")
        train_views += [(i, "R") for i in train_pairs]

    init_cloud = (
        cloud_path(seg_dir, cfg)
        if initial_cloud is None
        else Path(initial_cloud)
    )
    verify_cloud_matches_poses(
        init_cloud, viewmats_np,
        require_fingerprint=pose_artifact_name(cfg) != "rtk",
    )
    cloud_georeferencing_evidence(
        init_cloud,
        georeferencing,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    if tile_mode:
        verify_cloud_tile_binding(init_cloud, tile_plan)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to reuse existing training run: {run_dir}") from exc
    (run_dir / "renders").mkdir()
    georeferencing_text = json.dumps(
        georeferencing, indent=2, sort_keys=True
    ) + "\n"
    (run_dir / "georeferencing.json").write_text(
        georeferencing_text, encoding="utf-8"
    )
    if georeferencing["georeferencing_status"] == "FAILED":
        (run_dir / "GEOREFERENCING_FAILED.json").write_text(
            georeferencing_text, encoding="utf-8"
        )
    params = init_params(init_cloud, cfg, device)
    optimizers = make_optimizers(params, cfg)
    # Annealing: stop the entire third-party MCMC callback (including its
    # otherwise-unconditional noise) at refine_stop_frac, while continuing
    # ordinary gradient descent with the configured means-lr decay.
    refine_stop_iter = int(
        cfg.train.iterations * cfg.train.refine_stop_frac
    )
    strategy = MCMCStrategy(
        cap_max=cfg.train.max_gaussians,
        refine_stop_iter=refine_stop_iter,
        verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=float(cfg.train.means_lr_final_mult) ** (1.0 / cfg.train.iterations))
    color_start = int(cfg.train.iterations * (1.0 - float(cfg.train.color_finetune_frac)))
    if color_start < refine_stop_iter:
        raise RuntimeError("color_finetune_frac overlaps MCMC refinement; "
                           "keep color_finetune_frac <= 1 - refine_stop_frac")

    po = cfg.train.pose_opt
    pose_adj, pose_optim, pose_sched = None, None, None
    if po.enabled:
        pose_adj = PoseAdjust(n_pairs, po.max_trans_m, po.max_rot_deg).to(device)
        # eps=1e-15 is load-bearing: pose gradients through rasterization
        # are tiny and default eps stalls updates (nerfstudio practice)
        pose_optim = torch.optim.Adam(pose_adj.parameters(), lr=float(po.lr),
                                      eps=1e-15,
                                      weight_decay=float(po.weight_decay))
        # upstream schedule: ~100x lr decay over the run
        pose_sched = torch.optim.lr_scheduler.ExponentialLR(
            pose_optim, gamma=0.01 ** (1.0 / cfg.train.iterations))
    eo = cfg.train.exposure_opt
    exp_adj, exp_optim = None, None
    if eo.enabled:
        exp_adj = ExposureAdjust(n_pairs).to(device)
        exp_optim = torch.optim.Adam(exp_adj.parameters(), lr=float(eo.lr),
                                     eps=1e-15,
                                     weight_decay=float(eo.weight_decay))

    config_snapshot, config_snapshot_sha256 = \
        _training_config_snapshot(cfg)
    provenance_path = run_dir / "run_provenance.json"
    provenance = {
        "schema_version": 1,
        "train_seed": seed,
        "iterations": int(cfg.train.iterations),
        "max_gaussians": int(cfg.train.max_gaussians),
        "pose_artifact": pose_artifact_name(cfg),
        "pose_fingerprint": pose_fingerprint(viewmats_np),
        "georeferencing": georeferencing,
        "initial_cloud_sha256": _sha256_file(init_cloud),
        "manifest_sha256": _sha256_file(seg_dir / "manifest.json"),
        "frames_sha256": _sha256_file(seg_dir / "frames.npz"),
        "calibration_sha256": _sha256_file(seg_dir / "calibration.json"),
        "segment_meta_sha256": _sha256_file(seg_dir / "segment_meta.json"),
        "training_image_presentations": int(cfg.train.iterations),
        "available_training_views": len(train_views),
        "all_segment_frames_retain_poses": len(viewmats_np),
        "training_implementation_sha256": _sha256_file(Path(__file__)),
        "effective_training_config": config_snapshot,
        "effective_training_config_sha256": config_snapshot_sha256,
        "configuration": configuration_evidence(cfg),
        "launcher_config_sha256":
            os.environ.get("RTK_SPLAT_CONFIG_SHA256"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms_forced": False,
        "model_selection": {
            "status": "training_in_progress",
            "criterion": _BEST_METRIC_NAME,
            "direction": "maximize",
            "tie_policy": "earliest_evaluation_wins",
            "checkpoint": "best_params.checkpoint.pt",
            "optimizer_state_included": False,
        },
    }
    if tile_mode:
        provenance["tile_plan"] = _jsonable(tile_plan)
    _atomic_write_json(provenance_path, provenance)
    rng = np.random.default_rng(seed)
    order = []
    history = []
    best_metric = None
    best_step = None
    best_observation = None
    best_checkpoint = run_dir / "best_params.checkpoint.pt"
    best_record_path = run_dir / "best_checkpoint.json"
    dilate = int(cfg.train.mask_dilate_px)
    stereo_transform = np.asarray(
        segment.calibration.get("T_right_left", np.eye(4)), dtype=np.float64
    )
    right_from_left = torch.tensor(
        stereo_transform, dtype=torch.float32, device=device
    )

    for step in range(cfg.train.iterations):
        if not order:
            order = [train_views[j] for j in rng.permutation(len(train_views))]
        pair, side = order.pop()
        rgb_gt, depth_gt, valid, sup = _load_frame(
            seg_dir,
            frames,
            pair,
            dilate,
            device,
            side,
            context_mask=(context_masks[pair] if tile_mode else None),
        )

        c2w = c2ws[pair]
        if pose_adj is not None and step >= po.start_iter:
            pid = torch.tensor([pair], device=device)
            c2w = pose_adj(c2w[None], pid)[0]
        if side == "R":
            c2w = right_c2w(c2w, right_from_left)
        out, _, info = render(params, torch.linalg.inv(c2w), k_mat,
                              width, height, cfg.train.sh_degree,
                              cfg.train.rasterize_mode)
        rgb = out[0, ..., :3]
        depth = out[0, ..., 3]
        if exp_adj is not None:
            rgb = exp_adj(rgb, torch.tensor([pair], device=device))

        color_phase = step >= color_start
        if not color_phase:
            strategy.step_pre_backward(params, optimizers, state, step, info)
        sup3 = sup[..., None]
        l1 = (rgb - rgb_gt).abs()[sup3.expand_as(rgb)].mean()
        # masked SSIM done right: SSIM map over the FULL images, averaged
        # inside the mask (multiplying images by the mask first leaks border
        # halos into the loss)
        _, ssim_map = tm_ssim(rgb.permute(2, 0, 1)[None],
                              rgb_gt.permute(2, 0, 1)[None],
                              return_full_image=True)
        ssim_val = ssim_map[0].mean(dim=0)[sup].mean()
        loss = l1 + cfg.train.ssim_lambda * (1.0 - ssim_val)
        if side == "L" and valid.any():
            dl = (torch.log(depth.clamp(min=0.1)[valid])
                  - torch.log(depth_gt[valid])).abs().mean()
            loss = loss + cfg.train.depth_lambda * dl
        loss = loss + cfg.train.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        loss = loss + cfg.train.scale_reg * torch.exp(params["scales"]).mean()
        if pose_adj is not None and step >= po.start_iter:
            pt, pr = pose_adj.penalty()
            loss = loss + float(po.trans_penalty) * pt + float(po.rot_penalty) * pr
        loss.backward()
        # upstream order: optimizers step first, then MCMC relocate/perturb.
        # In the color-only tail, geometry (means/scales/quats) is frozen and
        # only appearance keeps training -- lets positions settle while colors
        # sharpen without positional lr/noise.
        color_params = ("sh0", "shN", "opacities")
        for name, opt in optimizers.items():
            if color_phase and name not in color_params:
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
        for extra in (pose_optim, exp_optim):
            if extra is not None:
                extra.step()
                extra.zero_grad(set_to_none=True)
        if pose_sched is not None:
            pose_sched.step()
        if (
            pose_adj is not None
            and step >= po.start_iter
            and getattr(po, "gauge_projection", True)
        ):
            pose_adj.project_gauge()
        if exp_adj is not None:
            exp_adj.project_gauge()
        if not color_phase:
            _step_mcmc_before_cutoff(
                strategy,
                params,
                optimizers,
                state,
                step,
                info,
                means_sched.get_last_lr()[0],
                refine_stop_iter,
            )
            means_sched.step()
            _project_scales(params, cfg)

        if (step + 1) % cfg.train.eval_every == 0 or step == cfg.train.iterations - 1:
            metrics = evaluate(
                params,
                c2ws,
                k_mat,
                width,
                height,
                seg_dir,
                run_dir,
                eval_ids,
                cfg,
                device,
                step + 1,
                False,
                frames=frames,
                context_masks=context_masks,
            )
            metrics.update(step=step + 1, n_gaussians=int(len(params["means"])),
                           loss=float(loss))
            history.append(metrics)
            candidate = float(metrics[_BEST_METRIC_NAME])
            if _is_better_model(candidate, best_metric):
                _atomic_save_checkpoint(best_checkpoint, params, pose_adj)
                best_metric = candidate
                best_step = step + 1
                best_observation = dict(metrics)
                _atomic_write_json(best_record_path, {
                    "schema_version": 1,
                    "status": "checkpoint_only_training_incomplete",
                    "criterion": _BEST_METRIC_NAME,
                    "direction": "maximize",
                    "tie_policy": "earliest_evaluation_wins",
                    "best_step": best_step,
                    "best_metric": best_metric,
                    "checkpoint": best_checkpoint.name,
                    "optimizer_state_included": False,
                })
            aligned = metrics.get("psnr_masked_aligned")
            print(f"step {step+1}: PSNR full {metrics['psnr']:.2f} "
                  f"masked {metrics['psnr_masked']:.2f} "
                  f"cc {metrics['psnr_masked_cc']:.2f} "
                  f"lpips_cc {metrics['lpips_cc']:.3f} "
                  f"near {metrics['psnr_near']:.2f} "
                  + (f"aligned {aligned:.2f} " if aligned else "")
                  + f"SSIM {metrics['ssim']:.3f} N {metrics['n_gaussians']}",
                  flush=True)

    if best_step is None or best_observation is None \
            or not best_checkpoint.is_file():
        raise RuntimeError(
            f"no non-NaN {_BEST_METRIC_NAME} model checkpoint was selected"
        )

    # The optimizer trajectory remains in `history`. Re-evaluate the restored
    # winner with final-only diagnostics, then append a tagged compatibility
    # record. Its public `step` remains the completed presentation count so
    # existing completion checks still work; `model_step` names the weights.
    params = _load_checkpoint_gaussians(best_checkpoint, device)
    selected = evaluate(
        params,
        c2ws,
        k_mat,
        width,
        height,
        seg_dir,
        run_dir,
        eval_ids,
        cfg,
        device,
        int(cfg.train.iterations),
        True,
        frames=frames,
        context_masks=context_masks,
    )
    selected.update(
        step=int(cfg.train.iterations),
        model_step=int(best_step),
        selected_for_export=True,
        selection_metric_name=_BEST_METRIC_NAME,
        selection_metric_at_selection=float(best_metric),
        n_gaussians=int(len(params["means"])),
        loss=float(best_observation["loss"]),
    )
    history.append(selected)
    export_pruned(
        params,
        run_dir,
        cfg,
        seg_dir,
        georeferencing=georeferencing,
        tile_plan=tile_plan,
    )
    _atomic_write_json(run_dir / "metrics.json", history)

    checkpoint_sha256 = _sha256_file(best_checkpoint)
    selection_record = {
        "schema_version": 1,
        "status": "complete",
        "criterion": _BEST_METRIC_NAME,
        "direction": "maximize",
        "tie_policy": "earliest_evaluation_wins",
        "best_step": int(best_step),
        "best_metric": float(best_metric),
        "completed_training_steps": int(cfg.train.iterations),
        "parameter_artifact": "params.pt",
        "parameter_artifact_sha256": checkpoint_sha256,
        "exported_splat_uses_best_model": True,
        "optimizer_state_included": False,
    }
    provenance["model_selection"] = selection_record
    _atomic_write_json(provenance_path, provenance)
    _atomic_write_json(best_record_path, selection_record)

    # Commit the conventional completion artifact last. Interrupted training
    # retains only the explicitly named checkpoint and cannot pass existing
    # run-completion checks that require params.pt.
    os.replace(best_checkpoint, run_dir / "params.pt")
    print(
        f"restored best model from step {best_step}: "
        f"{_BEST_METRIC_NAME}={best_metric:.4f}; exported after "
        f"{cfg.train.iterations} completed steps",
        flush=True,
    )
    return selected


def export_pruned(
    params,
    run_dir: Path,
    cfg,
    seg_dir: Path = None,
    *,
    georeferencing: dict | None = None,
    tile_plan: dict | None = None,
):
    """Viewer PLY without near-invisible Gaussians, cropped to the supervised
    core region (the periphery is unsupervised mush that ruins free orbit).
    Raw log/logit spaces -- exporter and viewers apply exp/sigmoid."""
    bounds = None
    if tile_plan is None and cfg.train.export_crop and seg_dir is not None:
        cloud = np.load(cloud_path(seg_dir, cfg))["xyz"]
        bounds = (
            np.percentile(cloud, 1, axis=0) - 1.0,
            np.percentile(cloud, 99, axis=0) + 1.0,
        )
    output_name = splat_output_name(georeferencing or {})
    output_path = run_dir / output_name
    tile_keep = None
    if tile_plan is not None:
        tile_keep = _tile_core_keep(params["means"].detach(), tile_plan)
    total, retained = export_splat_tensors(
        params,
        output_path,
        opacity_threshold=float(cfg.train.export_prune_opacity),
        crop_bounds=bounds,
        keep_mask=tile_keep,
    )
    if georeferencing is not None:
        splat_evidence = {
            **georeferencing,
            "splat_file": output_name,
            "splat_sha256": _sha256_file(output_path),
        }
        if tile_plan is not None:
            splat_evidence["tile_plan"] = _jsonable(tile_plan)
        (run_dir / "splat.georeferencing.json").write_text(
            json.dumps(splat_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"exported {retained}/{total} Gaussians "
          f"(pruned opacity <= {cfg.train.export_prune_opacity})")


def export_splat_tensors(
    params,
    output_path: str | Path,
    *,
    opacity_threshold: float,
    crop_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    keep_mask=None,
) -> tuple[int, int]:
    """Write one PLY from Gaussian tensors and return total/retained counts.

    This is the shared, policy-free writer used both by training and by the
    explicit completed-checkpoint export workflow.  It never chooses an output
    name or mutates run metadata.
    """
    threshold = float(opacity_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("opacity_threshold must be finite and in [0, 1)")
    missing = [name for name in _GAUSSIAN_PARAMETER_NAMES if name not in params]
    if missing:
        raise ValueError("Gaussian parameters are incomplete: " + ", ".join(missing))
    total = int(len(params["means"]))
    if total <= 0 or any(len(params[name]) != total for name in _GAUSSIAN_PARAMETER_NAMES):
        raise ValueError("Gaussian tensors must have one common non-zero length")
    opacity = params["opacities"].detach()
    if opacity.numel() != total:
        raise ValueError("opacities must contain exactly one value per Gaussian")
    keep = torch.sigmoid(opacity).reshape(-1) > threshold
    if keep_mask is not None:
        supplied = torch.as_tensor(
            keep_mask, dtype=torch.bool, device=params["means"].device
        ).reshape(-1)
        if supplied.shape != (total,):
            raise ValueError("keep_mask must contain one value per Gaussian")
        keep &= supplied
    if crop_bounds is not None:
        lo_value, hi_value = crop_bounds
        lo = torch.as_tensor(
            lo_value,
            dtype=params["means"].dtype,
            device=params["means"].device,
        )
        hi = torch.as_tensor(
            hi_value,
            dtype=params["means"].dtype,
            device=params["means"].device,
        )
        if lo.shape != (3,) or hi.shape != (3,) or not bool(torch.all(lo < hi)):
            raise ValueError("crop bounds must be finite ordered xyz vectors")
        if not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all()):
            raise ValueError("crop bounds must be finite ordered xyz vectors")
        means = params["means"].detach()
        keep &= ((means > lo) & (means < hi)).all(dim=1)
    retained = int(keep.sum().item())
    if retained <= 0:
        raise ValueError("export policy retained no Gaussians")
    output = Path(output_path)
    if not output.parent.is_dir():
        raise FileNotFoundError(f"export parent does not exist: {output.parent}")
    export_splats(
        params["means"].detach()[keep],
        params["scales"].detach()[keep],
        torch.nn.functional.normalize(params["quats"].detach()[keep], dim=-1),
        opacity[keep],
        params["sh0"].detach()[keep],
        params["shN"].detach()[keep],
        format="ply",
        save_to=str(output),
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"Gaussian exporter did not create a non-empty PLY: {output}")
    return total, retained


def _tile_core_keep(means: torch.Tensor, tile_plan: dict) -> torch.Tensor:
    """Exact half-open core ownership mask encoded by a tile binding."""
    origin = torch.as_tensor(
        tile_plan["partition_origin_enu_m"],
        dtype=means.dtype,
        device=means.device,
    )
    basis = torch.as_tensor(
        tile_plan["R_enu_from_partition"],
        dtype=means.dtype,
        device=means.device,
    )
    core = torch.as_tensor(
        tile_plan["core_bounds_uv_m"],
        dtype=means.dtype,
        device=means.device,
    )
    scene = torch.as_tensor(
        tile_plan["scene_bounds_uv_m"],
        dtype=means.dtype,
        device=means.device,
    )
    if (
        origin.shape != (3,)
        or basis.shape != (3, 3)
        or core.shape != (2, 2)
        or scene.shape != (2, 2)
        or tile_plan.get("boundary_rule")
        != "lower_closed_upper_open_global_max_closed"
    ):
        raise ValueError("tile-plan core export geometry is invalid")
    uv = ((means - origin) @ basis)[:, :2]
    lower = uv >= core[0]
    upper = uv < core[1]
    tolerance = float(tile_plan["ownership_tolerance_m"])
    for axis in (0, 1):
        if float(core[0, axis]) == float(scene[0, axis]):
            lower[:, axis] = uv[:, axis] >= core[0, axis] - tolerance
        if float(core[1, axis]) == float(scene[1, axis]):
            upper[:, axis] = uv[:, axis] <= core[1, axis] + tolerance
    return torch.all(lower & upper, dim=1)


def evaluate(params, c2ws, k_mat, width, height, seg_dir, run_dir,
             eval_ids, cfg, device, step, final=False, *, frames=None,
             context_masks=None, context_mask_dilate_px=None):
    """LEFT-only held-out metrics with RAW given poses (comparable across all
    configurations). On the final call, additionally reports the test-time
    pose-ALIGNED masked PSNR (the honest number when training refined poses;
    exposure stays identity at eval)."""
    full_psnr, m_psnr, near_psnr, ssims, aligned_psnr = [], [], [], [], []
    cc_psnr, brightness, lpips_raw, lpips_cc = [], [], [], []
    dilate = int(cfg.train.mask_dilate_px)
    po = cfg.train.pose_opt
    do_align = final and int(cfg.train.eval_pose_align_steps) > 0
    if do_align:
        frozen = {k: v.detach() for k, v in params.items()}
    if frames is None:
        frames = SegmentReader(seg_dir).frames
    for i in eval_ids:
        evaluation_dilate = (
            dilate
            if context_mask_dilate_px is None
            else int(context_mask_dilate_px)
        )
        if evaluation_dilate <= 0:
            raise ValueError("evaluation mask dilation must be positive")
        rgb_gt, _, _, sup = _load_frame(
            seg_dir,
            frames,
            i,
            evaluation_dilate,
            device,
            context_mask=(context_masks[i] if context_masks is not None else None),
        )
        with torch.no_grad():
            out, _, _ = render(params, torch.linalg.inv(c2ws[i]), k_mat,
                               width, height, cfg.train.sh_degree,
                               cfg.train.rasterize_mode)
            rgb = out[0, ..., :3].clamp(0, 1)
            full_psnr.append(_masked_psnr(rgb, rgb_gt, torch.ones_like(sup)))
            m_psnr.append(_masked_psnr(rgb, rgb_gt, sup))
            near = torch.zeros_like(sup)
            near[height // 2:, :] = True
            near_psnr.append(_masked_psnr(rgb, rgb_gt, near))
            ssims.append(float(tm_ssim(rgb.permute(2, 0, 1)[None],
                                       rgb_gt.permute(2, 0, 1)[None])))
            cc = _color_correct(rgb, rgb_gt, sup)
            cc_psnr.append(_masked_psnr(cc, rgb_gt, sup))
            lpips_raw.append(_masked_lpips(rgb, rgb_gt, sup, device))
            lpips_cc.append(_masked_lpips(cc, rgb_gt, sup, device))
            brightness.append(float(rgb_gt[sup].mean()))
        if do_align:
            def render_fn(viewmat):
                out, _, _ = render(frozen, viewmat, k_mat, width, height,
                                   cfg.train.sh_degree,
                                   cfg.train.rasterize_mode)
                return out[0, ..., :3]
            c2w_al = align_eval_pose(render_fn, c2ws[i], rgb_gt, sup,
                                     po.max_trans_m, po.max_rot_deg,
                                     int(cfg.train.eval_pose_align_steps),
                                     float(cfg.train.eval_pose_align_lr))
            with torch.no_grad():
                out, _, _ = render(frozen, torch.linalg.inv(c2w_al), k_mat,
                                   width, height, cfg.train.sh_degree,
                                   cfg.train.rasterize_mode)
                aligned_psnr.append(_masked_psnr(out[0, ..., :3].clamp(0, 1),
                                                 rgb_gt, sup))
        if i == eval_ids[len(eval_ids) // 2]:
            side = torch.cat([rgb_gt, rgb], dim=1).cpu().numpy()
            cv2.imwrite(str(run_dir / "renders" / f"eval_{step:05d}.jpg"),
                        cv2.cvtColor((side * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))
    out = {"psnr": float(np.mean(full_psnr)),
           "psnr_masked": float(np.mean(m_psnr)),
           "psnr_near": float(np.mean(near_psnr)),
           "psnr_masked_cc": float(np.mean(cc_psnr)),
           "lpips": float(np.mean(lpips_raw)),
           "lpips_cc": float(np.mean(lpips_cc)),
           "ssim": float(np.mean(ssims)), "n_eval": len(eval_ids)}
    if aligned_psnr:
        out["psnr_masked_aligned"] = float(np.mean(aligned_psnr))
    if final:
        # per-frame data for the brightness diagnostic (`diagnose` stage)
        out["per_frame"] = {"eval_ids": eval_ids,
                            "brightness": brightness,
                            "psnr_masked": m_psnr}
    return out
