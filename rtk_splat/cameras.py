"""Camera-side training modules: bounded pose refinement, stereo-pair views,
per-image exposure compensation.

Design (follows gsplat's reference CameraOptModule, nerfstudio camera
optimizers, and the RTK-constrained-refinement idea):
- One learnable SE(3) delta per STEREO PAIR (left+right are rigid), applied on
  the camera-to-world side, zero-initialized.
- Deltas are HARD-BOUNDED via tanh to the configured trust radius of the given
  poses (e.g. RTK jitter: +-3 cm, +-0.5 deg), so refinement can polish
  photometric consistency but can never fight the georeferenced anchor.
- Exposure: per-pair affine color (gain, bias), identity-initialized; applied
  to the *render* before the loss so the map itself stays exposure-neutral.
"""

import torch


def _rodrigues(w: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    axis = w / theta
    k1, k2, k3 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(k1)
    kmat = torch.stack([
        torch.stack([zero, -k3, k2], dim=-1),
        torch.stack([k3, zero, -k1], dim=-1),
        torch.stack([-k2, k1, zero], dim=-1)], dim=-2)
    eye = torch.eye(3, device=w.device).expand(*w.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return eye + s * kmat + (1 - c) * (kmat @ kmat)


class PoseAdjust(torch.nn.Module):
    """Bounded per-pair SE(3) refinement on camera-to-world matrices."""

    def __init__(self, n_pairs: int, max_trans_m: float, max_rot_deg: float):
        super().__init__()
        self.raw = torch.nn.Embedding(n_pairs, 6)
        torch.nn.init.zeros_(self.raw.weight)
        self.max_t = float(max_trans_m)
        self.max_r = float(torch.deg2rad(torch.tensor(max_rot_deg)))

    def forward(self, c2w: torch.Tensor, pair_ids: torch.Tensor) -> torch.Tensor:
        d = self.raw(pair_ids)
        dx = self.max_t * torch.tanh(d[..., :3] / self.max_t)
        dw = self.max_r * torch.tanh(d[..., 3:] / self.max_r)
        transform = torch.eye(4, device=d.device).repeat(*d.shape[:-1], 1, 1)
        transform[..., :3, :3] = _rodrigues(dw)
        transform[..., :3, 3] = dx
        return c2w @ transform

    def penalty(self):
        """Mean bounded-delta norms (nerfstudio-style pull toward zero)."""
        d = self.raw.weight
        dx = self.max_t * torch.tanh(d[:, :3] / self.max_t)
        dw = self.max_r * torch.tanh(d[:, 3:] / self.max_r)
        return dx.norm(dim=-1).mean(), dw.norm(dim=-1).mean()

    @torch.no_grad()
    def project_gauge(self):
        """Remove the mean delta: per-view corrections stay, the global mode
        (which would silently migrate the whole map) is projected out."""
        self.raw.weight -= self.raw.weight.mean(dim=0, keepdim=True)


class ExposureAdjust(torch.nn.Module):
    """Per-pair affine color: render' = render * (1 + gain) + bias."""

    def __init__(self, n_pairs: int):
        super().__init__()
        self.raw = torch.nn.Embedding(n_pairs, 6)
        torch.nn.init.zeros_(self.raw.weight)

    def forward(self, rgb: torch.Tensor, pair_id: torch.Tensor) -> torch.Tensor:
        d = self.raw(pair_id)
        return rgb * (1.0 + d[..., :3]) + d[..., 3:]

    @torch.no_grad()
    def project_gauge(self):
        """Remove the mean gain/bias: a global exposure shift is unobservable
        and would tint the map (seen as the purple road); only per-image
        deviations from the mean survive."""
        self.raw.weight -= self.raw.weight.mean(dim=0, keepdim=True)


def right_c2w(c2w_left: torch.Tensor, baseline_m: float) -> torch.Tensor:
    """Rectified right eye sits at +baseline along the left OPTICAL x axis."""
    shift = torch.eye(4, device=c2w_left.device, dtype=c2w_left.dtype)
    shift[0, 3] = baseline_m
    return c2w_left @ shift


@torch.no_grad()
def _viewmat(c2w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w)


def align_eval_pose(render_fn, c2w0: torch.Tensor, rgb_gt: torch.Tensor,
                    sup: torch.Tensor, max_trans_m: float, max_rot_deg: float,
                    steps: int, lr: float) -> torch.Tensor:
    """Test-time pose alignment for held-out views (BARF-style protocol):
    optimize ONLY this view's bounded delta against the photometric loss,
    with the map frozen. Returns the aligned camera-to-world matrix."""
    adj = PoseAdjust(1, max_trans_m, max_rot_deg).to(c2w0.device)
    opt = torch.optim.Adam(adj.parameters(), lr=lr)
    ids = torch.zeros(1, dtype=torch.long, device=c2w0.device)
    sup3 = sup[..., None]
    for _ in range(steps):
        c2w = adj(c2w0[None], ids)[0]
        rgb = render_fn(torch.linalg.inv(c2w))
        loss = (rgb - rgb_gt).abs()[sup3.expand_as(rgb)].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return adj(c2w0[None], ids)[0]
