"""Metric alignment and held-out RTK quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

from rtk_splat.backends.mapper_config import MapperConfig
from rtk_splat.frontends.artifact import ArtifactError

@dataclass(frozen=True)
class AlignmentResult:
    """Robust fixed-scale transform plus a diagnostic-only similarity scale."""

    rotation: np.ndarray
    translation: np.ndarray
    residuals_m: np.ndarray
    inlier_mask: np.ndarray
    thresholds_m: np.ndarray
    sim3_scale_diagnostic: float
    source_rank: int


@dataclass(frozen=True)
class TemporalAlignmentResult:
    """Calibration-only transform with untouched temporal holdout residuals."""

    alignment: AlignmentResult
    temporal_block_ids: np.ndarray
    calibration_block_ids: tuple[int, ...]
    holdout_block_ids: tuple[int, ...]
    calibration_mask: np.ndarray
    holdout_mask: np.ndarray
    residual_vectors_m: np.ndarray
    residuals_m: np.ndarray
    thresholds_m: np.ndarray
    calibration_inlier_mask: np.ndarray
    holdout_inlier_mask: np.ndarray


@dataclass(frozen=True)
class TemporalBlockSplit:
    """Deterministic alternating calibration/holdout time blocks."""

    block_ids: np.ndarray
    calibration_block_ids: tuple[int, ...]
    holdout_block_ids: tuple[int, ...]
    calibration_mask: np.ndarray
    holdout_mask: np.ndarray


def temporal_block_split(
    timestamps_ns: np.ndarray,
    *,
    temporal_blocks: int = 5,
) -> TemporalBlockSplit:
    """Assign strictly ordered timestamps without inspecting pose residuals.

    This is shared by post-solve evaluation and RTK-constrained refinement so
    an optimizer can never consume a prior later described as held out.
    """
    if (
        isinstance(temporal_blocks, bool)
        or not isinstance(temporal_blocks, int)
        or temporal_blocks < 3
    ):
        raise ValueError("temporal_blocks must be an integer of at least 3")
    timestamps = np.asarray(timestamps_ns)
    if (
        timestamps.ndim != 1
        or not len(timestamps)
        or timestamps.dtype.kind not in "iuf"
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError(
            "alignment timestamps must be finite, one-dimensional, and "
            "strictly increasing"
        )
    block_count = min(temporal_blocks, len(timestamps))
    block_ids = (
        np.arange(len(timestamps), dtype=np.int64) * block_count
        // len(timestamps)
    )
    unique_blocks = tuple(int(value) for value in np.unique(block_ids))
    holdout_blocks = tuple(value for value in unique_blocks if value % 2 == 1)
    calibration_blocks = tuple(
        value for value in unique_blocks if value % 2 == 0
    )
    calibration_mask = np.isin(block_ids, calibration_blocks)
    holdout_mask = np.isin(block_ids, holdout_blocks)
    if calibration_mask.sum() < 3 or not holdout_mask.any():
        raise ArtifactError(
            "temporal holdout requires at least three calibration priors and "
            "one untouched holdout prior"
        )
    return TemporalBlockSplit(
        block_ids=block_ids,
        calibration_block_ids=calibration_blocks,
        holdout_block_ids=holdout_blocks,
        calibration_mask=calibration_mask,
        holdout_mask=holdout_mask,
    )


def _alignment_inputs(
    source: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape != source.shape
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("alignment points must have finite matching shape (N, 3)")
    covariance_supplied = covariance is not None
    if covariance is None:
        covariance = np.repeat(np.eye(3)[None], len(source), axis=0)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (len(source), 3, 3) or not np.isfinite(covariance).all():
        raise ValueError("alignment covariance must have finite shape (N, 3, 3)")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues < -1e-10):
        raise ValueError("alignment covariance must be positive semidefinite")
    sigma = (
        np.sqrt(np.maximum(eigenvalues[:, -1], 1e-8))
        if covariance_supplied
        else np.zeros(len(source), dtype=np.float64)
    )
    weights = 1.0 / np.maximum(np.trace(covariance, axis1=1, axis2=2), 1e-8)
    weights /= weights.mean()
    return source, target, covariance, np.column_stack((weights, sigma))


def _weighted_rigid(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if weights.shape != (len(source),) or not np.isfinite(total) or total <= 0:
        raise ValueError("alignment weights are invalid")
    source_mean = np.sum(source * weights[:, None], axis=0) / total
    target_mean = np.sum(target * weights[:, None], axis=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    rank = int(np.linalg.matrix_rank(source_centered * np.sqrt(weights[:, None])))
    if rank < 2:
        raise ArtifactError("camera-centre priors do not observe a 3-D rigid alignment")
    cross = (target_centered * weights[:, None]).T @ source_centered
    u, _, vt = np.linalg.svd(cross)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = target_mean - rotation @ source_mean
    return rotation, translation, rank


def _similarity_scale(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    rotation: np.ndarray,
) -> float:
    total = float(weights.sum())
    source_mean = np.sum(source * weights[:, None], axis=0) / total
    target_mean = np.sum(target * weights[:, None], axis=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    rotated = source_centered @ rotation.T
    numerator = float(np.sum(weights * np.sum(target_centered * rotated, axis=1)))
    denominator = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    if denominator <= 1e-12:
        raise ArtifactError("similarity scale is unobservable")
    scale = numerator / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ArtifactError("similarity scale diagnostic is invalid")
    return scale


def estimate_rigid_alignment(
    source_centers_m: np.ndarray,
    target_centers_m: np.ndarray,
    covariance_m2: np.ndarray | None = None,
    *,
    ransac_threshold_m: float = 0.15,
    ransac_iterations: int = 512,
    random_seed: int = 7,
) -> AlignmentResult:
    """Estimate robust target-from-source SE(3), never applying Sim(3) scale."""
    if not np.isfinite(ransac_threshold_m) or ransac_threshold_m <= 0:
        raise ValueError("ransac_threshold_m must be finite and positive")
    if (
        isinstance(ransac_iterations, bool)
        or not isinstance(ransac_iterations, int)
        or ransac_iterations <= 0
    ):
        raise ValueError("ransac_iterations must be a positive integer")
    source, target, _, weight_sigma = _alignment_inputs(
        source_centers_m, target_centers_m, covariance_m2
    )
    weights = weight_sigma[:, 0]
    thresholds = np.maximum(ransac_threshold_m, 3.0 * weight_sigma[:, 1])
    rng = np.random.default_rng(random_seed)
    samples = [np.arange(len(source), dtype=np.int64)]
    samples.extend(
        rng.choice(len(source), size=3, replace=False)
        for _ in range(ransac_iterations)
    )
    best: tuple[tuple[int, float, float], np.ndarray] | None = None
    for sample in samples:
        try:
            rotation, translation, _ = _weighted_rigid(
                source[sample], target[sample], weights[sample]
            )
        except ArtifactError:
            continue
        residuals = np.linalg.norm(
            source @ rotation.T + translation - target, axis=1
        )
        inliers = residuals <= thresholds
        if inliers.sum() < 3:
            continue
        normalized = residuals[inliers] / thresholds[inliers]
        score = (
            int(inliers.sum()),
            -float(np.median(normalized)),
            -float(np.mean(normalized)),
        )
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        raise ArtifactError("no observable rigid alignment hypothesis")
    inliers = best[1]
    for _ in range(3):
        rotation, translation, rank = _weighted_rigid(
            source[inliers], target[inliers], weights[inliers]
        )
        residuals = np.linalg.norm(
            source @ rotation.T + translation - target, axis=1
        )
        updated = residuals <= thresholds
        if updated.sum() < 3 or np.array_equal(updated, inliers):
            break
        inliers = updated
    rotation, translation, rank = _weighted_rigid(
        source[inliers], target[inliers], weights[inliers]
    )
    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    scale = _similarity_scale(
        source[inliers], target[inliers], weights[inliers], rotation
    )
    return AlignmentResult(
        rotation=rotation,
        translation=translation,
        residuals_m=residuals,
        inlier_mask=inliers,
        thresholds_m=thresholds,
        sim3_scale_diagnostic=scale,
        source_rank=rank,
    )


def estimate_temporal_heldout_alignment(
    source_centers_m: np.ndarray,
    target_centers_m: np.ndarray,
    timestamps_ns: np.ndarray,
    covariance_m2: np.ndarray | None = None,
    *,
    temporal_blocks: int = 5,
    ransac_threshold_m: float = 0.15,
    ransac_iterations: int = 512,
    random_seed: int = 7,
) -> TemporalAlignmentResult:
    """Fit on alternating contiguous time blocks and evaluate the rest.

    Membership depends only on sorted timestamp order, never on pose residuals.
    Odd-numbered blocks are sealed holdouts. Their target positions are not
    passed to the SE(3) or diagnostic Sim(3) estimators.
    """
    covariance_supplied = covariance_m2 is not None
    source, target, covariance, weight_sigma = _alignment_inputs(
        source_centers_m, target_centers_m, covariance_m2
    )
    timestamps = np.asarray(timestamps_ns)
    if timestamps.shape != (len(source),):
        raise ValueError("alignment timestamps must match the pose count")
    split = temporal_block_split(
        timestamps, temporal_blocks=temporal_blocks
    )
    calibration_mask = split.calibration_mask
    holdout_mask = split.holdout_mask

    alignment = estimate_rigid_alignment(
        source[calibration_mask],
        target[calibration_mask],
        covariance[calibration_mask] if covariance_supplied else None,
        ransac_threshold_m=ransac_threshold_m,
        ransac_iterations=ransac_iterations,
        random_seed=random_seed,
    )
    residual_vectors = (
        source @ alignment.rotation.T + alignment.translation - target
    )
    residuals = np.linalg.norm(residual_vectors, axis=1)
    thresholds = np.maximum(ransac_threshold_m, 3.0 * weight_sigma[:, 1])
    calibration_inliers = np.zeros(len(source), dtype=bool)
    calibration_inliers[calibration_mask] = alignment.inlier_mask
    holdout_inliers = holdout_mask & (residuals <= thresholds)
    return TemporalAlignmentResult(
        alignment=alignment,
        temporal_block_ids=split.block_ids,
        calibration_block_ids=split.calibration_block_ids,
        holdout_block_ids=split.holdout_block_ids,
        calibration_mask=calibration_mask,
        holdout_mask=holdout_mask,
        residual_vectors_m=residual_vectors,
        residuals_m=residuals,
        thresholds_m=thresholds,
        calibration_inlier_mask=calibration_inliers,
        holdout_inlier_mask=holdout_inliers,
    )


def squared_mahalanobis_residuals(
    residual_vectors_m: np.ndarray,
    covariance_m2: np.ndarray,
) -> np.ndarray:
    """Return ``r.T @ covariance^-1 @ r`` for every 3-D residual.

    Stored covariance is interpreted in the residual/target coordinate frame.
    A strictly positive-definite covariance is required: silently adding an
    arbitrary numerical variance would turn the receiver's declared accuracy
    into an undocumented gate parameter.
    """
    residuals = np.asarray(residual_vectors_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if (
        residuals.ndim != 2
        or residuals.shape[1:] != (3,)
        or not len(residuals)
        or not np.isfinite(residuals).all()
        or covariance.shape != (len(residuals), 3, 3)
        or not np.isfinite(covariance).all()
    ):
        raise ValueError(
            "RTK residuals/covariances must have finite shapes (N, 3) and "
            "(N, 3, 3)"
        )
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    try:
        factors = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factors, residuals[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ArtifactError(
            "RTK covariance must be strictly positive definite for "
            "covariance-normalized evaluation"
        ) from exc
    values = np.einsum("ni,ni->n", whitened, whitened)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ArtifactError("covariance-normalized RTK residuals are invalid")
    return values


def rtk_residual_quality(
    residual_vectors_m: np.ndarray,
    covariance_m2: np.ndarray,
    euclidean_inlier_mask: np.ndarray,
    *,
    config: MapperConfig = MapperConfig(),
) -> dict[str, Any]:
    """Evaluate held-out RTK residuals in metres and covariance units.

    The chi-square checks use three degrees of freedom because each held-out
    observation contributes one 3-D camera-centre residual.  They do not use
    six SE(3) degrees of freedom: the fitted transform is calibrated on
    different temporal blocks and is not the observation being tested here.
    Absolute metre caps remain independent so a receiver cannot make an
    arbitrarily inaccurate result pass merely by declaring large covariance.
    """
    residual_vectors = np.asarray(residual_vectors_m, dtype=np.float64)
    inlier_mask = np.asarray(euclidean_inlier_mask)
    if (
        inlier_mask.shape != (len(residual_vectors),)
        or inlier_mask.dtype.kind != "b"
    ):
        raise ValueError("euclidean_inlier_mask must be a boolean vector")
    squared_mahalanobis = squared_mahalanobis_residuals(
        residual_vectors, covariance_m2
    )
    residuals_m = np.linalg.norm(residual_vectors, axis=1)
    euclidean_inlier_residuals = residuals_m[inlier_mask]
    residual_median = float(np.median(residuals_m))
    residual_p95 = float(np.percentile(residuals_m, 95))
    inlier_p95 = (
        float(np.percentile(euclidean_inlier_residuals, 95))
        if euclidean_inlier_residuals.size
        else None
    )
    inlier_maximum = (
        float(euclidean_inlier_residuals.max())
        if euclidean_inlier_residuals.size
        else None
    )
    euclidean_inlier_fraction = float(inlier_mask.mean())

    chi2_threshold = float(
        chi2.ppf(config.rtk_chi2_inlier_probability, df=3)
    )
    chi2_inlier_mask = squared_mahalanobis <= chi2_threshold
    chi2_inlier_fraction = float(chi2_inlier_mask.mean())
    median_mahalanobis_sq = float(np.median(squared_mahalanobis))
    expected_median_mahalanobis_sq = float(chi2.ppf(0.5, df=3))
    normalized_authoritative = config.rtk_covariance_gate_mode == "enforce"

    checks = {
        "median_holdout_rtk_residual_m": {
            "kind": "independent_absolute_cap",
            "authoritative": True,
            "value": residual_median,
            "maximum": config.max_rtk_median_error_m,
            "passed": residual_median <= config.max_rtk_median_error_m,
        },
        "p95_holdout_inlier_rtk_residual_m": {
            "kind": "independent_absolute_cap",
            "authoritative": True,
            "value": inlier_p95,
            "maximum": config.max_rtk_p95_inlier_error_m,
            "passed": bool(
                inlier_p95 is not None
                and inlier_p95 <= config.max_rtk_p95_inlier_error_m
            ),
        },
        "holdout_rtk_inlier_fraction": {
            "kind": "robust_absolute_support",
            "authoritative": True,
            "value": euclidean_inlier_fraction,
            "minimum": config.min_rtk_inlier_fraction,
            "passed": euclidean_inlier_fraction
            >= config.min_rtk_inlier_fraction,
        },
        "median_holdout_rtk_mahalanobis_sq": {
            "kind": "covariance_normalized_aggregate",
            "authoritative": normalized_authoritative,
            "value": median_mahalanobis_sq,
            "expected_chi2_3_median": expected_median_mahalanobis_sq,
            "maximum": config.max_rtk_median_mahalanobis_sq,
            "maximum_chi2_3_cdf": float(
                chi2.cdf(config.max_rtk_median_mahalanobis_sq, df=3)
            ),
            "passed": median_mahalanobis_sq
            <= config.max_rtk_median_mahalanobis_sq,
        },
        "holdout_rtk_chi2_inlier_fraction": {
            "kind": "covariance_normalized_coverage",
            "authoritative": normalized_authoritative,
            "value": chi2_inlier_fraction,
            "minimum": config.min_rtk_chi2_inlier_fraction,
            "point_probability": config.rtk_chi2_inlier_probability,
            "maximum_mahalanobis_sq": chi2_threshold,
            "degrees_of_freedom": 3,
            "passed": chi2_inlier_fraction
            >= config.min_rtk_chi2_inlier_fraction,
        },
    }
    return {
        "schema_version": 1,
        "covariance_model": (
            "stored_per_observation_camera_centre_position_covariance_only"
        ),
        "covariance_limitations": [
            (
                "gate authority must be configured from upstream covariance "
                "provenance"
            ),
            (
                "stored covariance may omit visual-pose and alignment-fit "
                "uncertainty"
            ),
            (
                "heading-by-lever-arm and timing uncertainty must be "
                "propagated upstream"
            ),
        ],
        "covariance_gate_mode": config.rtk_covariance_gate_mode,
        "degrees_of_freedom": 3,
        "n_residuals": len(residuals_m),
        "residual_m": {
            "median": residual_median,
            "p95": residual_p95,
            "p95_euclidean_inliers": inlier_p95,
            "maximum_euclidean_inliers": inlier_maximum,
        },
        "mahalanobis_sq": {
            "median": median_mahalanobis_sq,
            "expected_chi2_3_median": expected_median_mahalanobis_sq,
            "p95": float(np.percentile(squared_mahalanobis, 95)),
            "maximum": float(squared_mahalanobis.max()),
            "values": squared_mahalanobis.tolist(),
        },
        "euclidean_inlier_mask": inlier_mask.tolist(),
        "chi2_inlier_mask": chi2_inlier_mask.tolist(),
        "checks": checks,
        "passed": all(
            check["passed"]
            for check in checks.values()
            if check["authoritative"]
        ),
    }


def quality_summary(
    expected_image_names: Sequence[str],
    registered_image_names: Sequence[str],
    model_stats: Mapping[str, int | float],
    *,
    max_reprojection_error_px: float = 2.0,
    min_mean_track_length: float = 2.0,
) -> dict[str, Any]:
    """Build explicit registration and geometric quality gates."""
    expected = set(expected_image_names)
    registered = set(registered_image_names)
    missing = sorted(expected - registered)
    unexpected = sorted(registered - expected)
    count_consistent = int(model_stats["registered_images"]) == len(registered)
    checks = {
        "all_expected_images_registered": {
            "value": len(missing),
            "maximum": 0,
            "passed": not missing,
        },
        "no_unexpected_images": {
            "value": len(unexpected),
            "maximum": 0,
            "passed": not unexpected,
        },
        "analyzer_count_consistent": {
            "value": int(model_stats["registered_images"]),
            "expected": len(registered),
            "passed": count_consistent,
        },
        "mean_reprojection_error_px": {
            "value": float(model_stats["mean_reprojection_error_px"]),
            "maximum": float(max_reprojection_error_px),
            "passed": float(model_stats["mean_reprojection_error_px"])
            <= max_reprojection_error_px,
        },
        "mean_track_length": {
            "value": float(model_stats["mean_track_length"]),
            "minimum": float(min_mean_track_length),
            "passed": float(model_stats["mean_track_length"])
            >= min_mean_track_length,
        },
    }
    return {
        "schema_version": 1,
        "n_expected_images": len(expected),
        "n_registered_images": len(registered),
        "registration_fraction": len(expected & registered) / len(expected),
        "missing_images": missing,
        "unexpected_images": unexpected,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
