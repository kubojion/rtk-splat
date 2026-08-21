"""Sealed, held-out evaluation of one deterministic TilePlan seam.

The probe consumes two completed tile runs that remain bound to their original
immutable TilePlan.  It chooses the neighbor from core geometry alone, renders
the hard half-open ownership cut, and compares that cut with the arithmetic
mean quality of the two independently rendered context models on the exact
same held-out seam pixels.  Held-out evidence is evaluation-only: it never
selects a neighbor, moves a boundary, changes a model, or changes a threshold.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from rtk_splat.core.pose_artifacts import load_pose_artifact, pose_fingerprint
from rtk_splat.core.segment import SegmentReader, publish_directory_noreplace
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    collect_package_state,
    sha256_file,
)
from rtk_splat.workflows.tile_scene import (
    _METRICS,
    _completed_tile_run,
    _concatenate_core_params,
    _evaluate_combined,
    _expected_tile_binding,
    _training_identity,
    _verify_training_source_binding,
    select_geometric_tile_neighbor,
    _build_seam_masks,
)
from rtk_splat.workflows.tiles import verify_tile_plan


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "rtk_splat_tile_seam_probe"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _safe_name(value: str) -> str:
    value = str(value)
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid seam-probe name: {value!r}")
    return value


def _json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(value, allow_nan=False)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid JSON artifact: {path}") from exc
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"seam-probe file is missing or unsafe: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(root).as_posix()] = _record(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "files": files,
    }


def _finite_metrics(record: Any) -> bool:
    return isinstance(record, dict) and all(
        isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
        and math.isfinite(float(record[key]))
        for key in _METRICS
    )


def _recomputed_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    evidence = metrics.get("evidence", {})
    candidate = metrics.get("candidate")
    reference = metrics.get("component_mean_reference")
    components = metrics.get("components")
    regressions = metrics.get("regressions", {})
    gates = metrics.get("gates", {})
    components_finite = (
        isinstance(components, dict)
        and len(components) == 2
        and all(_finite_metrics(value) for value in components.values())
    )
    try:
        psnr_loss = float(regressions["psnr_masked_loss_db"])
        corrected_loss = float(regressions["psnr_masked_cc_loss_db"])
        ssim_loss = float(regressions["ssim_loss"])
        lpips_increase = float(regressions["lpips_cc_increase"])
        maximum_psnr_loss = float(gates["maximum_psnr_loss_db"])
        maximum_ssim_loss = float(gates["maximum_ssim_loss"])
        maximum_lpips_increase = float(gates["maximum_lpips_cc_increase"])
    except (KeyError, TypeError, ValueError):
        return {}
    finite_regressions = all(
        math.isfinite(value)
        for value in (psnr_loss, corrected_loss, ssim_loss, lpips_increase)
    )
    gates_valid = all(
        math.isfinite(value) and value >= 0
        for value in (
            maximum_psnr_loss,
            maximum_ssim_loss,
            maximum_lpips_increase,
        )
    )
    reference_matches = bool(
        components_finite
        and _finite_metrics(reference)
        and all(
            math.isclose(
                float(reference[key]),
                sum(float(record[key]) for record in components.values()) / 2.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in _METRICS
        )
    )
    regressions_match = bool(
        finite_regressions
        and _finite_metrics(candidate)
        and _finite_metrics(reference)
        and math.isclose(
            psnr_loss,
            float(reference["psnr_masked"] - candidate["psnr_masked"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            corrected_loss,
            float(reference["psnr_masked_cc"] - candidate["psnr_masked_cc"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            ssim_loss,
            float(reference["ssim"] - candidate["ssim"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            lpips_increase,
            float(candidate["lpips_cc"] - reference["lpips_cc"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    selected_segment = metrics.get("selection", {}).get("selected_segment_uv")
    evidence_segments = evidence.get("internal_segments_uv")
    return {
        "two_edge_adjacent_tiles_selected_without_heldout": (
            metrics.get("selection", {}).get("uses_heldout_evidence") is False
            and isinstance(selected_segment, list)
            and len(selected_segment) == 4
            and evidence_segments == [selected_segment]
        ),
        "seam_metric_depth_evidence_is_sufficient": (
            isinstance(evidence, dict)
            and int(evidence.get("n_validation_frames", 0)) >= 2
            and int(evidence.get("metric_depth_pixels_in_band", 0)) >= 512
        ),
        "component_metrics_are_finite": components_finite,
        "component_mean_reference_is_finite": _finite_metrics(reference),
        "component_mean_reference_matches_components": reference_matches,
        "hard_ownership_metrics_are_finite": _finite_metrics(candidate),
        "regressions_are_finite": finite_regressions,
        "regressions_match_recorded_metrics": regressions_match,
        "regression_gates_are_valid": gates_valid,
        "seam_masked_psnr_loss_within_gate": (
            finite_regressions and gates_valid and psnr_loss <= maximum_psnr_loss
        ),
        "seam_corrected_masked_psnr_loss_within_gate": (
            finite_regressions and gates_valid and corrected_loss <= maximum_psnr_loss
        ),
        "seam_ssim_loss_within_gate": (
            finite_regressions and gates_valid and ssim_loss <= maximum_ssim_loss
        ),
        "seam_lpips_cc_increase_within_gate": (
            finite_regressions
            and gates_valid
            and lpips_increase <= maximum_lpips_increase
        ),
    }


def verify_tile_seam_probe(root: str | Path) -> dict[str, Any]:
    """Verify the terminal seal and semantic claims of a seam probe."""
    supplied = Path(root).expanduser()
    if supplied.is_symlink():
        raise ArtifactError(f"tile seam probe cannot be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ArtifactError(f"tile seam probe is missing or unsafe: {root}")
    if _json(root / "manifest.json") != _manifest(root):
        raise ArtifactError("tile-seam terminal seal verification failed")
    probe = _json(root / "probe.json")
    quality = _json(root / "quality.json")
    metrics = _json(root / "metrics.json")
    provenance = _json(root / "provenance.json")
    records = (probe, quality, metrics, provenance)
    if any(
        not isinstance(record, dict)
        or record.get("schema_version") != SCHEMA_VERSION
        for record in records
    ):
        raise ArtifactError("tile-seam schema is invalid")
    name = str(probe.get("name", ""))
    valid_directory = root.name == name or root.name.startswith(
        f".{name}.writing-"
    )
    checks = quality.get("checks")
    recomputed = _recomputed_checks(metrics)
    selection = metrics.get("selection", {})
    tile_ids = probe.get("tile_ids")
    tile_run_records = provenance.get("tile_runs")
    run_ids = (
        [item.get("tile_id") for item in tile_run_records]
        if isinstance(tile_run_records, list)
        and all(isinstance(item, dict) for item in tile_run_records)
        else []
    )
    georeferencing = provenance.get("georeferencing", {})
    eligible = bool(
        georeferencing.get("artifact_class") == "production"
        and georeferencing.get("georeferencing_status") == "PASSED"
        and georeferencing.get("metric_georeferencing_claim_eligible") is True
    )
    if (
        probe.get("artifact_type") != ARTIFACT_TYPE
        or not _SAFE_NAME.fullmatch(name)
        or not valid_directory
        or not isinstance(tile_ids, list)
        or len(tile_ids) != 2
        or len(set(tile_ids)) != 2
        or run_ids != tile_ids
        or set(metrics.get("components", {})) != set(tile_ids)
        or selection.get("anchor_tile_id") != tile_ids[0]
        or selection.get("selected_tile_id") != tile_ids[1]
        or provenance.get("neighbor_selection") != selection
        or provenance.get("tile_plan") != probe.get("tile_plan")
        or metrics.get("evaluation", {}).get("eval_ids")
        != [
            item.get("frame_id")
            for item in metrics.get("evidence", {}).get("frames", [])
        ]
        or probe.get("metric_georeferencing_claim_eligible") is not eligible
        or probe.get("provisional") is not (not eligible)
        or provenance.get("diagnostic_nonproduction_georeferencing") is not (not eligible)
        or not isinstance(checks, dict)
        or checks != recomputed
        or quality.get("passed") is not all(checks.values())
        or probe.get("quality_passed") is not quality.get("passed")
        or metrics.get("evaluation", {}).get("aggregation")
        != "hard_half_open_core_ownership_on_shared_heldout_seam"
    ):
        raise ArtifactError("tile-seam status or quality claims disagree")
    for record in tile_run_records:
        for key in (
            "run_provenance_sha256",
            "best_checkpoint_sha256",
            "params_sha256",
            "source_splat_sha256",
        ):
            if not _SHA256.fullmatch(str(record.get(key, ""))):
                raise ArtifactError("tile-seam source hash is invalid")
    return probe


def publish_tile_seam_probe(
    *,
    segment: str | Path,
    cfg: Any,
    tile_plan_root: str | Path,
    pose_root: str | Path,
    tile_runs: Mapping[str, str | Path],
    anchor_tile_id: str,
    output_root: str | Path,
    probe_name: str,
    maximum_psnr_loss_db: float = 0.30,
    maximum_ssim_loss: float = 0.015,
    maximum_lpips_cc_increase: float = 0.030,
    maximum_visible_gaussians: int | None = None,
    device: str = "cuda",
    allow_nonproduction_georeferencing_for_diagnostic: bool = False,
) -> dict[str, Any]:
    """Evaluate and atomically publish one deterministic adjacent-tile seam."""
    name = _safe_name(probe_name)
    maximum_psnr_loss = float(maximum_psnr_loss_db)
    maximum_ssim_loss = float(maximum_ssim_loss)
    maximum_lpips_increase = float(maximum_lpips_cc_increase)
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            maximum_psnr_loss,
            maximum_ssim_loss,
            maximum_lpips_increase,
        )
    ):
        raise ValueError("seam regression gates must be finite and non-negative")
    if maximum_visible_gaussians is None:
        configured = getattr(getattr(cfg, "train", None), "max_gaussians", None)
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise ValueError("seam probe needs an explicit Gaussian visibility cap")
        maximum_visible_gaussians = configured
    if maximum_visible_gaussians <= 0:
        raise ValueError("maximum_visible_gaussians must be positive")

    destination = (
        Path(output_root).expanduser() / "seam_probe_artifacts" / name
    )
    if destination.exists():
        raise FileExistsError(f"refusing to modify existing seam probe: {destination}")

    reader = SegmentReader(segment).validate()
    plan_root = Path(tile_plan_root).expanduser().resolve()
    pose_root = Path(pose_root).expanduser().resolve()
    plan = verify_tile_plan(
        plan_root,
        segment=reader.root,
        pose_root=pose_root,
        rehash_sources=True,
    )
    inventory = _json(plan_root / "source_inventory.json")
    plan_provenance = _json(plan_root / "provenance.json")
    georeferencing = plan_provenance.get("georeferencing")
    if not isinstance(georeferencing, dict):
        raise ArtifactError("TilePlan has no georeferencing evidence")
    production_georeferencing = bool(
        plan.get("metric_georeferencing_claim_eligible") is True
        and georeferencing.get("artifact_class") == "production"
        and georeferencing.get("georeferencing_status") == "PASSED"
        and georeferencing.get("metric_georeferencing_claim_eligible") is True
    )
    diagnostic_nonproduction = not production_georeferencing
    if diagnostic_nonproduction and not bool(
        allow_nonproduction_georeferencing_for_diagnostic
    ):
        raise ArtifactError(
            "non-production georeferencing requires an explicit diagnostic seam probe"
        )

    selection = select_geometric_tile_neighbor(plan, str(anchor_tile_id))
    tile_ids = [
        str(selection["anchor_tile_id"]),
        str(selection["selected_tile_id"]),
    ]
    if set(tile_runs) != set(tile_ids) or len(tile_runs) != 2:
        raise ArtifactError(
            "tile-run mapping must contain exactly the anchor and geometry-selected neighbor"
        )
    plan_tiles = list(plan["tiles"])
    indices = {str(tile["tile_id"]): index for index, tile in enumerate(plan_tiles)}
    tiles = [plan_tiles[indices[tile_id]] for tile_id in tile_ids]
    validation_ids = sorted(
        set(int(value) for value in tiles[0]["frame_ids"]["val"])
        & set(int(value) for value in tiles[1]["frame_ids"]["val"])
    )
    if len(validation_ids) < 2:
        raise ArtifactError("adjacent tiles share insufficient sealed validation views")

    run_records = []
    completed = []
    common_training_identity = None
    for tile_id, tile in zip(tile_ids, tiles):
        run = Path(tile_runs[tile_id]).expanduser().resolve()
        binding = _expected_tile_binding(
            plan, plan_root, inventory, tile, indices[tile_id]
        )
        provenance, params, params_hash, artifacts = _completed_tile_run(
            run,
            binding,
            georeferencing,
            allow_nonproduction_georeferencing_for_diagnostic=(
                diagnostic_nonproduction
                and allow_nonproduction_georeferencing_for_diagnostic
            ),
        )
        _verify_training_source_binding(provenance, inventory)
        training_identity = _training_identity(provenance)
        if common_training_identity is None:
            common_training_identity = training_identity
        elif training_identity != common_training_identity:
            raise ArtifactError(
                "seam-probe tiles do not share one trainer implementation/configuration"
            )
        run_records.append({
            "tile_id": tile_id,
            "run": str(run),
            "run_provenance_sha256": sha256_file(run / "run_provenance.json"),
            "best_checkpoint_sha256": sha256_file(run / "best_checkpoint.json"),
            "params_sha256": params_hash,
            "initial_cloud_sha256": artifacts["initial_cloud_sha256"],
            "source_splat_file": artifacts["splat_file"],
            "source_splat_sha256": artifacts["splat_sha256"],
            "training_identity": training_identity,
            "tile_plan_binding": binding,
            "selected_model": provenance["model_selection"],
        })
        completed.append((tile, params, params_hash))

    combined, ownership = _concatenate_core_params(plan, completed)
    if any(item["core_owned_gaussians"] <= 0 for item in ownership):
        raise ArtifactError("each seam tile must contribute core-owned Gaussians")
    from rtk_splat.backends.gsplat import _load_checkpoint_gaussians

    component_params = {}
    for item, (_, params_path, params_hash) in zip(run_records, completed):
        component_params[item["tile_id"]] = _load_checkpoint_gaussians(
            params_path, "cpu"
        )
        if sha256_file(params_path) != params_hash:
            raise ArtifactError("tile params changed while loading seam components")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        candidate_root = staging / "hard_ownership"
        candidate_root.mkdir()
        (candidate_root / "renders").mkdir()
        component_root = staging / "components"
        component_root.mkdir()
        for tile_id in tile_ids:
            (component_root / tile_id / "renders").mkdir(parents=True)
        seam_masks, seam_evidence = _build_seam_masks(
            reader,
            cfg,
            plan,
            validation_ids,
            band_m=1.0,
            segments=[selection["selected_segment_uv"]],
        )
        seam_ids = sorted(seam_masks)
        candidate = _evaluate_combined(
            combined,
            reader,
            cfg,
            candidate_root,
            seam_ids,
            device=device,
            maximum_visible_gaussians=int(maximum_visible_gaussians),
            maximum_render_depth_m=None,
            context_masks=seam_masks,
            context_mask_dilate_px=1,
        )
        components = {}
        for tile_id in tile_ids:
            components[tile_id] = _evaluate_combined(
                component_params[tile_id],
                reader,
                cfg,
                component_root / tile_id,
                seam_ids,
                device=device,
                maximum_visible_gaussians=int(maximum_visible_gaussians),
                maximum_render_depth_m=None,
                context_masks=seam_masks,
                context_mask_dilate_px=1,
            )
        expected_fingerprint = plan["source_binding"]["pose_fingerprint"]
        if any(
            record["pose_fingerprint"] != expected_fingerprint
            for record in (candidate, *components.values())
        ):
            raise ArtifactError("seam evaluation used a different pose artifact")
        reference = {
            key: float(sum(record[key] for record in components.values()) / 2.0)
            for key in _METRICS
        }
        regressions = {
            "psnr_masked_loss_db": float(
                reference["psnr_masked"] - candidate["psnr_masked"]
            ),
            "psnr_masked_cc_loss_db": float(
                reference["psnr_masked_cc"] - candidate["psnr_masked_cc"]
            ),
            "ssim_loss": float(reference["ssim"] - candidate["ssim"]),
            "lpips_cc_increase": float(
                candidate["lpips_cc"] - reference["lpips_cc"]
            ),
        }
        metrics_record = {
            "schema_version": SCHEMA_VERSION,
            "evaluation": {
                "aggregation": (
                    "hard_half_open_core_ownership_on_shared_heldout_seam"
                ),
                "comparison": "component_arithmetic_mean_no_model_selection",
                "validation_split": "intersection_of_sealed_tile_val_selections",
                "eval_ids": seam_ids,
            },
            "selection": selection,
            "evidence": seam_evidence,
            "components": {
                tile_id: {key: components[tile_id][key] for key in _METRICS}
                for tile_id in tile_ids
            },
            "component_mean_reference": reference,
            "candidate": {key: candidate[key] for key in _METRICS},
            "regressions": regressions,
            "gates": {
                "maximum_psnr_loss_db": maximum_psnr_loss,
                "maximum_ssim_loss": maximum_ssim_loss,
                "maximum_lpips_cc_increase": maximum_lpips_increase,
            },
        }
        checks = _recomputed_checks(metrics_record)
        quality_passed = bool(checks) and all(checks.values())
        quality_record = {
            "schema_version": SCHEMA_VERSION,
            "passed": quality_passed,
            "checks": checks,
            "interpretation": (
                "accepted adjacent-tile hard-ownership seam"
                if quality_passed
                else "diagnostic seam regression; inspect before field expansion"
            ),
        }
        probe_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "name": name,
            "quality_passed": quality_passed,
            "provisional": diagnostic_nonproduction,
            "metric_georeferencing_claim_eligible": production_georeferencing,
            "tile_ids": tile_ids,
            "tile_plan": {
                "name": plan["name"],
                "manifest_sha256": sha256_file(plan_root / "manifest.json"),
                "tile_plan_sha256": sha256_file(plan_root / "tile_plan.json"),
                "source_inventory_sha256": canonical_hash(inventory),
            },
            "ownership": {
                "rule": plan["partition"]["boundary_rule"],
                "tiles": ownership,
                "core_owned_gaussians": int(len(combined["means"])),
            },
        }
        viewmats, _ = load_pose_artifact(reader.root, cfg)
        if pose_fingerprint(viewmats) != expected_fingerprint:
            raise ArtifactError("configured seam-probe pose disagrees with TilePlan")
        provenance_record = {
            "schema_version": SCHEMA_VERSION,
            "diagnostic_nonproduction_georeferencing": diagnostic_nonproduction,
            "tile_plan": copy.deepcopy(probe_record["tile_plan"]),
            "tile_plan_locator": str(plan_root),
            "source_segment": str(reader.root.resolve()),
            "source_contract_files": inventory["segment"]["contract_files"],
            "pose": inventory["pose"],
            "georeferencing": copy.deepcopy(georeferencing),
            "neighbor_selection": copy.deepcopy(selection),
            "common_training_identity": common_training_identity,
            "tile_runs": run_records,
            "package": collect_package_state(),
            "device": str(device),
        }
        _write_json(staging / "metrics.json", metrics_record)
        _write_json(staging / "quality.json", quality_record)
        _write_json(staging / "probe.json", probe_record)
        _write_json(staging / "provenance.json", provenance_record)

        verify_tile_plan(
            plan_root,
            segment=reader.root,
            pose_root=pose_root,
            rehash_sources=True,
        )
        if sha256_file(plan_root / "manifest.json") != (
            probe_record["tile_plan"]["manifest_sha256"]
        ):
            raise ArtifactError("TilePlan changed while seam probe was running")
        for item in run_records:
            run = Path(item["run"])
            if (
                sha256_file(run / "params.pt") != item["params_sha256"]
                or sha256_file(run / "run_provenance.json")
                != item["run_provenance_sha256"]
                or sha256_file(run / "best_checkpoint.json")
                != item["best_checkpoint_sha256"]
                or sha256_file(run / item["source_splat_file"])
                != item["source_splat_sha256"]
            ):
                raise ArtifactError("tile run changed while seam probe was running")
        _write_json(staging / "manifest.json", _manifest(staging))
        verify_tile_seam_probe(staging)
        publish_directory_noreplace(staging, destination)
        verify_tile_seam_probe(destination)
        return {
            "probe": str(destination),
            "quality_passed": quality_passed,
            "provisional": diagnostic_nonproduction,
            "tile_ids": tile_ids,
            "n_validation_frames": len(seam_ids),
            "metric_depth_pixels_in_band": seam_evidence[
                "metric_depth_pixels_in_band"
            ],
            "candidate_metrics": metrics_record["candidate"],
            "regressions": regressions,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
