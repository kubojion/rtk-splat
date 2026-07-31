"""Non-destructive COLMAP Global Mapper backend.

This backend deliberately reuses a completed calibrated-stereo front end:
features, verified matches, image names, cameras, frames, and rig definitions
come from an existing COLMAP sidecar.  The source database is never opened by
COLMAP again.  ``global-prepare`` creates and verifies an independent database
copy, ``global-solve`` runs the integrated GLOMAP backend in COLMAP 4.1+, and
``global-export`` publishes poses only after strict metric and structural
quality gates pass.

Global mapping is never part of the normal ``all`` pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np

from .calibration_io import load_raw_colmap_rig_trajectory
from .colmap_stereo import (aligned_viewmat, qvec_to_rotmat,
                            robust_similarity)
from .pose_artifacts import (pose_artifact_dir, pose_artifact_name,
                             pose_fingerprint)


_MODEL_STAT_NAMES = {
    "Rigs": "rigs",
    "Cameras": "cameras",
    "Frames": "frames",
    "Registered frames": "registered_frames",
    "Images": "images",
    "Registered images": "registered_images",
    "Points": "points",
    "Observations": "observations",
    "Mean track length": "mean_track_length",
    "Mean observations per image": "mean_observations_per_image",
    "Mean reprojection error": "mean_reprojection_error_px",
}
_MODEL_STAT_RE = re.compile(
    r"\]\s*(Rigs|Cameras|Frames|Registered frames|Images|"
    r"Registered images|Points|Observations|Mean track length|"
    r"Mean observations per image|Mean reprojection error):\s*"
    r"([-+0-9.eE]+)"
)
_INTEGER_MODEL_STATS = {
    "rigs", "cameras", "frames", "registered_frames", "images",
    "registered_images", "points", "observations",
}


def _value(namespace, name: str, default):
    return getattr(namespace, name, default)


def _settings(cfg) -> dict:
    if not hasattr(cfg, "global_mapper"):
        raise ValueError("configuration is missing the global_mapper section")
    g = cfg.global_mapper
    gates = _value(g, "quality_gates", SimpleNamespace())
    return {
        "source_pose_artifact":
            str(_value(g, "source_pose_artifact", "colmap_stereo")),
        "source_workspace_subdir":
            str(_value(g, "source_workspace_subdir", "colmap")),
        "num_threads": int(_value(g, "num_threads", 8)),
        "random_seed": int(_value(g, "random_seed", 7)),
        "ba_num_iterations": int(_value(g, "ba_num_iterations", 3)),
        "max_num_tracks": int(_value(g, "max_num_tracks", 0)),
        "required_tracks_per_view":
            int(_value(g, "required_tracks_per_view", 0)),
        "skip_retriangulation":
            bool(_value(g, "skip_retriangulation", False)),
        "process_nice": int(_value(g, "process_nice", 10)),
        "resource_sample_interval_s":
            float(_value(g, "resource_sample_interval_s", 2.0)),
        "minimum_available_memory_gb":
            float(_value(g, "minimum_available_memory_gb", 4.0)),
        "low_memory_consecutive_samples":
            int(_value(g, "low_memory_consecutive_samples", 2)),
        "minimum_free_space_gb":
            float(_value(g, "minimum_free_space_gb", 15.0)),
        "minimum_runtime_free_space_gb":
            float(_value(g, "minimum_runtime_free_space_gb", 5.0)),
        "alignment_max_error_m": float(cfg.colmap.alignment_max_error_m),
        "alignment_ransac_iterations":
            int(cfg.colmap.alignment_ransac_iterations),
        "scale_range": [float(value) for value in cfg.colmap.scale_range],
        "gates": {
            "max_reprojection_error_px":
                float(_value(gates, "max_reprojection_error_px", 1.0)),
            "min_observation_fraction":
                float(_value(gates, "min_observation_fraction", 0.90)),
            "min_track_length_fraction":
                float(_value(gates, "min_track_length_fraction", 0.75)),
            "max_baseline_error_m":
                float(_value(gates, "max_baseline_error_m", 5.0e-4)),
            "max_rig_rotation_error_deg":
                float(_value(gates, "max_rig_rotation_error_deg", 0.02)),
            "max_intrinsics_error":
                float(_value(gates, "max_intrinsics_error", 1.0e-8)),
            "min_observations_per_image":
                int(_value(gates, "min_observations_per_image", 100)),
            "max_rtk_median_error_m":
                float(_value(gates, "max_rtk_median_error_m", 0.06)),
            "max_rtk_p95_error_m":
                float(_value(gates, "max_rtk_p95_error_m", 0.12)),
            "max_rtk_error_m":
                float(_value(gates, "max_rtk_error_m", 0.15)),
            "min_alignment_inlier_fraction":
                float(_value(gates, "min_alignment_inlier_fraction", 1.0)),
            "max_step_m": float(_value(gates, "max_step_m", 0.05)),
            "max_rotation_step_deg":
                float(_value(gates, "max_rotation_step_deg", 0.9)),
            "reference_regression_fraction":
                float(_value(gates, "reference_regression_fraction", 0.05)),
        },
    }


def _sidecar_dir(seg_dir: Path, cfg) -> Path:
    name = pose_artifact_name(cfg)
    if name == "rtk":
        raise ValueError("global stages require a named pose artifact")
    return pose_artifact_dir(seg_dir, cfg)


def _source_sidecar(seg_dir: Path, settings: dict) -> Path:
    source = settings["source_pose_artifact"]
    source_cfg = SimpleNamespace(pose=SimpleNamespace(artifact=source))
    if pose_artifact_name(source_cfg) == "rtk":
        raise ValueError("global_mapper.source_pose_artifact must be a sidecar")
    return pose_artifact_dir(seg_dir, source_cfg)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _new_sidecar(root: Path, name: str):
    """Atomically publish a newly prepared pose sidecar."""
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"pose artifact already exists: {final}")
    lock = root / f".{name}.prepare.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"another prepare process owns {lock}") from exc
    temporary = None
    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{name}.prepare-", dir=root))
        yield temporary
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"pose artifact appeared during prepare: {final}")
        os.rename(temporary, final)
        temporary = None
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _readonly_database(path: Path) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    uri = "file:" + quote(str(resolved), safe="/") + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def database_inventory(path: Path, *, run_quick_check: bool = True) -> dict:
    """Return a compact, deterministic fingerprint of a COLMAP database."""
    connection = _readonly_database(path)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0] \
            if run_quick_check else "not_run"
        tables = {
            name: int(connection.execute(
                f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in (
                "cameras", "images", "keypoints", "descriptors", "matches",
                "two_view_geometries", "pose_priors", "rigs", "frames",
                "frame_data")
        }
        cameras = []
        for camera_id, model, width, height, params, prior in \
                connection.execute(
                    "SELECT camera_id, model, width, height, params, "
                    "prior_focal_length FROM cameras ORDER BY camera_id"):
            cameras.append({
                "camera_id": int(camera_id),
                "model_id": int(model),
                "width": int(width),
                "height": int(height),
                "params": np.frombuffer(params, dtype="<f8").tolist(),
                "prior_focal_length": int(prior),
            })
        image_rows = [
            (int(image_id), str(name).replace("\\", "/"), int(camera_id))
            for image_id, name, camera_id in connection.execute(
                "SELECT image_id, name, camera_id FROM images "
                "ORDER BY image_id")
        ]
        name_digest = hashlib.sha256()
        for image_id, name, camera_id in image_rows:
            name_digest.update(f"{image_id}\t{camera_id}\t{name}\n".encode())
        frame_sensor_counts = [
            int(count) for count, in connection.execute(
                "SELECT COUNT(*) FROM frame_data GROUP BY frame_id")
        ]
        sums = {
            "keypoints": int(connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM keypoints").fetchone()[0]),
            "verified_correspondences": int(connection.execute(
                "SELECT COALESCE(SUM(rows), 0) "
                "FROM two_view_geometries").fetchone()[0]),
        }
        user_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()
    return {
        "quick_check": str(quick_check),
        "user_version": user_version,
        "tables": tables,
        "sums": sums,
        "cameras": cameras,
        "image_name_rows_sha256": name_digest.hexdigest(),
        "frame_sensor_count_min":
            min(frame_sensor_counts) if frame_sensor_counts else 0,
        "frame_sensor_count_max":
            max(frame_sensor_counts) if frame_sensor_counts else 0,
    }


def _validate_source_inventory(inventory: dict, n_frames: int) -> None:
    expected = {
        "cameras": 2,
        "images": 2 * n_frames,
        "rigs": 1,
        "frames": n_frames,
        "frame_data": 2 * n_frames,
    }
    for name, count in expected.items():
        actual = inventory["tables"][name]
        if actual != count:
            raise ValueError(
                f"source database {name} count {actual}, expected {count}")
    if inventory["quick_check"] != "ok":
        raise ValueError(
            f"source database quick_check={inventory['quick_check']!r}")
    if (inventory["frame_sensor_count_min"],
            inventory["frame_sensor_count_max"]) != (2, 2):
        raise ValueError("source database does not contain two sensors per frame")
    if inventory["tables"]["pose_priors"] != 0:
        raise ValueError(
            "visual-only Global Mapper A/B requires zero database pose priors")
    for camera in inventory["cameras"]:
        if camera["model_id"] != 1 or not camera["prior_focal_length"]:
            raise ValueError(
                "global mapping requires calibrated PINHOLE focal priors")


def _stat_record(path: Path) -> dict:
    stat = Path(path).stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
    }


def _verify_prepared_provenance(
        sidecar: Path,
        settings: dict,
        *,
        verify_database: bool,
) -> dict:
    record = json.loads((sidecar / "source.json").read_text())
    if record["settings"] != settings:
        raise RuntimeError(
            "Global Mapper settings changed after prepare; use a new artifact")
    if record["implementation_sha256"] != _sha256_file(Path(__file__)):
        raise RuntimeError(
            "Global Mapper implementation changed after prepare; use a new "
            "artifact so the experiment remains attributable")
    source = Path(record["source_sidecar"])
    checks = {
        "source_quality_sha256": source / "quality.json",
        "source_sidecar_config_sha256": source / "sidecar_config.json",
        "source_frame_manifest_sha256": source / "frame_manifest.json",
        "source_rig_config_sha256": Path(record["source_rig_config"]),
        "source_model_cameras_sha256":
            Path(record["source_text_model"]) / "cameras.txt",
        "source_model_rigs_sha256":
            Path(record["source_text_model"]) / "rigs.txt",
        "source_model_frames_sha256":
            Path(record["source_text_model"]) / "frames.txt",
        "source_exported_viewmats_sha256": source / "viewmats.npy",
        "source_exported_centers_sha256": source / "cam_centers.npy",
    }
    for field, path in checks.items():
        if not path.is_file() or _sha256_file(path) != record[field]:
            raise RuntimeError(
                f"prepared source provenance changed: {path}")
    if verify_database and (
            _sha256_file(Path(record["source_database"]))
            != record["database_sha256"]):
        raise RuntimeError("source database changed after global-prepare")
    return record


def prepare(seg_dir: Path, cfg) -> Path:
    """Clone a completed stereo front end into a new isolated artifact."""
    settings = _settings(cfg)
    destination = _sidecar_dir(seg_dir, cfg)
    source = _source_sidecar(seg_dir, settings)
    if destination == source:
        raise ValueError("source and destination pose artifacts must differ")
    source_work = source / settings["source_workspace_subdir"]
    source_database = source_work / "database.db"
    source_images = source_work / "images"
    source_rig = source_work / "rig_config.json"
    for required in (
            source_database, source_rig, source / "sidecar_config.json",
            source / "frame_manifest.json", source / "quality.json"):
        if not required.is_file():
            raise FileNotFoundError(f"source stereo sidecar is incomplete: {required}")
    if not source_images.is_dir():
        raise FileNotFoundError(f"source image tree is missing: {source_images}")
    wal = source_database.with_name(source_database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(
            f"source database has a non-empty WAL; wait for its writer: {wal}")

    meta = json.loads((seg_dir / "segment_meta.json").read_text())
    n_frames = int(meta["n_frames"])
    inventory = database_inventory(source_database)
    _validate_source_inventory(inventory, n_frames)
    # The pose_artifacts directory need not exist for a new generic segment.
    free_bytes = shutil.disk_usage(seg_dir).free
    required_bytes = int(settings["minimum_free_space_gb"] * (1024 ** 3))
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.1f} GiB free; require "
            f"{settings['minimum_free_space_gb']:.1f} GiB")

    source_stat_before = _stat_record(source_database)
    source_hash_before = _sha256_file(source_database)
    root = destination.parent
    with _new_sidecar(root, destination.name) as temporary:
        work = temporary / "global"
        work.mkdir()
        clone_partial = work / ".database.db.copying"
        shutil.copy2(source_database, clone_partial)
        with clone_partial.open("rb") as stream:
            os.fsync(stream.fileno())
        clone_hash = _sha256_file(clone_partial)
        source_stat_after = _stat_record(source_database)
        source_hash_after = _sha256_file(source_database)
        if (source_stat_before != source_stat_after
                or source_hash_before != source_hash_after):
            raise RuntimeError("source database changed while it was copied")
        if clone_hash != source_hash_before:
            raise RuntimeError("database clone hash does not match its source")
        if clone_partial.stat().st_ino == source_database.stat().st_ino:
            raise RuntimeError("database clone unexpectedly shares the source inode")
        clone_inventory = database_inventory(clone_partial)
        if clone_inventory != inventory:
            raise RuntimeError("database clone inventory differs from its source")
        database = work / "database.db"
        os.replace(clone_partial, database)
        (work / "images").symlink_to(source_images.resolve(),
                                     target_is_directory=True)
        rtk_snapshot = temporary / "rtk_camera_centers.npy"
        raw_rtk_centers = np.load(seg_dir / "cam_centers.npy").astype(np.float64)
        if (raw_rtk_centers.shape != (n_frames, 3)
                or not np.isfinite(raw_rtk_centers).all()):
            raise ValueError("segment RTK camera centres are invalid")
        np.save(rtk_snapshot, raw_rtk_centers)

        source_quality = json.loads((source / "quality.json").read_text())
        source_text_model = (
            source_work / "models_text"
            / Path(source_quality["selected_model"]).name)
        for required in (
                source_text_model / "cameras.txt",
                source_text_model / "rigs.txt",
                source_text_model / "frames.txt",
                source / "viewmats.npy", source / "cam_centers.npy"):
            if not required.is_file():
                raise FileNotFoundError(
                    f"selected source pose model is incomplete: {required}")
        source_record = {
            "schema_version": 1,
            "backend": "COLMAP 4.1+ integrated Global Mapper (GLOMAP)",
            "destination_pose_artifact": destination.name,
            "source_pose_artifact": settings["source_pose_artifact"],
            "source_sidecar": str(source.resolve()),
            "source_database": str(source_database.resolve()),
            "source_images": str(source_images.resolve()),
            "source_rig_config": str(source_rig.resolve()),
            "n_frames": n_frames,
            "database_sha256": source_hash_before,
            "database_stat": source_stat_before,
            "database_inventory": inventory,
            "source_pose_fingerprint": source_quality.get("pose_fingerprint"),
            "source_text_model": str(source_text_model.resolve()),
            "source_quality_sha256": _sha256_file(source / "quality.json"),
            "source_sidecar_config_sha256":
                _sha256_file(source / "sidecar_config.json"),
            "source_frame_manifest_sha256":
                _sha256_file(source / "frame_manifest.json"),
            "source_rig_config_sha256": _sha256_file(source_rig),
            "source_model_cameras_sha256":
                _sha256_file(source_text_model / "cameras.txt"),
            "source_model_rigs_sha256":
                _sha256_file(source_text_model / "rigs.txt"),
            "source_model_frames_sha256":
                _sha256_file(source_text_model / "frames.txt"),
            "source_exported_viewmats_sha256":
                _sha256_file(source / "viewmats.npy"),
            "source_exported_centers_sha256":
                _sha256_file(source / "cam_centers.npy"),
            "source_segment_rtk_centers_sha256":
                _sha256_file(seg_dir / "cam_centers.npy"),
            "rtk_centers_snapshot_sha256": _sha256_file(rtk_snapshot),
            "implementation_sha256": _sha256_file(Path(__file__)),
            "settings": settings,
            "features_or_matches_recomputed": False,
            "source_artifacts_modified": False,
        }
        _atomic_json(temporary / "source.json", source_record)
        _atomic_json(temporary / "sidecar_config.json", {
            "schema_version": 1,
            "pose_artifact": destination.name,
            "source_pose_artifact": settings["source_pose_artifact"],
            "n_frames": n_frames,
            "metric_export": "robust fixed-scale SE(3)",
            "sim3_use": "diagnostic only",
            "settings": settings,
        })
    print(
        f"prepared isolated Global Mapper artifact {destination}; reused "
        f"{inventory['tables']['images']} images and "
        f"{inventory['tables']['two_view_geometries']} verified pairs")
    print(
        f"copied and SHA-256 verified {source_stat_before['size_bytes']/1024**3:.2f} "
        "GiB database; source was not modified")
    return destination / "global"


def global_mapper_command(
        work: Path,
        cfg,
        executable: str,
        output_path: Path | None = None,
) -> list[str]:
    settings = _settings(cfg)
    output = output_path or work / "sparse_global.incomplete"
    command = [
        str(executable), "global_mapper",
        "--log_target", "stdout",
        "--database_path", str(work / "database.db"),
        "--image_path", str(work / "images"),
        "--output_path", str(output),
        "--GlobalMapper.num_threads", str(settings["num_threads"]),
        "--GlobalMapper.random_seed", str(settings["random_seed"]),
        "--GlobalMapper.decompose_relative_pose", "1",
        "--GlobalMapper.ba_num_iterations",
        str(settings["ba_num_iterations"]),
        "--GlobalMapper.gp_optimize_positions", "1",
        "--GlobalMapper.gp_optimize_points", "1",
        "--GlobalMapper.gp_optimize_scales", "1",
        "--GlobalMapper.gp_use_gpu", "0",
        "--GlobalMapper.ba_refine_focal_length", "0",
        "--GlobalMapper.ba_refine_principal_point", "0",
        "--GlobalMapper.ba_refine_extra_params", "0",
        "--GlobalMapper.refine_sensor_from_rig", "0",
        "--GlobalMapper.ba_refine_rig_from_world", "1",
        "--GlobalMapper.ba_refine_points3D", "1",
        "--GlobalMapper.ba_ceres_use_gpu", "0",
    ]
    if settings["max_num_tracks"] > 0:
        command.extend([
            "--GlobalMapper.keep_max_num_tracks",
            str(settings["max_num_tracks"]),
        ])
    if settings["required_tracks_per_view"] > 0:
        command.extend([
            "--GlobalMapper.track_required_tracks_per_view",
            str(settings["required_tracks_per_view"]),
        ])
    if settings["skip_retriangulation"]:
        command.extend([
            "--GlobalMapper.skip_retriangulation", "1",
        ])
    return command


def _resolve_colmap(cfg) -> str:
    requested = os.environ.get("COLMAP_BIN", str(cfg.colmap.executable))
    executable = shutil.which(requested)
    if executable is None:
        raise FileNotFoundError(
            f"COLMAP executable {requested!r} was not found")
    help_text = subprocess.run(
        [executable, "-h"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False).stdout
    if "COLMAP 4.1.1" not in help_text or "global_mapper" not in help_text:
        raise RuntimeError(
            "this backend was validated with COLMAP 4.1.1 and requires its "
            "integrated global_mapper command")
    return executable


def _meminfo() -> dict[str, int]:
    result = {}
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            name, value = line.split(":", 1)
            token = value.strip().split()[0]
            result[name] = int(token) * 1024
    return result


def _process_memory(pid: int) -> tuple[int, int]:
    rss = hwm = 0
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text().splitlines()
    except FileNotFoundError:
        return 0, 0
    for line in lines:
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    return rss, hwm


def _terminate_process_group(process: subprocess.Popen, log, reason: str) -> None:
    log.write(f"\nSAFETY ABORT: {reason}\n")
    log.flush()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class _MonitoredProcessInterrupted(RuntimeError):
    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.usage = usage


def _run_monitored(
        argv: list[str],
        log,
        samples_path: Path,
        settings: dict,
) -> dict:
    """Run COLMAP while sampling memory and failing safely before host OOM."""
    started = time.time()

    def set_nice():
        requested = settings["process_nice"]
        if requested:
            os.nice(requested)

    process = subprocess.Popen(
        argv, stdout=log, stderr=subprocess.STDOUT, text=True,
        start_new_session=True, preexec_fn=set_nice)
    peak_rss = 0
    minimum_available = math.inf
    minimum_disk_free = math.inf
    low_samples = 0
    aborted_reason = None
    caught_exception = None
    forwarded_signal = None
    old_handlers = {}

    def forward_signal(signum, _frame):
        nonlocal forwarded_signal
        forwarded_signal = signum
        raise InterruptedError(f"received signal {signum}")

    monitored_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        monitored_signals.append(signal.SIGHUP)
    for signum in monitored_signals:
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    try:
        with samples_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "elapsed_s", "pid", "process_rss_bytes", "process_hwm_bytes",
                "mem_available_bytes", "swap_free_bytes", "disk_free_bytes",
            ])
            while True:
                rss, hwm = _process_memory(process.pid)
                memory = _meminfo()
                available = memory.get("MemAvailable", 0)
                swap_free = memory.get("SwapFree", 0)
                disk_free = shutil.disk_usage(samples_path.parent).free
                elapsed = time.time() - started
                peak_rss = max(peak_rss, rss, hwm)
                minimum_available = min(minimum_available, available)
                minimum_disk_free = min(minimum_disk_free, disk_free)
                writer.writerow([
                    f"{elapsed:.3f}", process.pid, rss, hwm, available,
                    swap_free, disk_free])
                stream.flush()
                memory_threshold = int(
                    settings["minimum_available_memory_gb"] * (1024 ** 3))
                low_samples = (
                    low_samples + 1
                    if available < memory_threshold else 0)
                if low_samples >= settings["low_memory_consecutive_samples"]:
                    aborted_reason = (
                        f"MemAvailable stayed below "
                        f"{settings['minimum_available_memory_gb']:.1f} GiB")
                    _terminate_process_group(process, log, aborted_reason)
                    break
                disk_threshold = int(
                    settings["minimum_runtime_free_space_gb"] * (1024 ** 3))
                if disk_free < disk_threshold:
                    aborted_reason = (
                        f"free disk fell below "
                        f"{settings['minimum_runtime_free_space_gb']:.1f} GiB")
                    _terminate_process_group(process, log, aborted_reason)
                    break
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(settings["resource_sample_interval_s"])
    except BaseException as exc:
        caught_exception = exc
        if process.poll() is None:
            _terminate_process_group(
                process, log,
                f"monitor interrupted by {type(exc).__name__}: {exc}")
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if process.poll() is None:
            _terminate_process_group(
                process, log, "monitor exited while COLMAP was still running")
    returncode = process.wait()
    ended = time.time()
    usage = {
        "started_unix_s": started,
        "ended_unix_s": ended,
        "wall_time_s": ended - started,
        "returncode": int(returncode),
        "peak_process_rss_bytes": int(peak_rss),
        "minimum_system_available_bytes":
            int(minimum_available if math.isfinite(minimum_available) else 0),
        "minimum_disk_free_bytes":
            int(minimum_disk_free if math.isfinite(minimum_disk_free) else 0),
        "safety_aborted": aborted_reason is not None,
        "safety_abort_reason": aborted_reason,
        "forwarded_signal": forwarded_signal,
        "samples_csv": str(samples_path),
    }
    if caught_exception is not None:
        raise _MonitoredProcessInterrupted(
            f"Global Mapper monitor was interrupted: {caught_exception}",
            usage) from caught_exception
    return usage


def parse_model_analyzer(text: str) -> dict:
    result = {}
    for label, raw_value in _MODEL_STAT_RE.findall(text):
        name = _MODEL_STAT_NAMES[label]
        value = float(raw_value)
        result[name] = int(round(value)) if name in _INTEGER_MODEL_STATS else value
    missing = set(_MODEL_STAT_NAMES.values()) - set(result)
    if missing:
        raise ValueError(
            "model_analyzer output is missing: " + ", ".join(sorted(missing)))
    return result


def _analyze_model(executable: str, model: Path, log) -> dict:
    argv = [
        executable, "model_analyzer", "--log_target", "stdout",
        "--path", str(model),
    ]
    result = subprocess.run(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    log.write("\n$ " + " ".join(argv) + "\n")
    log.write(result.stdout)
    log.flush()
    if result.returncode:
        raise RuntimeError(f"model_analyzer failed for {model}")
    stats = parse_model_analyzer(result.stdout)
    stats["path"] = str(model)
    return stats


def _analyze_models(executable: str, root: Path, log) -> tuple[Path, dict]:
    candidates = []
    for model in sorted(root.iterdir()) if root.is_dir() else []:
        if model.is_dir() and (model / "images.bin").is_file():
            candidates.append((model, _analyze_model(executable, model, log)))
    if not candidates:
        raise RuntimeError(f"no COLMAP model was produced under {root}")
    return max(
        candidates,
        key=lambda item: (
            item[1]["registered_images"], item[1]["observations"],
            -item[1]["mean_reprojection_error_px"]),
    )


def _convert_model(executable: str, model: Path, output: Path, log) -> None:
    output.mkdir(parents=True)
    argv = [
        executable, "model_converter", "--log_target", "stdout",
        "--input_path", str(model), "--output_path", str(output),
        "--output_type", "TXT",
    ]
    log.write("\n$ " + " ".join(argv) + "\n")
    log.flush()
    result = subprocess.run(
        argv, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"model_converter failed for {model}")


def _data_rows(path: Path) -> list[list[str]]:
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append(stripped.split())
    return rows


def _image_observation_stats(path: Path) -> dict:
    """Count retained 3D observations per registered COLMAP image."""
    counts = []
    names = set()
    expecting_pose = True
    with Path(path).open() as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = stripped.split()
            if expecting_pose:
                if len(row) < 10:
                    raise ValueError("malformed images.txt pose row")
                name = row[9].replace("\\", "/")
                if name in names:
                    raise ValueError(f"duplicate image name in model: {name}")
                names.add(name)
            else:
                if len(row) % 3:
                    raise ValueError("malformed images.txt observations row")
                counts.append(sum(
                    int(row[cursor]) >= 0
                    for cursor in range(2, len(row), 3)))
            expecting_pose = not expecting_pose
    if not expecting_pose:
        raise ValueError("images.txt is missing its final observations row")
    if not counts:
        raise ValueError("images.txt contains no registered images")
    values = np.asarray(counts, dtype=np.int64)
    return {
        "names": names,
        "minimum": int(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "maximum": int(values.max()),
    }


def inspect_rig_model(
        model_text: Path,
        database_path: Path,
        expected_inventory: dict,
        n_frames: int,
        baseline_m: float,
) -> dict:
    """Validate cameras, rigid stereo calibration, and complete rig frames."""
    camera_rows = _data_rows(model_text / "cameras.txt")
    model_cameras = {}
    for row in camera_rows:
        if len(row) < 8:
            raise ValueError("malformed cameras.txt row")
        camera_id = int(row[0])
        if row[1] != "PINHOLE":
            raise ValueError(f"camera {camera_id} is not PINHOLE")
        model_cameras[camera_id] = {
            "width": int(row[2]),
            "height": int(row[3]),
            "params": [float(value) for value in row[4:]],
        }
    expected_cameras = {
        item["camera_id"]: item for item in expected_inventory["cameras"]}
    if set(model_cameras) != set(expected_cameras):
        raise ValueError("output camera IDs differ from the source database")
    maximum_intrinsics_error = 0.0
    for camera_id, actual in model_cameras.items():
        expected = expected_cameras[camera_id]
        if (actual["width"], actual["height"]) != (
                expected["width"], expected["height"]):
            raise ValueError(f"camera {camera_id} dimensions changed")
        error = float(np.max(np.abs(
            np.asarray(actual["params"]) - np.asarray(expected["params"]))))
        maximum_intrinsics_error = max(maximum_intrinsics_error, error)

    rig_rows = _data_rows(model_text / "rigs.txt")
    if len(rig_rows) != 1:
        raise ValueError(f"expected one rig, found {len(rig_rows)}")
    rig = rig_rows[0]
    if len(rig) != 14 or int(rig[1]) != 2 or rig[2] != "CAMERA":
        raise ValueError("output rig is not the expected two-camera rig")
    reference_camera_id = int(rig[3])
    if rig[4] != "CAMERA" or int(rig[6]) != 1:
        raise ValueError("output rig has an invalid non-reference sensor")
    right_camera_id = int(rig[5])
    rotation = qvec_to_rotmat([float(v) for v in rig[7:11]])
    translation = np.asarray([float(v) for v in rig[11:14]])
    expected_translation = np.array([-float(baseline_m), 0.0, 0.0])
    baseline_error = float(np.linalg.norm(translation - expected_translation))
    rotation_cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error_deg = float(np.degrees(np.arccos(rotation_cosine)))

    connection = _readonly_database(database_path)
    try:
        image_by_id = {
            int(image_id): (str(name).replace("\\", "/"), int(camera_id))
            for image_id, name, camera_id in connection.execute(
                "SELECT image_id, name, camera_id FROM images")
        }
    finally:
        connection.close()
    expected_names = {name for name, _ in image_by_id.values()}
    observation_stats = _image_observation_stats(model_text / "images.txt")
    if observation_stats["names"] != expected_names:
        raise ValueError("registered images.txt data does not cover every source image")
    observed_names = set()
    frame_rows = _data_rows(model_text / "frames.txt")
    if len(frame_rows) != n_frames:
        raise ValueError(
            f"output contains {len(frame_rows)} frames, expected {n_frames}")
    for row in frame_rows:
        if len(row) < 10:
            raise ValueError("malformed frames.txt row")
        count = int(row[9])
        if count != 2 or len(row) != 10 + 3 * count:
            raise ValueError("output frame does not contain exactly two images")
        names = []
        sensors = set()
        for cursor in range(10, len(row), 3):
            sensor_type = row[cursor]
            sensor_id = int(row[cursor + 1])
            image_id = int(row[cursor + 2])
            if sensor_type != "CAMERA" or image_id not in image_by_id:
                raise ValueError("output frame references invalid image data")
            name, database_camera_id = image_by_id[image_id]
            if database_camera_id != sensor_id:
                raise ValueError("frame sensor ID disagrees with image camera ID")
            names.append(name)
            sensors.add(sensor_id)
            observed_names.add(name)
        if sensors != {reference_camera_id, right_camera_id}:
            raise ValueError("output frame does not contain both rig cameras")
        stems = {Path(name).stem for name in names}
        if len(stems) != 1:
            raise ValueError("left/right output frame indices do not match")
    if observed_names != expected_names:
        raise ValueError("registered frame data does not cover every source image")

    trajectory = load_raw_colmap_rig_trajectory(
        model_text, database_path, expected_frame_count=n_frames)
    return {
        "n_cameras": len(model_cameras),
        "n_frames": len(frame_rows),
        "n_registered_images": len(observed_names),
        "reference_camera_id": reference_camera_id,
        "right_camera_id": right_camera_id,
        "baseline_m": float(np.linalg.norm(translation)),
        "baseline_translation": translation.tolist(),
        "baseline_error_m": baseline_error,
        "rig_rotation_error_deg": rotation_error_deg,
        "max_intrinsics_error": maximum_intrinsics_error,
        "observations_per_image": {
            key: value for key, value in observation_stats.items()
            if key != "names"
        },
        "left_pose_count": len(trajectory.frame_indices),
    }


def _find_source_model(source: Path, settings: dict) -> Path:
    quality = json.loads((source / "quality.json").read_text())
    selected = Path(str(quality.get("selected_model", "")))
    if selected.name and (source / settings["source_workspace_subdir"]
                          / "sparse_visual" / selected.name).is_dir():
        return (
            source / settings["source_workspace_subdir"]
            / "sparse_visual" / selected.name)
    models = sorted(
        path for path in (
            source / settings["source_workspace_subdir"] / "sparse_visual"
        ).iterdir()
        if path.is_dir() and (path / "images.bin").is_file())
    if not models:
        raise FileNotFoundError("source incremental COLMAP model is missing")
    return models[0]


def _structural_gates(
        candidate: dict,
        reference: dict,
        structure: dict,
        settings: dict,
        n_frames: int,
) -> dict:
    gates = settings["gates"]
    checks = {
        "registered_frames": {
            "value": candidate["registered_frames"],
            "required": n_frames,
            "passed": candidate["registered_frames"] == n_frames,
            "enforced": True,
        },
        "registered_images": {
            "value": candidate["registered_images"],
            "required": 2 * n_frames,
            "passed": candidate["registered_images"] == 2 * n_frames,
            "enforced": True,
        },
        "reprojection_error_px": {
            "value": candidate["mean_reprojection_error_px"],
            "maximum": gates["max_reprojection_error_px"],
            "passed": candidate["mean_reprojection_error_px"]
                      <= gates["max_reprojection_error_px"],
            "enforced": False,
            "reason": "different output point sets make this a result diagnostic",
        },
        "observation_fraction": {
            "value": candidate["observations"] / reference["observations"],
            "minimum": gates["min_observation_fraction"],
            "passed": candidate["observations"] / reference["observations"]
                      >= gates["min_observation_fraction"],
            "enforced": False,
            "reason": "different mappers retriangulate and prune different points",
        },
        "track_length_fraction": {
            "value": candidate["mean_track_length"]
                     / reference["mean_track_length"],
            "minimum": gates["min_track_length_fraction"],
            "passed": candidate["mean_track_length"]
                      / reference["mean_track_length"]
                      >= gates["min_track_length_fraction"],
            "enforced": False,
            "reason": "different output point sets are not a paired pose metric",
        },
        "baseline_error_m": {
            "value": structure["baseline_error_m"],
            "maximum": gates["max_baseline_error_m"],
            "passed": structure["baseline_error_m"]
                      <= gates["max_baseline_error_m"],
            "enforced": True,
        },
        "rig_rotation_error_deg": {
            "value": structure["rig_rotation_error_deg"],
            "maximum": gates["max_rig_rotation_error_deg"],
            "passed": structure["rig_rotation_error_deg"]
                      <= gates["max_rig_rotation_error_deg"],
            "enforced": True,
        },
        "intrinsics_error": {
            "value": structure["max_intrinsics_error"],
            "maximum": gates["max_intrinsics_error"],
            "passed": structure["max_intrinsics_error"]
                      <= gates["max_intrinsics_error"],
            "enforced": True,
        },
        "minimum_image_observations": {
            "value": structure["observations_per_image"]["minimum"],
            "minimum": gates["min_observations_per_image"],
            "passed": structure["observations_per_image"]["minimum"]
                      >= gates["min_observations_per_image"],
            "enforced": True,
        },
    }
    return checks


def solve(seg_dir: Path, cfg) -> dict:
    """Run Global Mapper against the isolated database and validate its model."""
    settings = _settings(cfg)
    sidecar = _sidecar_dir(seg_dir, cfg)
    work = sidecar / "global"
    source_record_path = sidecar / "source.json"
    if not source_record_path.is_file() or not (work / "database.db").is_file():
        raise FileNotFoundError("run global-prepare before global-solve")
    done = work / ".global_mapper.done.json"
    if done.exists():
        result = json.loads(done.read_text())
        print(f"Global Mapper already complete (verified record): {done}")
        return result
    for path in (
            work / "sparse_global", work / "sparse_global.incomplete",
            work / "models_text", work / "models_text.incomplete"):
        if path.exists():
            raise RuntimeError(
                f"refusing ambiguous or partial prior solve at {path}; use a "
                "new pose artifact name")

    source_record = _verify_prepared_provenance(
        sidecar, settings, verify_database=True)
    source_database = Path(source_record["source_database"])
    clone_inventory = database_inventory(work / "database.db")
    if clone_inventory != source_record["database_inventory"]:
        raise RuntimeError("prepared database clone changed before global-solve")
    free_bytes = shutil.disk_usage(work).free
    runtime_required = int(
        settings["minimum_runtime_free_space_gb"] * (1024 ** 3))
    if free_bytes < runtime_required:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.1f} GiB free before solve; "
            f"require {settings['minimum_runtime_free_space_gb']:.1f} GiB")

    executable = _resolve_colmap(cfg)
    sparse_incomplete = work / "sparse_global.incomplete"
    sparse_incomplete.mkdir()
    command = global_mapper_command(
        work, cfg, executable, sparse_incomplete)
    _atomic_json(sidecar / "commands.json", [{
        "step": "global_mapper",
        "argv": command,
        "features_recomputed": False,
        "matches_recomputed": False,
    }])
    log_path = work / "global.log"
    print(
        "COLMAP Global Mapper: starting with fixed intrinsics/rig, CPU "
        f"optimization, {settings['num_threads']} threads", flush=True)
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        try:
            resources = _run_monitored(
                command, log, work / "resource_samples.csv", settings)
        except _MonitoredProcessInterrupted as exc:
            _atomic_json(work / "resource_usage.json", exc.usage)
            raise RuntimeError(str(exc)) from exc
        _atomic_json(work / "resource_usage.json", resources)
        if resources["returncode"]:
            reason = resources["safety_abort_reason"] or \
                f"exit code {resources['returncode']}"
            raise RuntimeError(
                f"Global Mapper failed ({reason}); see {log_path}")

        candidate_model, candidate_stats = _analyze_models(
            executable, sparse_incomplete, log)
        source = _source_sidecar(seg_dir, settings)
        reference_model = _find_source_model(source, settings)
        reference_stats = _analyze_model(executable, reference_model, log)

        text_incomplete = work / "models_text.incomplete"
        text_model = text_incomplete / candidate_model.name
        _convert_model(executable, candidate_model, text_model, log)
        n_frames = int(source_record["n_frames"])
        baseline = float(json.loads(
            (source / "sidecar_config.json").read_text()
        )["baseline_m_camera_info"])
        structure = inspect_rig_model(
            text_model, work / "database.db",
            source_record["database_inventory"], n_frames, baseline)
        checks = _structural_gates(
            candidate_stats, reference_stats, structure, settings, n_frames)
        structural_passed = all(
            item["passed"] for item in checks.values()
            if item.get("enforced", True))
        solve_result = {
            "schema_version": 1,
            "backend": "COLMAP 4.1.1 integrated Global Mapper (GLOMAP)",
            "command": command,
            "candidate_model_name": candidate_model.name,
            "candidate_model_stats": candidate_stats,
            "reference_incremental_model_stats": reference_stats,
            "rig_structure": structure,
            "structural_gates": checks,
            "structural_gates_passed": structural_passed,
            "resources": resources,
            "source_database_sha256_before":
                source_record["database_sha256"],
            "source_database_sha256_after": _sha256_file(source_database),
            "source_artifacts_modified": False,
        }
        _atomic_json(work / "solve_result.json", solve_result)
        if (solve_result["source_database_sha256_after"]
                != source_record["database_sha256"]):
            raise RuntimeError("source database changed during global mapping")
        if not structural_passed:
            failed = [
                name for name, item in checks.items()
                if item.get("enforced", True) and not item["passed"]]
            raise RuntimeError(
                "Global Mapper output failed structural gates: "
                + ", ".join(failed))

    os.rename(sparse_incomplete, work / "sparse_global")
    os.rename(work / "models_text.incomplete", work / "models_text")
    solve_result["candidate_model_stats"]["path"] = str(
        work / "sparse_global" / solve_result["candidate_model_name"])
    _atomic_json(work / "solve_result.json", solve_result)
    _atomic_json(done, solve_result)
    print(
        f"Global Mapper solve passed structural gates in "
        f"{resources['wall_time_s']/60:.1f} min; peak RSS "
        f"{resources['peak_process_rss_bytes']/1024**3:.1f} GiB")
    return solve_result


def rigid_alignment(
        source: np.ndarray,
        target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-scale SE(3) least squares: target ~= R @ source + t."""
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("SE(3) alignment needs matching (N,3), N >= 3")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    if np.linalg.matrix_rank(xc, tol=1.0e-8) < 2:
        raise ValueError("camera trajectory is too close to a line for SE(3)")
    covariance = (yc.T @ xc) / len(x)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    translation = my - rotation @ mx
    return rotation, translation


def robust_rigid_alignment(
        source: np.ndarray,
        target: np.ndarray,
        max_error_m: float,
        iterations: int,
        seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC and inlier refit for a metric, fixed-scale alignment."""
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if len(x) < 3:
        raise ValueError("at least three pose correspondences are required")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(iterations)):
        indices = rng.choice(len(x), 3, replace=False)
        try:
            rotation, translation = rigid_alignment(x[indices], y[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = np.linalg.norm(
            (rotation @ x.T).T + translation - y, axis=1)
        inliers = errors <= max_error_m
        score = (
            int(inliers.sum()),
            -float(np.median(errors[inliers]))
            if inliers.any() else -math.inf,
        )
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[1].sum() < 3:
        raise RuntimeError("RANSAC could not find a fixed-scale RTK alignment")
    inliers = best[1]
    for _ in range(3):
        rotation, translation = rigid_alignment(x[inliers], y[inliers])
        errors = np.linalg.norm(
            (rotation @ x.T).T + translation - y, axis=1)
        updated = errors <= max_error_m
        if np.array_equal(updated, inliers) or updated.sum() < 3:
            break
        inliers = updated
    rotation, translation = rigid_alignment(x[inliers], y[inliers])
    errors = np.linalg.norm(
        (rotation @ x.T).T + translation - y, axis=1)
    return rotation, translation, inliers, errors


def _error_stats(errors: np.ndarray, inliers: np.ndarray | None = None) -> dict:
    values = np.asarray(errors, dtype=float)
    result = {
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(values.max()),
    }
    if inliers is not None:
        result["n_inliers"] = int(np.asarray(inliers).sum())
        result["inlier_fraction"] = float(np.asarray(inliers).mean())
    return result


def _rotation_step_degrees(rotations: np.ndarray) -> np.ndarray:
    result = []
    for previous, current in zip(rotations[:-1], rotations[1:]):
        relative = current @ previous.T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        result.append(np.degrees(np.arccos(cosine)))
    return np.asarray(result)


def _trajectory_comparison(
        candidate_viewmats: np.ndarray,
        candidate_centers: np.ndarray,
        reference_viewmats: np.ndarray,
        reference_centers: np.ndarray,
) -> dict:
    center_errors = np.linalg.norm(
        candidate_centers - reference_centers, axis=1)
    rotation_errors = []
    for candidate, reference in zip(candidate_viewmats, reference_viewmats):
        relative = candidate[:3, :3] @ reference[:3, :3].T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(np.degrees(np.arccos(cosine)))
    rotation_errors = np.asarray(rotation_errors)
    return {
        "position_difference_m": {
            "median": float(np.median(center_errors)),
            "p95": float(np.percentile(center_errors, 95)),
            "max": float(center_errors.max()),
        },
        "rotation_difference_deg": {
            "median": float(np.median(rotation_errors)),
            "p95": float(np.percentile(rotation_errors, 95)),
            "max": float(rotation_errors.max()),
        },
        "manual_review_recommended": bool(
            np.median(center_errors) > 0.01
            or np.percentile(center_errors, 95) > 0.03
            or np.median(rotation_errors) > 0.1),
    }


def export(seg_dir: Path, cfg) -> dict:
    """Georegister at fixed metric scale and publish only a gate-passing pose."""
    settings = _settings(cfg)
    sidecar = _sidecar_dir(seg_dir, cfg)
    work = sidecar / "global"
    done_path = work / ".global_mapper.done.json"
    if not done_path.is_file():
        raise FileNotFoundError("run global-solve before global-export")
    quality_path = sidecar / "quality.json"
    if quality_path.exists():
        existing = json.loads(quality_path.read_text())
        required = (
            sidecar / "viewmats.npy", sidecar / "cam_centers.npy",
            sidecar / "frame_ids.npy", sidecar / "registered.npy")
        if existing.get("accepted_for_gs") and all(
                path.is_file() for path in required):
            print(f"Global Mapper export already complete: {sidecar}")
            return existing
        raise FileExistsError(
            f"refusing to overwrite failed or incomplete export: {quality_path}")
    solved = json.loads(done_path.read_text())
    model_name = solved["candidate_model_name"]
    model_text = work / "models_text" / model_name
    source_record = _verify_prepared_provenance(
        sidecar, settings, verify_database=False)
    n_frames = int(source_record["n_frames"])
    trajectory = load_raw_colmap_rig_trajectory(
        model_text, work / "database.db", expected_frame_count=n_frames)
    rtk_snapshot = sidecar / "rtk_camera_centers.npy"
    if (_sha256_file(rtk_snapshot)
            != source_record["rtk_centers_snapshot_sha256"]):
        raise RuntimeError("prepared RTK camera-centre snapshot changed")
    raw_centers = np.load(rtk_snapshot).astype(float)
    threshold = settings["alignment_max_error_m"]
    iterations = settings["alignment_ransac_iterations"]

    rotation, translation, inliers, rigid_errors = robust_rigid_alignment(
        trajectory.camera_centers_visual, raw_centers,
        threshold, iterations, settings["random_seed"])
    sim_scale, sim_rotation, sim_translation, sim_inliers, sim_errors = \
        robust_similarity(
            trajectory.camera_centers_visual, raw_centers,
            threshold, iterations, settings["random_seed"])

    viewmats = []
    centers = []
    for camera_from_world, center in zip(
            trajectory.camera_from_visual_world,
            trajectory.camera_centers_visual):
        viewmat, aligned_center = aligned_viewmat(
            camera_from_world[:3, :3], center,
            1.0, rotation, translation)
        viewmats.append(viewmat)
        centers.append(aligned_center)
    viewmats = np.asarray(viewmats, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)

    source = _source_sidecar(seg_dir, settings)
    source_quality = json.loads((source / "quality.json").read_text())
    source_model_text = Path(source_record["source_text_model"])
    if not source_model_text.is_dir():
        raise FileNotFoundError(
            f"source incremental text model is missing: {source_model_text}")
    source_trajectory = load_raw_colmap_rig_trajectory(
        source_model_text,
        source / settings["source_workspace_subdir"] / "database.db",
        expected_frame_count=n_frames)
    source_rotation, source_translation, source_fixed_inliers, \
        source_fixed_errors = robust_rigid_alignment(
            source_trajectory.camera_centers_visual, raw_centers,
            threshold, iterations, settings["random_seed"])
    source_fixed_viewmats = []
    source_fixed_centers = []
    for camera_from_world, center in zip(
            source_trajectory.camera_from_visual_world,
            source_trajectory.camera_centers_visual):
        viewmat, aligned_center = aligned_viewmat(
            camera_from_world[:3, :3], center,
            1.0, source_rotation, source_translation)
        source_fixed_viewmats.append(viewmat)
        source_fixed_centers.append(aligned_center)
    source_fixed_viewmats = np.asarray(source_fixed_viewmats)
    source_fixed_centers = np.asarray(source_fixed_centers)
    legacy_viewmats = np.load(source / "viewmats.npy").astype(float)
    legacy_centers = np.load(source / "cam_centers.npy").astype(float)
    continuity_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    continuity_rotations = _rotation_step_degrees(viewmats[:, :3, :3])
    comparison_fixed = _trajectory_comparison(
        viewmats, centers, source_fixed_viewmats, source_fixed_centers)
    comparison_legacy = _trajectory_comparison(
        viewmats, centers, legacy_viewmats, legacy_centers)

    gates = settings["gates"]
    scale_range = settings["scale_range"]
    reference_fixed = _error_stats(
        source_fixed_errors, source_fixed_inliers)
    median_limit = min(
        gates["max_rtk_median_error_m"],
        reference_fixed["median_m"]
        * (1.0 + gates["reference_regression_fraction"]))
    p95_limit = min(
        gates["max_rtk_p95_error_m"],
        reference_fixed["p95_m"]
        * (1.0 + gates["reference_regression_fraction"]))
    rigid_stats = _error_stats(rigid_errors, inliers)
    sim_stats = _error_stats(sim_errors, sim_inliers)
    checks = dict(solved["structural_gates"])
    checks.update({
        "sim3_scale_diagnostic": {
            "value": sim_scale,
            "range": scale_range,
            "passed": scale_range[0] <= sim_scale <= scale_range[1],
            "enforced": True,
        },
        "fixed_scale_rtk_median_m": {
            "value": rigid_stats["median_m"],
            "maximum": median_limit,
            "passed": rigid_stats["median_m"] <= median_limit,
            "enforced": False,
            "reason": "rough TF and same-trajectory fit; result target, not integrity",
        },
        "fixed_scale_rtk_p95_m": {
            "value": rigid_stats["p95_m"],
            "maximum": p95_limit,
            "passed": rigid_stats["p95_m"] <= p95_limit,
            "enforced": False,
            "reason": "rough TF and same-trajectory fit; result target, not integrity",
        },
        "fixed_scale_rtk_max_m": {
            "value": rigid_stats["max_m"],
            "maximum": gates["max_rtk_error_m"],
            "passed": rigid_stats["max_m"] <= gates["max_rtk_error_m"],
            "enforced": False,
            "reason": "single-sample maximum is reported, not a solver-validity test",
        },
        "fixed_scale_alignment_inlier_fraction": {
            "value": rigid_stats["inlier_fraction"],
            "minimum": gates["min_alignment_inlier_fraction"],
            "passed": rigid_stats["inlier_fraction"]
                      >= gates["min_alignment_inlier_fraction"],
            "enforced": False,
            "reason": "RTK target pending held-out covariance-aware evaluation",
        },
        "maximum_translation_step_m": {
            "value": float(continuity_steps.max()),
            "maximum": gates["max_step_m"],
            "passed": float(continuity_steps.max()) <= gates["max_step_m"],
            "enforced": True,
        },
        "maximum_rotation_step_deg": {
            "value": float(continuity_rotations.max()),
            "maximum": gates["max_rotation_step_deg"],
            "passed": float(continuity_rotations.max())
                      <= gates["max_rotation_step_deg"],
            "enforced": True,
        },
    })
    accepted = all(
        item["passed"] for item in checks.values()
        if item.get("enforced", True))
    diagnostic_targets_passed = all(
        item["passed"] for item in checks.values()
        if not item.get("enforced", True))
    quality = {
        "schema_version": 1,
        "source": "COLMAP 4.1.1 integrated Global Mapper, calibrated stereo",
        "n_frames": n_frames,
        "n_registered_left": len(trajectory.frame_indices),
        "n_registered_total":
            solved["candidate_model_stats"]["registered_images"],
        "fixed_scale_alignment": {
            "method": "SE(3) RANSAC to immutable RTK camera centres",
            "scale_applied": 1.0,
            "rotation_visual_world_to_enu": rotation.tolist(),
            "translation_enu_m": translation.tolist(),
            **rigid_stats,
        },
        "sim3_diagnostic_only": {
            "scale": sim_scale,
            "rotation_visual_world_to_enu": sim_rotation.tolist(),
            "translation_enu_m": sim_translation.tolist(),
            **sim_stats,
            "applied_to_export": False,
        },
        "reference_incremental_fixed_scale_alignment": reference_fixed,
        "reference_incremental_legacy_sim3_alignment":
            source_quality["alignment"],
        "continuity": {
            "translation_step_median":
                float(np.median(continuity_steps)),
            "translation_step_p95":
                float(np.percentile(continuity_steps, 95)),
            "translation_step_max": float(continuity_steps.max()),
            "rotation_step_deg_median":
                float(np.median(continuity_rotations)),
            "rotation_step_deg_p95":
                float(np.percentile(continuity_rotations, 95)),
            "rotation_step_deg_max": float(continuity_rotations.max()),
        },
        "comparison_to_incremental_fixed_scale": comparison_fixed,
        "comparison_to_incremental_legacy_sim3": comparison_legacy,
        "model_stats": solved["candidate_model_stats"],
        "reference_incremental_model_stats":
            solved["reference_incremental_model_stats"],
        "rig_structure": solved["rig_structure"],
        "resources": solved["resources"],
        "quality_gates": checks,
        "accepted_for_gs": accepted,
        "diagnostic_targets_passed": diagnostic_targets_passed,
        "metric_scale_preserved": True,
        "raw_rtk_poses_modified": False,
        "source_artifacts_modified": False,
        "pose_fingerprint":
            pose_fingerprint(viewmats) if accepted else None,
    }

    # Candidate arrays are retained for diagnosis, but standard pose names are
    # published only if every gate passes. This prevents cloud/train from
    # accidentally consuming a failed reconstruction. quality.json is the
    # completion marker and is deliberately written last for accepted exports.
    _atomic_save_npy(sidecar / "candidate_viewmats.npy", viewmats)
    _atomic_save_npy(sidecar / "candidate_cam_centers.npy", centers)
    if not accepted:
        _atomic_json(quality_path, quality)
        failed = [
            name for name, item in checks.items()
            if item.get("enforced", True) and not item["passed"]]
        raise RuntimeError(
            "Global Mapper pose failed publication gates: "
            + ", ".join(failed))
    _atomic_save_npy(sidecar / "viewmats.npy", viewmats)
    _atomic_save_npy(sidecar / "cam_centers.npy", centers)
    _atomic_save_npy(
        sidecar / "frame_ids.npy",
        trajectory.frame_indices.astype(np.int64))
    _atomic_save_npy(
        sidecar / "registered.npy",
        np.ones(n_frames, dtype=np.bool_))
    _atomic_json(quality_path, quality)
    print(
        f"published {n_frames} fixed-scale Global Mapper poses -> {sidecar}")
    print(
        f"metric RTK residual median {rigid_stats['median_m']*100:.1f} cm, "
        f"p95 {rigid_stats['p95_m']*100:.1f} cm; Sim(3) diagnostic scale "
        f"{sim_scale:.6f}; integrity gates passed, diagnostic targets "
        f"{'passed' if diagnostic_targets_passed else 'need review'}")
    return quality
