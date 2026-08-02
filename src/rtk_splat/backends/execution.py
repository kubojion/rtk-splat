"""Subprocess execution and resource monitoring shared by backends."""

from __future__ import annotations

import csv
import math
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rtk_splat.backends.artifact_io import _atomic_json
from rtk_splat.backends.mapper_config import MapperConfig
from rtk_splat.frontends.artifact import ArtifactError


Runner = Callable[..., Any]

def _execute(command: Sequence[str], runner: Runner, *, capture: bool = False) -> Any:
    kwargs: dict[str, Any] = {"check": True}
    if capture:
        kwargs.update(
            {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
        )
    return runner(list(command), **kwargs)


def _meminfo_bytes() -> dict[str, int]:
    """Read Linux memory counters without invoking another process."""
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError("the mapper resource guard requires /proc/meminfo") from exc
    result: dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        fields = raw.strip().split()
        if fields:
            result[name] = int(fields[0]) * 1024
    if "MemAvailable" not in result:
        raise ArtifactError("/proc/meminfo has no MemAvailable counter")
    return result


def _process_memory_bytes(pid: int) -> tuple[int, int]:
    """Return resident and high-water memory for one threaded COLMAP process."""
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text(
            encoding="utf-8"
        ).splitlines()
    except FileNotFoundError:
        return 0, 0
    except OSError:
        return 0, 0
    values = {"VmRSS": 0, "VmHWM": 0}
    for line in lines:
        name = line.split(":", 1)[0]
        if name in values:
            values[name] = int(line.split()[1]) * 1024
    return values["VmRSS"], values["VmHWM"]


def _terminate_process_group(
    process: subprocess.Popen[Any], log, reason: str
) -> None:
    """Terminate the isolated mapper process group, escalating if necessary."""
    log.write(f"\nRESOURCE GUARD: {reason}\n")
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
    def __init__(self, message: str, usage: Mapping[str, Any]):
        super().__init__(message)
        self.usage = dict(usage)


def _next_solve_attempt(root: Path) -> Path:
    attempts = root / "attempts" / "solve"
    attempts.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10_000):
        candidate = attempts / f"attempt-{index:04d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise ArtifactError("too many mapper solve attempts")


def _resource_sample(
    root: Path, pid: int, elapsed_s: float
) -> dict[str, int | float]:
    rss, hwm = _process_memory_bytes(pid)
    memory = _meminfo_bytes()
    return {
        "elapsed_s": elapsed_s,
        "pid": pid,
        "process_rss_bytes": rss,
        "process_hwm_bytes": hwm,
        "mem_available_bytes": memory["MemAvailable"],
        "swap_free_bytes": memory.get("SwapFree", 0),
        "disk_free_bytes": shutil.disk_usage(root).free,
    }


def _run_monitored_mapper(
    command: Sequence[str], attempt: Path, config: MapperConfig
) -> dict[str, Any]:
    """Run Global Mapper in its own process group with Linux safety guards."""
    csv_path = attempt / "resource_samples.csv"
    usage_path = attempt / "resource_usage.json"
    log_path = attempt / "colmap.log"
    started_unix_s = time.time()
    started = time.monotonic()
    initial_free = shutil.disk_usage(attempt).free
    initial_floor = int(config.minimum_free_space_gb * 1024**3)
    if initial_free < initial_floor:
        usage = {
            "schema_version": 1,
            "monitor": "linux-process-group",
            "launched": False,
            "returncode": None,
            "safety_aborted": True,
            "safety_abort_reason": (
                "free disk is below the configured launch floor of "
                f"{config.minimum_free_space_gb:.1f} GiB"
            ),
            "started_unix_s": started_unix_s,
            "ended_unix_s": time.time(),
            "wall_time_s": 0.0,
            "peak_process_rss_bytes": 0,
            "minimum_system_available_bytes": _meminfo_bytes()["MemAvailable"],
            "minimum_disk_free_bytes": initial_free,
            "samples_csv": str(csv_path.relative_to(attempt.parents[2])),
            "log": str(log_path.relative_to(attempt.parents[2])),
        }
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                (
                    "elapsed_s",
                    "pid",
                    "process_rss_bytes",
                    "process_hwm_bytes",
                    "mem_available_bytes",
                    "swap_free_bytes",
                    "disk_free_bytes",
                )
            )
        _atomic_json(usage_path, usage)
        raise _MonitoredProcessInterrupted(usage["safety_abort_reason"], usage)

    def set_nice() -> None:
        if config.process_nice:
            os.nice(config.process_nice)

    peak_rss = 0
    minimum_available = math.inf
    minimum_disk = math.inf
    low_memory_samples = 0
    abort_reason: str | None = None
    interrupted: BaseException | None = None
    returncode: int | None = None
    process: subprocess.Popen[Any] | None = None
    old_handlers: dict[int, Any] = {}

    def interrupt(signum, _frame) -> None:
        raise InterruptedError(f"received signal {signum}")

    with log_path.open("x", encoding="utf-8") as log, csv_path.open(
        "x", newline="", encoding="utf-8"
    ) as sample_stream:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        writer = csv.DictWriter(
            sample_stream,
            fieldnames=(
                "elapsed_s",
                "pid",
                "process_rss_bytes",
                "process_hwm_bytes",
                "mem_available_bytes",
                "swap_free_bytes",
                "disk_free_bytes",
            ),
        )
        writer.writeheader()
        process = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            preexec_fn=set_nice,
        )
        monitored_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            monitored_signals.append(signal.SIGHUP)
        try:
            for signum in monitored_signals:
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupt)
            while True:
                sample = _resource_sample(
                    attempt, process.pid, time.monotonic() - started
                )
                writer.writerow(sample)
                sample_stream.flush()
                peak_rss = max(
                    peak_rss,
                    int(sample["process_rss_bytes"]),
                    int(sample["process_hwm_bytes"]),
                )
                available = int(sample["mem_available_bytes"])
                disk_free = int(sample["disk_free_bytes"])
                minimum_available = min(minimum_available, available)
                minimum_disk = min(minimum_disk, disk_free)
                returncode = process.poll()
                if returncode is not None:
                    break
                memory_floor = int(config.minimum_available_memory_gb * 1024**3)
                low_memory_samples = (
                    low_memory_samples + 1 if available < memory_floor else 0
                )
                if low_memory_samples >= config.low_memory_consecutive_samples:
                    abort_reason = (
                        "MemAvailable stayed below "
                        f"{config.minimum_available_memory_gb:.1f} GiB for "
                        f"{low_memory_samples} consecutive samples"
                    )
                    _terminate_process_group(process, log, abort_reason)
                    returncode = process.wait()
                    break
                runtime_floor = int(
                    config.minimum_runtime_free_space_gb * 1024**3
                )
                if disk_free < runtime_floor:
                    abort_reason = (
                        "free disk fell below the configured runtime floor of "
                        f"{config.minimum_runtime_free_space_gb:.1f} GiB"
                    )
                    _terminate_process_group(process, log, abort_reason)
                    returncode = process.wait()
                    break
                time.sleep(config.resource_sample_interval_s)
        except BaseException as exc:
            interrupted = exc
            if process.poll() is None:
                _terminate_process_group(
                    process,
                    log,
                    f"monitor interrupted by {type(exc).__name__}: {exc}",
                )
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if process.poll() is None:
                _terminate_process_group(
                    process, log, "monitor exited while COLMAP was still running"
                )
            returncode = process.wait()

    ended_unix_s = time.time()
    usage = {
        "schema_version": 1,
        "monitor": "linux-process-group",
        "launched": True,
        "returncode": int(returncode),
        "safety_aborted": abort_reason is not None,
        "safety_abort_reason": abort_reason,
        "started_unix_s": started_unix_s,
        "ended_unix_s": ended_unix_s,
        "wall_time_s": time.monotonic() - started,
        "peak_process_rss_bytes": int(peak_rss),
        "minimum_system_available_bytes": int(
            minimum_available if math.isfinite(minimum_available) else 0
        ),
        "minimum_disk_free_bytes": int(
            minimum_disk if math.isfinite(minimum_disk) else initial_free
        ),
        "samples_csv": str(csv_path.relative_to(attempt.parents[2])),
        "log": str(log_path.relative_to(attempt.parents[2])),
    }
    _atomic_json(usage_path, usage)
    if interrupted is not None:
        raise _MonitoredProcessInterrupted(
            f"Global Mapper monitor was interrupted: {interrupted}", usage
        ) from interrupted
    if abort_reason is not None:
        raise _MonitoredProcessInterrupted(abort_reason, usage)
    if returncode:
        raise subprocess.CalledProcessError(returncode, list(command))
    return usage


def _run_injected_mapper(
    command: Sequence[str], attempt: Path, runner: Runner
) -> dict[str, Any]:
    """Keep deterministic evidence when a unit-test runner replaces COLMAP."""
    csv_path = attempt / "resource_samples.csv"
    usage_path = attempt / "resource_usage.json"
    started = time.time()
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("elapsed_s", "pid", "process_rss_bytes"))
        writer.writerow(("0.000", os.getpid(), 0))
    try:
        _execute(command, runner)
    except BaseException as exc:
        usage = {
            "schema_version": 1,
            "monitor": "injected-runner",
            "returncode": None,
            "exception": type(exc).__name__,
            "wall_time_s": time.time() - started,
            "samples_csv": str(csv_path.relative_to(attempt.parents[2])),
        }
        _atomic_json(usage_path, usage)
        raise
    usage = {
        "schema_version": 1,
        "monitor": "injected-runner",
        "returncode": 0,
        "wall_time_s": time.time() - started,
        "samples_csv": str(csv_path.relative_to(attempt.parents[2])),
    }
    _atomic_json(usage_path, usage)
    return usage


def _solve_resource_paths(root: Path, usage: Mapping[str, Any]) -> tuple[Path, Path]:
    samples = root / str(usage["samples_csv"])
    return samples, samples.with_name("resource_usage.json")

