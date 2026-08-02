"""Lazy config-to-contract adapter dispatch.

Dataset dependencies stay outside the core package and are imported only for
the adapter selected by the robot/sequence configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rtk_splat.core.segment import SegmentReader


ADAPTER_NAMES = ("agrigs", "ros1_citrusfarm", "ros2_zed_ublox")


def _reject_unsupported_adapter_options(cfg, name: str) -> None:
    """Fail closed until a built-in adapter exposes an option validator."""
    options = getattr(cfg, "adapter_options", None)
    if options is None:
        return
    if isinstance(options, Mapping):
        authored = dict(options)
    elif hasattr(options, "__dict__"):
        authored = vars(options)
    else:
        raise ValueError("adapter_options must be a mapping")
    if authored:
        raise ValueError(
            f"adapter {name!r} does not declare any adapter_options; "
            "remove them or add an adapter-owned validator"
        )


def publish_from_config(
    cfg,
    destination: str | Path,
    *,
    selection: Mapping[str, Any] | None = None,
) -> SegmentReader:
    """Publish one immutable contract-v2 segment with the selected adapter."""
    name = str(getattr(cfg, "adapter", "")).strip()
    _reject_unsupported_adapter_options(cfg, name)
    if name == "ros2_zed_ublox":
        from .ros2_zed_ublox import ingest_config_v2

        return ingest_config_v2(cfg, destination, window=selection)
    if name == "ros1_citrusfarm":
        from .ros1_citrusfarm import ingest_config_v2

        return ingest_config_v2(cfg, destination, window=selection)
    if name == "agrigs":
        if selection is not None:
            raise ValueError("the AgriGS adapter does not consume ROS time windows")
        from .agrigs import ingest_config_v2

        return ingest_config_v2(cfg, destination)
    raise ValueError(
        f"unknown adapter {name!r}; choose one of {', '.join(ADAPTER_NAMES)}"
    )
