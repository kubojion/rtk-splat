"""Lazy config-to-contract adapter dispatch.

Dataset dependencies stay outside the core package and are imported only for
the adapter selected by the robot/sequence configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rtk_splat.segment import SegmentReader


ADAPTER_NAMES = ("agrigs", "ros2_zed_ublox")


def publish_from_config(
    cfg,
    destination: str | Path,
    *,
    selection: Mapping[str, Any] | None = None,
) -> SegmentReader:
    """Publish one immutable contract-v2 segment with the selected adapter."""
    name = str(getattr(cfg, "adapter", "")).strip()
    if name == "ros2_zed_ublox":
        from .ros2_zed_ublox import ingest_config_v2

        return ingest_config_v2(cfg, destination, window=selection)
    if name == "agrigs":
        if selection is not None:
            raise ValueError("the AgriGS adapter does not consume ROS time windows")
        from .agrigs import ingest_config_v2

        return ingest_config_v2(cfg, destination)
    raise ValueError(
        f"unknown adapter {name!r}; choose one of {', '.join(ADAPTER_NAMES)}"
    )
