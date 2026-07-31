"""Dataset-facing utilities for producing canonical RTK Splat segments.

This package deliberately has no ROS or dataset imports at module import time.
Concrete adapters may load optional dependencies inside their entry points.
"""

from .image_decode import (
    ImageDecodeError,
    decode_compressed_image,
    decode_raw_image,
    detect_compressed_format,
)
from .synchronization import (
    TimestampMatchError,
    TimestampMatches,
    associate_timestamps,
    monotonic_matches,
    nearest_matches,
)
from .registry import ADAPTER_NAMES, publish_from_config

__all__ = [
    "ImageDecodeError",
    "ADAPTER_NAMES",
    "TimestampMatchError",
    "TimestampMatches",
    "associate_timestamps",
    "decode_compressed_image",
    "decode_raw_image",
    "detect_compressed_format",
    "monotonic_matches",
    "nearest_matches",
    "publish_from_config",
]
