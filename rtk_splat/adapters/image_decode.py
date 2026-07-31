"""ROS-independent decoding of raw and compressed image payloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BytePayload = bytes | bytearray | memoryview | np.ndarray


class ImageDecodeError(ValueError):
    """Raised when image metadata or payload bytes are inconsistent."""


@dataclass(frozen=True)
class _RawEncoding:
    dtype: str
    channels: int


_RAW_ENCODINGS = {
    "mono8": _RawEncoding("u1", 1),
    "8uc1": _RawEncoding("u1", 1),
    "8uc3": _RawEncoding("u1", 3),
    "8uc4": _RawEncoding("u1", 4),
    "rgb8": _RawEncoding("u1", 3),
    "bgr8": _RawEncoding("u1", 3),
    "rgba8": _RawEncoding("u1", 4),
    "bgra8": _RawEncoding("u1", 4),
    "mono16": _RawEncoding("u2", 1),
    "16uc1": _RawEncoding("u2", 1),
    "16sc1": _RawEncoding("i2", 1),
    "32fc1": _RawEncoding("f4", 1),
    "32fc3": _RawEncoding("f4", 3),
}


def _byte_array(payload: BytePayload, name: str) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        if payload.dtype != np.uint8 or payload.ndim != 1:
            raise ImageDecodeError(
                f"{name} numpy payload must be a one-dimensional uint8 array"
            )
        return np.ascontiguousarray(payload)
    try:
        view = memoryview(payload)
    except TypeError as exc:
        raise ImageDecodeError(
            f"{name} must be bytes-like or a one-dimensional uint8 array"
        ) from exc
    if view.ndim != 1 or view.itemsize != 1:
        try:
            view = view.cast("B")
        except (TypeError, ValueError) as exc:
            raise ImageDecodeError(f"{name} is not a contiguous byte buffer") from exc
    return np.frombuffer(view, dtype=np.uint8)


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ImageDecodeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ImageDecodeError(f"{name} must be a positive integer")
    return result


def decode_raw_image(
    payload: BytePayload,
    *,
    width: int,
    height: int,
    encoding: str,
    step: int | None = None,
    is_bigendian: bool | int = False,
) -> np.ndarray:
    """Decode a raw image buffer using explicit ROS-style image metadata.

    Row padding declared by ``step`` is removed.  Channel order is preserved:
    for example, ``rgb8`` returns RGB while ``bgr8`` returns BGR.  Multi-byte
    input is converted to native byte order in the returned array.
    """
    image_width = _positive_dimension(width, "width")
    image_height = _positive_dimension(height, "height")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ImageDecodeError("encoding must be a non-empty string")
    normalized = encoding.strip().lower()
    try:
        specification = _RAW_ENCODINGS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_RAW_ENCODINGS))
        raise ImageDecodeError(
            f"unsupported raw image encoding {encoding!r}; supported: {supported}"
        ) from exc

    native_dtype = np.dtype(specification.dtype)
    row_bytes = image_width * specification.channels * native_dtype.itemsize
    row_step = row_bytes if step is None else _positive_dimension(step, "step")
    if row_step < row_bytes:
        raise ImageDecodeError(
            f"step {row_step} is smaller than the encoded row size {row_bytes}"
        )
    data = _byte_array(payload, "raw image")
    expected_bytes = image_height * row_step
    if data.size != expected_bytes:
        raise ImageDecodeError(
            f"raw image has {data.size} bytes; metadata requires {expected_bytes}"
        )
    if is_bigendian not in (False, True, 0, 1):
        raise ImageDecodeError("is_bigendian must be boolean or 0/1")

    rows = data.reshape(image_height, row_step)
    packed = np.ascontiguousarray(rows[:, :row_bytes]).reshape(-1)
    if native_dtype.itemsize > 1:
        byte_order = ">" if bool(is_bigendian) else "<"
        wire_dtype = native_dtype.newbyteorder(byte_order)
    else:
        wire_dtype = native_dtype
    image = packed.view(wire_dtype)
    shape = (image_height, image_width)
    if specification.channels > 1:
        shape += (specification.channels,)
    image = image.reshape(shape)
    if image.dtype != native_dtype:
        image = image.astype(native_dtype)
    return np.ascontiguousarray(image)


def detect_compressed_format(payload: BytePayload) -> str:
    """Identify a JPEG or PNG payload from its signature."""
    data = _byte_array(payload, "compressed image")
    if data.size >= 3 and bytes(data[:3]) == b"\xff\xd8\xff":
        return "jpeg"
    if data.size >= 8 and bytes(data[:8]) == b"\x89PNG\r\n\x1a\n":
        return "png"
    raise ImageDecodeError("compressed image is neither a JPEG nor a PNG")


def _normalize_format_hint(format_hint: str) -> str:
    if not isinstance(format_hint, str) or not format_hint.strip():
        raise ImageDecodeError("format_hint must be a non-empty string")
    normalized = format_hint.lower()
    has_jpeg = "jpeg" in normalized or "jpg" in normalized
    has_png = "png" in normalized
    if has_jpeg == has_png:
        raise ImageDecodeError(
            f"format_hint {format_hint!r} does not identify exactly one of JPEG/PNG"
        )
    return "jpeg" if has_jpeg else "png"


def decode_compressed_image(
    payload: BytePayload,
    *,
    format_hint: str | None = None,
    color_order: str = "bgr",
) -> np.ndarray:
    """Decode a JPEG/PNG payload with validated format and output order.

    OpenCV's native BGR/BGRA output is returned by default.  Set
    ``color_order="rgb"`` to convert color images to RGB/RGBA; one-channel
    depth images remain unchanged.  OpenCV is imported lazily so importing an
    adapter does not load an imaging or ROS runtime.
    """
    data = _byte_array(payload, "compressed image")
    detected_format = detect_compressed_format(data)
    if format_hint is not None:
        expected_format = _normalize_format_hint(format_hint)
        if expected_format != detected_format:
            raise ImageDecodeError(
                f"format_hint says {expected_format}, payload is {detected_format}"
            )
    if color_order not in {"bgr", "rgb"}:
        raise ImageDecodeError("color_order must be 'bgr' or 'rgb'")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImageDecodeError(
            "decoding compressed images requires OpenCV (cv2)"
        ) from exc
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ImageDecodeError(f"OpenCV could not decode the {detected_format} image")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ImageDecodeError(
            f"decoded image has unsupported shape {tuple(image.shape)}"
        )
    if color_order == "rgb" and image.ndim == 3:
        if image.shape[2] == 3:
            image = image[..., [2, 1, 0]]
        else:
            image = image[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(image)
