#!/usr/bin/env bash
# Encode a contract-v2 segment's left camera into a timelapse.
#
# Every stored frame is kept, at its native resolution, played back at a
# constant 30 fps. The source capture rate therefore sets the speed-up: a 3 Hz
# segment plays 10x real time, a 10 Hz segment plays 3x.
#
#   bash scripts/tools/segment_timelapse.sh                       # the default set
#   bash scripts/tools/segment_timelapse.sh label=/path/to/segment ...
#
# Each argument is label=path, optionally label=path#prefix to pick a stream
# other than the left camera (for example #rgb for an RGB observations
# artifact). Any directory holding images/<prefix>_%06d.<ext> and a frames.npz
# with timestamp_ns works; the stereo contract is not required.
#
# Overlay: sequence label, exact frame index, and elapsed source seconds.
# The clock is a linear map from frame index, so it drifts wherever the
# recorder dropped frames; the script measures that drift and prints it, and
# the frame index is always exact -- it indexes images/<prefix>_NNNNNN.<ext>
# directly, which is the handle you want when something looks wrong.
#
# Overridable: OUTPUT_DIR, FPS, CRF, PRESET.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/agromap4d_work/timelapses}"
FPS="${FPS:-30}"
CRF="${CRF:-20}"
PRESET="${PRESET:-medium}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}"
PYTHON="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"
MIN_FREE_GIB="${MIN_FREE_GIB:-5}"

DEFAULT_SEGMENTS=(
    "headland_zed_rgb=$HOME/agromap4d_work/field_turn_contract_v2_normalized/segment"
    "citrusfarm_retrace_rgb=$HOME/agromap4d_work/citrusfarm_05_13d_330_410_row_retrace_v1/segments/citrus-05-13d-330-410-row-retrace-v1-rgb"
    "rosario_seq5_ir=$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment"
    "rosario_seq5_rgb=$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment.rgb_observations#rgb"
)

die() { echo "FATAL: $*" >&2; exit 1; }

command -v ffmpeg >/dev/null || die "ffmpeg is unavailable"
[[ "$(ffmpeg -hide_banner -encoders 2>/dev/null)" == *libx264* ]] || die "ffmpeg lacks libx264"
[ -r "$FONT" ] || die "font is unreadable: $FONT"
[ -x "$PYTHON" ] || die "python is not executable: $PYTHON"

SEGMENTS=("$@")
[ "${#SEGMENTS[@]}" -eq 0 ] && SEGMENTS=("${DEFAULT_SEGMENTS[@]}")

mkdir -p "$OUTPUT_DIR"
free_gib=$(df -PBG "$OUTPUT_DIR" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
[ "$free_gib" -ge "$MIN_FREE_GIB" ] || die "only ${free_gib}G free at $OUTPUT_DIR (need ${MIN_FREE_GIB}G)"

echo "output   : $OUTPUT_DIR"
echo "encoding : ${FPS} fps, libx264 crf ${CRF}, preset ${PRESET}, native resolution"
echo

for entry in "${SEGMENTS[@]}"; do
    [[ "$entry" == *=* ]] || die "expected label=path[#prefix], got: $entry"
    label="${entry%%=*}"
    spec="${entry#*=}"
    segment="${spec%%#*}"
    prefix="left"
    [[ "$spec" == *#* ]] && prefix="${spec##*#}"
    [ -d "$segment/images" ] || die "$label: missing $segment/images"
    [ -f "$segment/frames.npz" ] || die "$label: missing $segment/frames.npz"

    # The encoder reads <prefix>_%06d starting at 0, so frame zero must exist.
    first="$(find -L "$segment/images" -maxdepth 1 -name "${prefix}_000000.*")"
    [ -n "$first" ] || die "$label: no ${prefix}_000000.* in $segment/images"
    [ "$(printf '%s\n' "$first" | wc -l)" -eq 1 ] \
        || die "$label: several ${prefix}_000000.* extensions in $segment/images"
    ext="${first##*.}"
    n_images=$(find -L "$segment/images" -maxdepth 1 -name "${prefix}_*.$ext" | wc -l)

    # Resolution comes from the frame itself, so no calibration schema is assumed.
    read -r width height < <(
        ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
            -of csv=p=0 "$first" | tr ',' ' '
    )
    # Frame count, span and the drift of a linear clock, straight from the segment.
    read -r n_frames span drift < <(
        "$PYTHON" - "$segment" <<'PY'
import sys
import numpy as np
frames = np.load(f"{sys.argv[1]}/frames.npz")
t = frames["timestamp_ns"].astype(np.float64) / 1e9
t -= t[0]
drift = np.abs(t - np.linspace(0.0, t[-1], len(t))).max()
print(len(t), f"{t[-1]:.3f}", f"{drift:.3f}")
PY
    )
    [ "$n_images" -eq "$n_frames" ] \
        || die "$label: $n_images images but $n_frames rows in frames.npz"

    out="$OUTPUT_DIR/${label}.mp4"
    [ -e "$out" ] && die "$out already exists; remove it or set OUTPUT_DIR"

    # Elapsed seconds as a linear function of the frame index; n is exact.
    per_frame=$(awk -v s="$span" -v n="$n_frames" 'BEGIN {printf "%.9f", s/(n-1)}')
    text="$label  frame %{eif\\:n\\:d}/$((n_frames - 1))  t=%{eif\\:n*${per_frame}\\:d}s"

    echo "--- $label"
    echo "    $n_frames frames  ${width}x${height}  ${span}s source  clock drift <= ${drift}s"
    ffmpeg -hide_banner -loglevel warning -stats -y \
        -framerate "$FPS" -start_number 0 -i "$segment/images/${prefix}_%06d.$ext" \
        -vf "drawtext=fontfile=$FONT:text='$text':x=16:y=16:fontsize=24:\
fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8" \
        -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p \
        -movflags +faststart "$out"

    "$PYTHON" - "$out" "$label" "$segment" "$n_frames" "$span" "$drift" \
                "$width" "$height" "$FPS" "$CRF" "$prefix" <<'PY'
import json, os, sys
out, label, segment, n, span, drift, w, h, fps, crf, prefix = sys.argv[1:12]
n, span, fps = int(n), float(span), float(fps)
json.dump({
    "label": label,
    "segment": segment,
    "image_prefix": prefix,
    "frames": n,
    "resolution": f"{w}x{h}",
    "source_span_s": span,
    "source_rate_hz": round(n / span, 3),
    "playback_fps": fps,
    "speed_up": round(fps / (n / span), 2),
    "clock_overlay_max_drift_s": float(drift),
    "encoder": f"libx264 crf {crf}",
    "size_bytes": os.path.getsize(out),
}, open(os.path.splitext(out)[0] + ".json", "w"), indent=2, sort_keys=True)
PY
    echo "    -> $out ($(du -h "$out" | cut -f1))"
    echo
done

echo "done."
du -ch "$OUTPUT_DIR"/*.mp4 | tail -1
