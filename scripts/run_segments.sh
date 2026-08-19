#!/usr/bin/env bash
# Run the persistent-ID pipeline over a directory of consecutive CCTV segments,
# sharing ONE gallery across them. Identity must survive the segment boundary --
# that is the whole point of a persistent DB, and it is the closest thing we have
# to a multi-hour test.
#
#   scripts/run_segments.sh <segment-dir> <gallery.npz> <out-dir> [extra args...]
set -euo pipefail

SEG_DIR="$1"; GALLERY="$2"; OUT_DIR="$3"; shift 3
mkdir -p "$OUT_DIR"
rm -f "$GALLERY"

for f in "$SEG_DIR"/*.mp4; do
    base=$(basename "$f" .mp4)
    echo "=== $base"
    python scripts/track_rtdetr_db.py \
        --input "$f" \
        --output "$OUT_DIR/${base}.mp4" \
        --gallery "$GALLERY" \
        --dump-json "$OUT_DIR/${base}.json" \
        "$@"
done
