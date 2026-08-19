"""
Step 7 — ID consistency report.

Answers the only question that matters when you already know how many people
are in the clip: how many identities did the system invent, and where?

Input is either
  * a tracklet file from make_tracklets.py       ({"traj": {tid: [[f,x1,y1,x2,y2]]}})
  * a person-ID dump from track_rtdetr_db.py     ({"pid_frames": {f: [[pid,x1,y1,x2,y2]]}})

Usage:
    python bench/07_report_id_consistency.py runs/render/gunsan_test_reid_ids.json \
        --truth 4 --fps 15
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[int, list[int]]:
    """-> {id: sorted list of frames it appears on}"""
    d = json.loads(path.read_text())
    out: dict[int, list[int]] = defaultdict(list)
    if "pid_frames" in d:
        for fr, rows in d["pid_frames"].items():
            for pid, *_ in rows:
                out[int(pid)].append(int(fr))
    else:
        for tid, rows in d["traj"].items():
            for r in rows:
                out[int(tid)].append(int(r[0]))
    return {k: sorted(v) for k, v in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--truth", type=int, default=0,
                    help="true number of distinct people in the clip")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--min-frames", type=int, default=5,
                    help="ignore IDs seen fewer times than this")
    args = ap.parse_args()

    ids = load(args.path)
    kept = {k: v for k, v in ids.items() if len(v) >= args.min_frames}
    dropped = len(ids) - len(kept)

    print(f"[in ] {args.path}")
    print(f"[ids] {len(ids)} distinct  ({len(kept)} with >={args.min_frames} "
          f"frames, {dropped} blips)")
    if args.truth:
        print(f"[gt ] {args.truth} real people  ->  "
              f"fragmentation x{len(kept)/args.truth:.1f}")

    # Concurrency ceiling: no honest system needs more IDs than the most people
    # ever on screen at once... but it does need at least that many.
    per_frame: dict[int, int] = defaultdict(int)
    for frames in kept.values():
        for f in frames:
            per_frame[f] += 1
    print(f"[max] {max(per_frame.values(), default=0)} people on screen at once")

    print(f"\n{'id':>6} {'frames':>7} {'first':>9} {'last':>9}  span")
    for k, v in sorted(kept.items(), key=lambda kv: kv[1][0]):
        t0, t1 = v[0] / args.fps, v[-1] / args.fps
        print(f"{k:>6} {len(v):>7} {t0:>8.1f}s {t1:>8.1f}s  "
              f"{t1-t0:>6.1f}s")


if __name__ == "__main__":
    main()
