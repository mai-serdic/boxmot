"""
Tracklet-level person labelling aid.

The reachability bench (bench/08) can only prove "same person" *within* one
tracklet, which is precisely the case the long-gap rebind feature does not
exercise. The missing ground truth is which tracklets are the same person -
about 30 judgements for a clip, not per-frame boxes.

This renders one contact sheet per tracklet into a single self-contained HTML
page, so the call is made by looking at crops side by side rather than by
scrubbing video. Fill in a person name per tracklet (any string; reuse it for
tracklets that are the same person, leave blank to skip) and hit Download.

    python scripts/label_tracklets.py --video videos/gunsan_test.mp4 \\
        --traj runs/traj/gunsan_test.json --out labels/gunsan_test.html

The downloaded JSON is `{"<track_id>": "<person>"}` and is what
bench/08 --labels consumes.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np


def crops_for(cap, rows: list, n: int, crop_h: int, pad: float) -> list:
    """n evenly spaced crops across the tracklet's lifetime."""
    picks = np.unique(np.linspace(0, len(rows) - 1, n).astype(int))
    out = []
    for i in picks:
        f, x1, y1, x2, y2 = rows[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        w, h = x2 - x1, y2 - y1
        x1 = int(max(x1 - pad * w, 0)); x2 = int(min(x2 + pad * w, W))
        y1 = int(max(y1 - pad * h, 0)); y2 = int(min(y2 + pad * h, H))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        c = frame[y1:y2, x1:x2]
        s = crop_h / c.shape[0]
        c = cv2.resize(c, (max(int(c.shape[1] * s), 8), crop_h))
        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            out.append((int(f), base64.b64encode(buf).decode()))
    return out


PAGE_HEAD = """<meta charset="utf-8"><title>Tracklet labelling &mdash; {name}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#14161a;color:#e8eaed}}
 header{{position:sticky;top:0;background:#1b1e24;border-bottom:1px solid #2c313a;
   padding:12px 18px;display:flex;gap:16px;align-items:center;z-index:9}}
 h1{{font-size:16px;margin:0;font-weight:600}}
 .hint{{color:#9aa3af;font-size:13px}}
 button{{background:#3b82f6;color:#fff;border:0;border-radius:6px;padding:8px 14px;
   font:inherit;font-weight:600;cursor:pointer}}
 button.sec{{background:#2c313a}}
 .trk{{border-bottom:1px solid #23272f;padding:12px 18px;display:flex;gap:16px;
   align-items:flex-start}}
 .meta{{min-width:190px;flex-shrink:0}}
 .tid{{font-weight:700;font-size:15px}}
 .sub{{color:#9aa3af;font-size:12px;font-variant-numeric:tabular-nums}}
 input{{margin-top:8px;width:170px;background:#0f1114;border:1px solid #39404b;
   color:#e8eaed;border-radius:6px;padding:7px 9px;font:inherit}}
 input:focus{{outline:2px solid #3b82f6;border-color:transparent}}
 .strip{{display:flex;gap:6px;flex-wrap:wrap;overflow-x:auto}}
 figure{{margin:0;text-align:center}}
 img{{display:block;border-radius:4px;background:#000}}
 figcaption{{color:#6b7280;font-size:11px;margin-top:2px}}
 .done{{background:#16351f}}
</style>
<header>
 <h1>{name}</h1>
 <span class="hint">Type a person name (e.g. <b>A</b>) &mdash; reuse it for
  tracklets that are the same person. Blank = skip. <b>{n}</b> tracklets.</span>
 <span style="flex:1"></span>
 <span class="hint" id="count"></span>
 <button class="sec" onclick="copyJson()">Copy JSON</button>
 <button onclick="dl()">Download labels.json</button>
</header>
"""

PAGE_TAIL = """<script>
const inputs = () => [...document.querySelectorAll('input[data-tid]')];
function collect(){
  const o = {};
  for (const i of inputs()){
    const v = i.value.trim();
    i.closest('.trk').classList.toggle('done', !!v);
    if (v) o[i.dataset.tid] = v;
  }
  document.getElementById('count').textContent =
    Object.keys(o).length + ' / ' + inputs().length + ' labelled, ' +
    new Set(Object.values(o)).size + ' distinct people';
  return o;
}
document.addEventListener('input', collect);
function dl(){
  const b = new Blob([JSON.stringify(collect(), null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'labels.json'; a.click();
}
function copyJson(){ navigator.clipboard.writeText(JSON.stringify(collect(), null, 2)); }
collect();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-crops", type=int, default=10)
    ap.add_argument("--crop-h", type=int, default=150)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    tracks = {int(k): v for k, v in traj["traj"].items() if len(v) >= args.min_obs}
    order = sorted(tracks, key=lambda t: tracks[t][0][0])
    print(f"[INFO] {len(tracks)} tracklets with >= {args.min_obs} observations")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    name = Path(args.traj).stem
    parts = [PAGE_HEAD.format(name=html.escape(name), n=len(tracks))]
    for tid in order:
        rows = tracks[tid]
        cs = crops_for(cap, rows, args.n_crops, args.crop_h, args.pad)
        f0, f1 = int(rows[0][0]), int(rows[-1][0])
        strip = "".join(
            f'<figure><img src="data:image/jpeg;base64,{b64}" alt="f{f}">'
            f'<figcaption>{f}</figcaption></figure>' for f, b64 in cs)
        parts.append(
            f'<div class="trk"><div class="meta">'
            f'<div class="tid">track {tid}</div>'
            f'<div class="sub">frames {f0}&ndash;{f1} &middot; '
            f'{(f1 - f0) / args.fps:.0f}s &middot; {len(rows)} obs</div>'
            f'<input data-tid="{tid}" placeholder="person…" autocomplete="off">'
            f'</div><div class="strip">{strip}</div></div>')
        print(f"  track {tid:3d}: {len(cs)} crops, frames {f0}-{f1}")
    cap.release()

    parts.append(PAGE_TAIL)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"\n[DONE] {out}  ({out.stat().st_size / 1e6:.1f} MB) — open it in a browser")


if __name__ == "__main__":
    main()
