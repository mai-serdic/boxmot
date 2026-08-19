# AGENTS.md — working guidelines for SOLUTION-ReID

Person re-identification for fixed CCTV. The premise is that **appearance is not
enough** — workers change clothes, put vests on and off, and are seen from behind
most of the time — so the system reasons about the floor in metres and about
whole motion paths, and treats appearance as one signal among several.

This file replaces the upstream BoxMOT `AGENTS.md`, which described a different
project and told you to use `uv`. It does not apply here.

---

## 1. Environment

Use the miniforge env **`mai`**. Not `uv`, not `.venv`, not the system Python.

```bash
/home/serdic-mai/miniforge3/envs/mai/bin/python
```

`boxmot` is **vendored in this repo** at `boxmot/`, not pip-installed. It
resolves because the repo root is on `sys.path` — every entry point inserts
`PROJECT_ROOT` before importing it. Run scripts from the repo root.

Watch for `onnxruntime` shadowing `onnxruntime-gpu`; if inference silently drops
to CPU, that is usually why.

## 2. Layout

```
reid/       the library — pure algorithm, no I/O beyond load/save
scripts/    command-line entry points
bench/      13 numbered benchmarks + REPORT.md
docs/       COMMISSIONING.md
labels/     hand ground truth (JSON); generated .html is ignored
boxmot/     vendored upstream tracker
models/     weights          ─┐
calib/      commissioned scenes │ all gitignored: per-camera or regenerable
runs/       tracking output     │
gallery/    person galleries   ─┘
```

Import the library as a package: `from reid.scene_geometry import GroundPlane`.
`reid/__init__.py` re-exports nothing on purpose, so importing `reid` does not
pull in cv2, torch or onnxruntime.

## 3. Git

`git-lfs` is **not installed**, but `.gitattributes` routes `*.pt`, `*.onnx`,
`*.engine`, `*.pth` and `*.data` through it. Staging those files without the
filter can overwrite real weights with pointer text. Install `git-lfs` before
committing anything that touches `models/`.

`models/`, `gallery/persons.npz` and `.omc/` are gitignored but were committed
earlier, so the ignore rules do not apply to them. Untracking needs
`git rm --cached`, which is an LFS-sensitive operation — do it deliberately.

## 4. Rules that came from the user, not from the code

- **No per-site annotation.** Door gating was proposed and rejected: "sometimes
  the door location is not even in camera, and we want a general solution, not a
  one-site one." Commissioning must work from footage alone. Drawn occluder
  zones are acceptable; marked doorways are not.
- **Face is not a fallback.** "Majority of the time it is people's back."
  `face_anchor` is confirm-only — a face mismatch never blocks a rebind.
- **The goal is a 3D map and real tracking**, "not just IoU."

## 5. Before trusting a calibration

Look at `calib/<camera>/08_metre_check.png` — a 1 m floor grid and 1.70 m
silhouettes drawn back into the image. If the squares are not square and the
sticks do not match real people, no summary statistic should talk you out of it.
This has already caught a real bug: RANSAC in `fit_height_field` maximised inlier
*count*, so three seated people supplying 41% of observations fitted the plane to
a chair. The fix was `dwell_weights` — weight by distinct viewpoint, not dwell
time. Hard spatial dedup was tried first and made things worse; see REPORT §15
before trying it again.

Commission on a **whole session**, not a 30 s clip. Calibration is a property of
the camera and a short clip yields too few tracklets to fit it.

## 6. Benchmarks are the record

`bench/REPORT.md` documents what was measured, including what failed. Negative
results are kept deliberately — stature as an identity cue was tested and
**rejected** (AUC 0.574). Read the relevant section before re-proposing an idea.

REPORT.md is a dated log and refers to modules by bare filename
(`scene_geometry.py`); it predates the move into `reid/`.

Known open issues: `birth_cost` in `trajectory_stitch` does not transfer between
cameras; monocular depth scale is badly off on some rooms, which leaves the
ground plane correct but the occluder map wrong; the global stitcher is validated
in `bench/` but is not yet wired into the tracking pipeline.
