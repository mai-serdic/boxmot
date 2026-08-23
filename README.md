# SOLUTION_ReID

Person re-identification for fixed CCTV, built on the premise that **appearance
is not enough**. Workers change clothes, put on and take off hi-vis vests, and
are seen from behind most of the time. What stays constant is the space they
move through, so this system reasons about the floor in metres and about whole
motion paths, and uses appearance only as one signal among several.

Measured: the body embedding separates same-person from different-person pairs
at 0.944 AUC *within* a tracklet and only **0.803** across tracklets, and a
0.55 embedding veto refuses **24.5%** of genuine re-entries. Replacing the
pixel-space spatial prior with a geodesic one on the walkable floor took
re-identification ranking accuracy from **61.7% to 84.9%**. Full numbers and
the negative results in [`bench/REPORT.md`](bench/REPORT.md).

## How it works

```
footage ──▶ commission_scene.py ──▶ calib/<camera>/     (once per camera)
                                     scene.json          ground plane, metres
                                     scene_depth.npz     height above floor
                                     08_metre_check.png  the sheet you check

footage ──▶ track_rtdetr_db.py ──▶ runs/traj/<clip>.json
             RT-DETRv4 + BoT-SORT + CLIP/OSNet embeddings
             + ghost pool (rebinding) + geodesic reachability prior
             ──▶ stitch_traj.py   whole-path IDs (offline post-pass)
```

**Commissioning takes no annotation.** No checkerboard, no surveyor, no drawn
zones, no marked doorways. The ground plane is fitted from people walking
through the frame, which is what makes it transferable to a new camera instead
of a per-site ritual. It has been run on three unrelated rooms.

Accept a calibration only from `08_metre_check.png` — a 1 m floor grid and
1.70 m human silhouettes drawn back into the image. If the squares are not
square and the sticks do not match real people, no summary statistic should
talk you out of it. This has caught a real bug.

## Modules

| file | what it does |
|---|---|
| `reid/scene_geometry.py` | ground plane from pedestrians: distortion, horizon, focal, metre scale |
| `reid/scene_depth.py` | monocular depth → height above floor → occluder map; stature bias field |
| `reid/reachability.py` | geodesic prior — could a person have *walked* from here to there? |
| `reid/floor_plan.py` | hand-drawn walkable/obstacle footprints, stamped onto the reach map |
| `reid/trajectory_stitch.py` | min-cost flow over the tracklet graph; one unit of flow = one person's path |
| `reid/motion_prior.py` | learned (cell, heading) Markov model of how people move through the room |
| `reid/ghost_pool.py` | within-session rebinding of lost tracks, multi-signal with an embedding veto |
| `reid/person_db.py` | persistent gallery that survives across runs and days |
| `reid/face_anchor.py` | pose-gated face confirmation (confirm-only; a mismatch never blocks) |

## Layout

```
reid/       the library — pure algorithm, no I/O beyond load/save
scripts/    command-line entry points (commission, track, label, render)
bench/      13 numbered benchmarks + REPORT.md, negative results included
docs/       COMMISSIONING.md — how to commission a camera and what to check
            DELIVERY.md — what this repo ships (scripts + demo); web-team handoff
labels/     hand ground truth for the benchmarks
models/     detector and ReID weights (gitignored)
calib/      commissioned scenes, one per camera (gitignored)
runs/       tracking output (gitignored)
gallery/    persistent person galleries (gitignored)
videos/     source footage (gitignored)
boxmot/     vendored upstream tracker — BoT-SORT, ByteTrack, ReID zoo
```

`bench/REPORT.md` is a dated record and refers to modules by bare filename
(`scene_geometry.py`); it predates the move into `reid/`.

## Usage

```bash
pip install -r requirements.txt   # boxmot itself is vendored in this repo

# 1. Commission a camera. Input is footage and nothing else.
#    Use the whole session, not one clip — calibration is a property of the
#    camera, and a 30 s clip yields too few tracklets to fit it.
python scripts/commission_scene.py --name mycam --input footage.mp4 --fps 10

# 2. Look at calib/mycam/08_metre_check.png before trusting anything.

# 3. Track.
python scripts/track_rtdetr_db.py \
    --input clip.mp4 --scene calib/mycam \
    --gallery gallery/mycam/persons.npz \
    --dump-json runs/traj/clip.json --stitch \
    --output runs/track/clip.mp4

# 4. Reproduce the geodesic-prior benchmark.
python bench/08_eval_geodesic_prior.py --scene calib/mycam \
    --traj runs/traj/clip.json --labels labels/mycam.json

# 5. Stitch whole paths (offline). Omit --birth to derive it from the links.
python scripts/stitch_traj.py --scene calib/mycam \
    --traj runs/traj/clip.json --out runs/traj/clip_stitched.json
```

## What is not here

Model weights, footage, commissioned scenes and pipeline output are all
gitignored. This repo is source only.

**This is not the factory product.** DeepStream, BEV, Control-API, agents and
attendance are owned by the web / on-site stack. What you ship from here is
the pipeline scripts, a stitch JSON, and a demo video. What to *propose* they
wire is in [`docs/DELIVERY.md`](docs/DELIVERY.md).

## Known limitations

- `birth_cost` is derived from occupancy and the prior floor (`p_floor`
  hops, clipped to [4, 6]). Cheap best-incomings can be impostors; true
  long-gap re-entries often sit at `p_floor`, which is why office_cam1
  needs 6 rather than 4.
- Monocular depth scale can be badly off on some rooms (2.5× on one camera).
  The ground plane stays correct, but the occluder map does not.
- The global stitcher is a post-pass: `scripts/stitch_traj.py` (also
  `--stitch` on `track_rtdetr_db.py`). Live tracking still writes the greedy
  ghost-pool IDs; identity is rewritten when the clip ends. The demo is that
  contrast — greedy mixing two people, stitch recovering the split.
- Stature was tested as an identity cue and **rejected** (AUC 0.574): human
  stature spread is ~0.07 m against 0.12–0.24 m of instrument noise. See
  REPORT §12 before trying it again.
