# Commissioning a camera

Turning a new camera into a metric 3D scene. No checkerboard, no tape measure,
no drawn zones, no door annotation — just footage of people walking around.

---

## TL;DR — paste and run

```bash
conda activate mai
cd ~/SOLUTION-ReID

# from a video
python scripts/commission_scene.py --name MYSITE --input /path/to/footage.mp4

# ...or from a folder of exported frames
python scripts/commission_scene.py --name MYSITE --input /path/to/frames/

# then open the image pack and check it
xdg-open calib/MYSITE/08_metre_check.png
```

That is the whole procedure. Replace `MYSITE` and the input path; everything
else is automatic. Tracklets, distortion, ground plane, depth scene, occluder
map and stature field are all built and written to `calib/MYSITE/`.

Runtime is dominated by detection: roughly real-time on GPU, so a 5-minute
clip takes about 5 minutes.

---

## What footage to give it

The calibration learns the room's geometry *from the people in it*, so the
footage has to contain people walking. Requirements, in order of importance:

| requirement | why | minimum |
|---|---|---|
| people walk around **different parts** of the floor | the ground plane is fitted across image positions; people in one spot fit nothing | coverage of the walkable area |
| people are **fully visible** (head to feet) sometimes | the head↔feet pair is the measurement | ≥200 such observations |
| several **different people**, or one person over time | separates "person" from "position" in the bias field | ≥8 usable tracklets |
| camera is **fixed** | everything assumes a static scene | must not pan/zoom |

A 5–15 minute clip from a normal working period is usually plenty. A busy
period is better than a long empty one. If you get
`very few usable tracklets`, use a longer or busier clip — that message is the
tool telling you the footage cannot support a calibration, not a bug.

**You do not need to do anything special.** No one walks a pattern, no one
holds a target. Normal work footage is the input.

---

## Reading the result

The run prints checks as it goes and writes them to
`calib/MYSITE/commission.md`:

```
[PASS] observations                 764 standing full-body obs from 24 tracklets
[PASS] height-field fit             rel-rms 5.3% (want <=6%)
[PASS] focal length                 f=1354px  fov=71deg  near/far bias 1.021
[PASS] camera height                2.88 m
[PASS] stature spread               p10/p50/p90 = 1.59/1.70/1.82 m
[WARN] depth vs floor               1-sigma 18 cm on known floor points
[PASS] floor coverage               18% of pixels are floor
[PASS] free floor area              2.6 m^2 standable of 20.2 m^2 observed
[WARN] stature position artefact    20 -> 20 (gated) -> 9 cm (bias field)
[PASS] bias field reproducible      split-half correlation r=+0.934
```

### The one check that actually decides it

**`08_metre_check.png`.** It draws a 1 m floor grid and 1.70 m human
silhouettes back into the image.

- Do the grid squares look **square on the floor**, and roughly the size of a
  large floor tile pattern you can recognise?
- Do the yellow sticks match the **height of real people** in the scene?

If yes, the metre scale is real and everything downstream — speed limits,
reachability, stature — inherits it. If no, **reject the calibration**; no
summary statistic should talk you out of what you can see.

This sheet exists because a calibration that cannot be *seen* to be right will
eventually be trusted when it is wrong.

### The rest of the pack

| sheet | shows | what "good" looks like |
|---|---|---|
| `01_distortion.png` | raw vs undistorted | room edges straight after |
| `02_distortion_fit.png` | the k1 search curve | a clear peak, not a flat line |
| `03_horizon.png` | horizon + kept/rejected observations | green points spread over the floor |
| `04_depth.png` | rescaled depth + height above floor | floor near 0 m, furniture 0.5–1.5 m |
| `05_occluder_map.png` | **derived occluder map** | green = walkable floor, red = furniture |
| `06_floorplan.png` | top-down obstacles, free space, paths | room shape recognisable, paths in free space |
| `07_stature.png` | stature raw → gated → corrected | the third box much tighter than the first |
| `08_metre_check.png` | **the acceptance test** | see above |
| `09_stature_field.png` | the bias field + split-half check | points on the diagonal, r > 0.4 |

---

## Outputs, and what consumes them

```
calib/MYSITE/
  scene.json          ground plane: focal length, horizon, camera height, floor basis
  scene_depth.npz     per-pixel height above floor (the occluder map)
  stature_field.json  per-cell stature correction
  commission.md       the checks, in writing
  01..09_*.png        the image pack
  depth_raw.npy       cached depth (skip the model on a re-run with --depth-npy)
```

Using them:

```python
from reid.scene_geometry import GroundPlane
from reid.scene_depth import SceneModel, StatureField, localize

gp    = GroundPlane.load("calib/MYSITE/scene.json")
scene = SceneModel.load("calib/MYSITE/scene_depth.npz", gp)

scene.foot_occluded(foot_px)        # is this box bottom furniture, not feet?
gp.floor_xy(foot_norm_h)            # metres on the floor
gp.stature_m(foot_h, head_h)        # metric height
localize(gp, scene, foot_px, foot_h, head_h)   # occlusion-aware (xy, stature, feet_visible)
```

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `very few usable tracklets` | not enough people, or clip too short | longer / busier clip |
| `observations` WARN | people rarely fully visible | clip with more open-floor walking |
| `height-field fit` rel-rms > 6% | floor not planar, or bad tracklets | check `03_horizon.png` for scattered rejects |
| `focal length` near/far bias > 5% | stature still encodes position | usually follows a bad height-field fit — fix that first |
| `camera height` implausible | wrong `--mean-stature` for the population | pass e.g. `--mean-stature 1.65` |
| grid on sheet 08 is visibly skewed | calibration failed | do not use it; get better footage |
| `depth vs floor` WARN | depth model precision | tolerable; it only affects the occluder map's sharpness |
| CUDA / onnxruntime errors | wrong env | `conda activate mai` (see the env note below) |

Useful flags:

```bash
--mean-stature 1.65     # population mean height; sets the metre scale
--traj path.json        # reuse existing tracklets instead of rebuilding
--depth-npy path.npy    # reuse a cached depth map (fast re-runs)
--k1 -0.185             # skip distortion estimation, supply it
--fps 25                # frame rate of the source, for the tracker
--out-dir somewhere/    # write elsewhere than calib/<name>/
```

Environment: use the miniforge `mai` env. `onnxruntime-gpu` is shadowed by
`onnxruntime` if both are installed, which silently drops you to CPU — the
`[det ]` line prints the active provider, so check it says
`CUDAExecutionProvider`.

---

## Running the steps separately

`commission_scene.py` does all of this; run the pieces only if you want to
inspect or reuse an intermediate.

```bash
# 1. tracklets (motion-only ByteTrack, no appearance model)
python scripts/make_tracklets.py --input footage.mp4 --out runs/traj/MYSITE.json

# 2. commissioning, reusing them
python scripts/commission_scene.py --name MYSITE --input footage.mp4 \
    --traj runs/traj/MYSITE.json

# 3. benchmark the result (reachability + stature as a ReID cue)
python bench/06_eval_metric_geometry.py --traj runs/traj/MYSITE.json \
    --calib calib/MYSITE/scene.json --scene calib/MYSITE/scene_depth.npz
```

---

## What this does and does not buy you

**Does:** a metric floor, so distances are metres and a speed limit is physics
rather than a tuned pixel budget; and an automatically derived occluder map.

**Does not:** identify people across days, or use stature as an ID cue
(tested, rejected — see REPORT §12). All of the geometry is session-local.
Only face (or a badge) answers "is this the same worker as Monday", and face
is confirm-only because people are seen from behind.

The identity answer for one recording is the offline stitcher
(`scripts/stitch_traj.py`), not the live tracker. See `docs/DELIVERY.md`.

See `bench/REPORT.md` steps 7–8 for the measurements behind these claims,
including the honest confidence intervals.
