# Delivery — what this repo is for

This is an **AI-engineer delivery**: scripts that track people the way a human
would in a room — by **where they walk**, not by what they look like. Appearance
is short-term glue; **within-session identity is a path through floor space**.

Gallery ids (`Person_XXX`, `pid_frames`) are for **cross-session enrolment only**.
They are not the answer to "who is this in this clip after they put a vest on."

Production wiring (DeepStream, BEV overlay, Control-API, agents, attendance) is a
**proposal** for the web team. It is not work in this repo.

---

## Identity model

Humans follow an object by understanding the space: where it can go, where it
cannot, how fast it could have moved. That is what this pipeline implements.

```
detector + BoT-SORT     short-term boxes, many trk_ids per person
commission + reach map  floor in metres, geodesic "can they get there?"
ghost pool (live)       greedy rebind — fast, often wrong after occlusion
whole-path stitch       min-cost flow over tracklets → path id 1..N
```

**Output that matters:** `path_frames` — per frame, each box labelled by **path
id** (one person walking through the room), not gallery id.

| field | meaning |
|---|---|
| `traj` | BoT-SORT tracklets — input to stitch |
| `path_frames` | stitched path ids — **within-session identity** |
| `owner_path` | tracklet id → path id |
| `groups_path` | path id → list of tracklet ids |
| `pid_frames` | gallery labels — cross-session only; breaks on clothing change |
| `trk_embs` | mean CLIP embedding per tracklet — optional tie-break only (`--w-emb`; default off) |

Appearance is deliberately weak here. Workers change clothes and are seen from
behind most of the time — a human tracks by following the object through the
room, not by matching pixels. CLIP/gallery may help BoT-SORT hold a box for a
few seconds, but **identity in this clip is `path_frames`**, not `pid_frames`.

---

## What you ship

```
footage
  → scripts/commission_scene.py     calib/<camera>/  (check 08_metre_check.png)
  → scripts/annotate_floor.py       floor_plan.json  (when depth misses furniture)
  → scripts/track_rtdetr_db.py      --dump-json --stitch --scene
  → scripts/stitch_traj.py          same stitch offline on an existing dump
```

Measured numbers live in `bench/REPORT.md`. Run artefacts live in `runs/`
(gitignored).

| you deliver | you do not deliver |
|---|---|
| traj + path_frames JSON | DeepStream / nvtracker plugins |
| commissioned calib + floor plan | BEV homography / virtual_aruco |
| scripts + honest limits | Control-API, agents, attendance |
| IDF1 / split metrics from bench | SaaS UI, live dashboard |

---

## Run recipe (office_cam1)

Use a **3–5 minute clip** for iteration — the full 15.5 min block is for final
validation only. Example:

```bash
# optional: cut a short test clip from the long recording
ffmpeg -y -i runs/track/office_cam1_block2.mp4 -t 300 -c copy \
    runs/track/office_cam1_block2_5min.mp4
```

The dump **must** contain `traj`. Stitching `pid_frames` alone is wrong — those
are gallery fragments, not spatial tracklets.

```bash
conda activate mai
cd ~/SOLUTION-ReID

# Track + stitch (~5 min clip, not the full 15 min session)
python scripts/track_rtdetr_db.py \
    --input runs/track/office_cam1_block2_5min.mp4 \
    --scene calib/office_cam1 \
    --gallery gallery/office_cam1/persons.npz \
    --dump-json runs/traj/office_cam1_block2_5min.json \
    --stitch \
    --output runs/track/office_cam1_block2_5min_path.mp4

# Or stitch an existing dump that has "traj"
python scripts/stitch_traj.py \
    --scene calib/office_cam1 \
    --traj  runs/traj/office_cam1_block2_tracks.json \
    --fps 10 --out runs/traj/office_cam1_block2_stitched.json
```

Inspect `path_frames` / `groups_path` in the JSON. That is the identity answer
for this recording.

---

## What is proven — and what is not

**Space-first stitching beats greedy on labelled gunsan footage:** +18 IDF1
(50.7% → 68.6%), geodesic rebind ranking 61.7% → 84.9% (`bench/REPORT.md`).

**Two people, vest change:** geometry is the primary signal. When fragments never
overlap in time, stitch can still get the count wrong or swap paths — that is
why the bench keeps negative results on clothing (REPORT §10). Optional
`--w-emb` can nudge ambiguous links; it defaults to **0** because appearance
already failed before stitch runs.

**Do not claim** gallery ReID or path stitch alone solves PPE / vest change.
Say: path through space is the primary signal; appearance is auxiliary; cross-day
names need enrolment and are a separate product.

---

## Handoff proposal (web team)

```
this repo                         their stack
─────────                         ───────────
path_frames + owner_path    ──►   personId on an alert (nullable)
floor metres + groups_path  ──►   draw path on BEV
stitched count in one clip  ──►   incident doc for that recording
                                  (not attendance, not cross-day names)
```

Contract: one JSON with `{frames, fps, traj, path_frames, owner_path,
groups_path}` in pixel boxes + floor metres via `scene.json`. They map it.

---

## Out of scope

- DeepStream, BEV, Control-API, PPE pipeline, agents, attendance
- Demo comparison videos (optional; not required for delivery)
- Aligning ReID ground plane with BEV virtual_aruco

Those are downstream. This folder ships the spatial path identity pipeline and
honest measured limits.
