# ReID embedding benchmark — OSNet vs CLIP-ReID on gunsan_test

**Question (option A):** before changing the pipeline, is a better embedding
model actually worth it — and how far does it get us on this project's real
failure modes (side / pose / viewpoint change, and telling co-present workers
apart)?

## Method — ground-truth-free, from your own footage

No manual labels. We ran **RT-DETRv4 + motion-only ByteTrack** (no appearance
model, so labels can't favor any embedding) over `videos/gunsan_test.mp4`
(4484 frames), then derived weak labels from the tracklets:

- **Positive pair** = two crops of the *same* tracklet, sampled ≥30 frames
  apart (median 4.0 s, up to 39.5 s) → captures real pose / side / viewpoint
  change.
- **Negative pair** = two crops from tracklets active in the *same frame* →
  physically guaranteed different people (this is the "two workers on screen"
  merge risk).

645 positive + 645 negative pairs, identical for every model.
Scripts: [`bench/01_extract_tracklets.py`](01_extract_tracklets.py),
[`02_build_pairs.py`](02_build_pairs.py), [`03_eval_embeddings.py`](03_eval_embeddings.py).
Distributions: `dist_hist.png`. Raw numbers: `metrics.json`.

## Results

| model | dim | AUC ↑ | EER ↓ | pos med | neg med | separation ↑ | fragment@τ=0.25 | merge@τ=0.25 |
|---|---|---|---|---|---|---|---|---|
| osnet_x1_0 (msmt17)  | 512  | 0.772 | 0.307 | 0.351 | 0.466 | +0.115 | **0.749** | 0.016 |
| osnet_x0_25 (msmt17) | 512  | 0.777 | 0.304 | 0.367 | 0.489 | +0.122 | **0.778** | 0.009 |
| **clip_market1501**  | 1280 | **0.860** | **0.219** | 0.363 | 0.622 | **+0.259** | **0.750** | 0.026 |

*AUC = P(same-pair closer than diff-pair), threshold-free. EER = balanced error
at each model's own best threshold. fragment = same-person pairs judged
different (→ new ID). merge = different-person pairs judged same (→ ID collapse).*

## Findings

1. **CLIP-ReID clearly beats OSNet.** AUC 0.77 → 0.86, EER 0.31 → 0.22. It earns
   this on the **merge axis**: it pushes different-people distance from 0.47 to
   0.62 while keeping same-person at ~0.36 — i.e. it separates co-present workers
   much better. Swapping OSNet → CLIP is worth it. (Your `gallery/persons.npz`
   is already 1280-dim, so the *gallery* was built with CLIP — but
   `track_rtdetr_db.py` still defaults `--reid osnet_x1_0`. Make the whole
   pipeline consistently CLIP.)

2. **The operating threshold τ=0.25 is far too strict — this is the biggest,
   cheapest fix.** At τ=0.25, **~75 % of same-person pairs are declared new
   identities** (fragment rate), for *every* model — because even CLIP's
   same-person median distance is 0.36, well past 0.25. The `tau=0.25` line in
   `dist_hist.png` sits at the left edge of the same-person mass. Raising τ
   toward CLIP's EER point (~0.44) would cut fragmentation from ~75 % to ~22 %.

3. **But appearance alone cannot win the trade-off.** Even the best model floors
   at **EER ≈ 0.22** — pushing τ up to kill fragmentation pushes merge rate up
   symmetrically (frag 22 % ⇄ merge 22 % at the balance point). There is *no* τ
   that makes both small. This is the quantitative proof that a better backbone
   helps but does not solve identity stability under pose/appearance change —
   you need a complementary, clothing-independent cue.

## Scope / caveats

- Measures **side / pose / viewpoint** robustness and **co-present separation**
  (2 of the 3 stated problems). It does **not** measure cross-day **clothing
  change** — a single clip has no same-person-different-outfit labels.
- Weak labels: within-tracklet positives are high-confidence but ByteTrack can
  occasionally ID-switch; co-active negatives are exact. 645+645 pairs from 23
  multi-crop tracks — directional, not a precise leaderboard.

## Recommendation

1. **Now (config-only):** standardize the pipeline on **clip_market1501** and
   **re-tune `--match-threshold`** up from 0.25 (sweep 0.35–0.45 with the ratio
   test kept on). Biggest immediate win, zero new code.
2. **Next (the real fix):** add a clothing-independent anchor. Given the project
   already has **human detection + pose estimation** and **face is stable**,
   the natural design is **face + pose-gated re-anchoring** with appearance as
   short-term glue — see the architecture note.
3. Consider a stronger ViT (SOLIDER / TransReID) only after 1–2; the benchmark
   says the appearance ceiling is ~0.22 EER, so extra backbone gains are
   secondary to adding a second cue.

## Threshold sweep (step 4) — run 2026-08-14, regenerated data

Data was regenerated on this machine (RTX 5090): 410 crops, 49 tracks,
715 + 715 pairs. Step 3 re-run confirms the ranking (CLIP AUC 0.842 /
EER 0.227 vs OSNet ≈0.79 / ≈0.29). Sweep on CLIP distances
(`threshold_sweep.png`):

| tau | fragment | merge | note |
|---|---|---|---|
| 0.25 | ~74 % | 2.2 % | old default |
| **0.34** | **52.9 %** | **9.1 %** | merge ≤ 10 % budget point |
| 0.44 | 24.9 % | 21.5 % | EER-balanced |

For the *persistent* gallery a wrong merge cascades while a fragment is
recoverable, so we operate at the merge-budget point rather than EER.
**Chosen operating point: τ = 0.35** (current `--match-threshold` default in
`track_rtdetr_db.py`) — ≈50 % fragmentation (down from ~74 % at 0.25) while
holding merges near the 10 % budget. Pushing further toward the EER point
buys fragmentation down only by paying symmetrically in merges — confirming
finding 3: the remaining gap needs a second, clothing-independent cue.

## Face anchoring (step 5) — run 2026-08-14

Recommendation 2 is now implemented: `face_anchor.py` (SCRFD + ArcFace via
insightface `buffalo_l`, pose-gated on detection confidence ≥0.5, face min
side ≥24 px, and landmark-geometry frontality), stored per person in
`PersonDB` (`face_*` arrays, K-best of 5), consumed by `IdentityResolver`
(face-first decision, confirm-only, plus re-anchoring of already-resolved
tracks), wired into `track_rtdetr_db.py` behind `--face`.
Measured by [`05_eval_face_anchor.py`](05_eval_face_anchor.py) on the same
715+715 pairs (`face_metrics.json`):

- **Coverage:** 252/410 crops have a detectable face; **96 (23 %) pass the
  pose gate**; both-sides-gated pair coverage is only **6.6 % pos / 11 % neg**.
- **Precision on covered pairs:** face AUC 0.907 / EER 0.127 (τ≈0.70) —
  good, but body-CLIP scores *better* (AUC 0.959) on that same subset,
  because both-frontal pairs are exactly the pairs body already nails
  (body pos-median distance there: 0.148 vs 0.36 overall).
- **Fusion within-clip:** face-override *hurts* (fragment 49.9→52.9 %);
  strict **confirm-only** fusion (the pipeline rule: a face match can only
  add a same-person verdict, never reject one) is neutral: fragment
  unchanged, merge +0.6 pp at face-τ=0.50.

**Interpretation — don't over-read the neutral number.** This single-clip
benchmark can only test the regime where body appearance already works
(same clothes, minutes apart, frontal subset). The face anchor exists for
the regimes the benchmark cannot see: cross-day clothing change and
correcting long-gap body-appearance drift. The measured result does establish
two design rules now baked into the implementation: face must be
**confirm-only** (override measurably hurts), and face-τ must sit well below
the face EER point (~5 % of covered negative pairs collide at τ=0.5 on 52 px
faces; the live pipeline additionally applies a Lowe ratio test across the
gallery and requires anchor-grade observations to re-anchor). Validating the
actual payoff needs multi-day footage with the same workers in different
clothing.

### Full-video A/B — face anchoring pays off in the live pipeline

Two complete runs of `gunsan_test.mp4` (4484 frames), identical settings and
fresh galleries, differing only in `--face`:

| run | video | gallery | persons | face interventions |
|---|---|---|---|---|
| baseline | `runs/track/rtdetr_db_clip_noface.mp4` | `gallery/persons_noface.npz` | 5 | — |
| face | `runs/track/rtdetr_db_clip_face.mp4` | `gallery/persons_face.npz` | 5 | 1 anchor, 2 re-anchors |

The interventions: one face-anchored decision (frame 1175, same identity body
would have chosen — confirmatory), and **two independent re-anchors, 1400
frames apart, both moving a track from Person_004 to Person_002**
(frames 1581 and 2995).

The saved gallery shows those re-anchors were right. Min cosine distance
between persons:

|  | P001–P002 | P002–P004 | all other pairs |
|---|---|---|---|
| **body** (τ=0.35) | **0.228** | **0.314** | 0.43 – 0.64 |
| **face** (τ=0.50) | 0.748 | **0.445** | 0.75 – 0.87 |

Read the face row first: it is cleanly bimodal — every pair sits at 0.75+
except P002–P004 at 0.445. Face says these five IDs are really **four
people**, with P004 a fragment of P002. That matches how P004 was born:
enrolled as new at frame 1376 with nearest neighbour at d=0.417, a textbook
fragment at our ~50 %-fragmentation operating point.

The body row is the alarming one. Body appearance puts **P001 and P002 at
0.228** — far *below* the τ=0.35 match threshold, i.e. these two genuinely
different workers (face distance 0.748) are one unlucky query away from
merging, which for a persistent gallery is the expensive, cascading error.
Body also can't fix the P002/P004 fragment for the same reason: it ranks the
wrong pair closest. This is finding 3 from the original benchmark showing up
as concrete IDs in production, and it is exactly the failure the second cue
was added to catch.

**Known limitation — re-anchoring relabels, it does not consolidate.** A
re-anchor corrects that track's `person_id`, but P002 and P004 both remain in
the gallery, so body appearance keeps re-matching later tracks to P004 and
face has to re-correct each time (visible at frames 2948 → 2995). The natural
follow-up is a **face-driven gallery merge**: when a face match shows two
person_ids are the same physical person, merge their rows instead of only
relabelling the track. That would collapse this run from 5 persons to the
correct 4.

## Scene-geometry probe (step 6) — run 2026-08-14

Prompted by the observation that the deployment's workers all wear the same
uniform, we asked how much identity churn is explainable by **space alone**,
with no appearance model. Motion-only ByteTrack trajectories over the same
4484 frames (35 tracklets ≥5 frames; peak 5 simultaneous, ≤3 for 95 % of
frames):

| measurement | value | meaning |
|---|---|---|
| births at frame border | **5 / 35** | genuine entries into view |
| births **mid-scene** | **30 / 35** | a person cannot materialise mid-floor |
| deaths **mid-scene** | 31 / 35 | …nor vanish mid-floor |
| mid-scene births with a spatially feasible predecessor | **26 / 30** | recoverable from geometry alone |

Feasibility = a dead tracklet whose last foot position is within a
walking-speed budget (1.8 m/s, scaled by box height as px-per-metre) of the
new tracklet's first foot position, within 20 s.

**Reading.** ~87 % of the identity churn is a geometric re-appearance, not a
recognition problem — the scene is a small enclosed control room (fisheye
ceiling camera, one door, and three large static occluders: the centre
cabinet, the black rack, the desk/monitor bank). 35 tracklets are generated
for roughly 4 people. The pipeline currently answers *"which gallery entry
does this crop resemble?"* (open-set retrieval, unbounded gallery) when the
scene poses *"which of the ≤5 people already in this room is this?"*
(closed-set assignment under reachability). The second question is far
better conditioned and barely depends on appearance quality.

**Also note clothing is unstable *within* a single session** — sampling frames
300 / 1500 / 3000 shows the same people removing outer layers indoors (a red
puffer worn, then draped over an arm 80 s later). Body appearance is therefore
degraded in both directions here: different people look alike (uniforms), and
the same person changes appearance across minutes.

**Implication for the roadmap.** Demote body appearance from *the decision* to
*a tiebreaker within a spatially feasible candidate set*; promote scene
structure to primary. Highest payoff / lowest effort first: (a) annotate the
door region and forbid new-identity enrollment for mid-scene births;
(b) occluder map with reappearance priors; (c) occupancy bookkeeping against
room capacity; (d) fisheye→ground-plane rectification so speed constraints are
metric rather than pixel-based; (e) workstation/chair zones as semantic
anchors across seated detector dropouts. Face stays as the arbiter — it is the
only cue that survives a uniform — but at 23 % post-gate coverage it cannot
carry identity alone.

**Limit to be explicit about:** all of (a)–(e) are *session-local*. They give
near-perfect continuity within one recording and nothing across days. Only
face (or a badge/RFID) can answer "is this Monday's worker".

## Metric scene geometry (step 7) — run 2026-08-14

Step 6's ranked list opened with "annotate the door region". That is rejected:
the door is often outside the frame, and a per-site annotation is not a
product. Step 7 therefore builds the geometry **automatically, from the
footage itself** — no checkerboard, no drawn zones, no site-specific input.

### Calibration (`scene_geometry.py`, `scripts/calibrate_scene.py`)

Input: the video plus its own motion-only tracklets. Output: a metric ground
plane.

1. **Radial distortion** — plumb-line fit. Scan the division-model `k1` that
   maximises total squared LSD segment length on a median background frame;
   barrel distortion breaks straight edges into short pieces, so undoing it
   merges them. Recovered `k1 = -0.185`.
2. **Horizon** — *not* by intersecting head→foot lines. Their angular spread
   here is only **5.8°** (close-range indoor camera, near-parallel verticals),
   so the classical vertical-VP intersection is ill-conditioned and returned a
   negative `f²`. Instead we use the fact that apparent height of a
   fixed-stature upright on a plane is **linear in image coordinates**:
   fit `h = a·x + b·y + c` by RANSAC + Huber IRLS over 764 observations. The
   horizon is where that plane crosses zero. rel-rms **3.4 %**.
3. **Focal length** — from **stature position-independence**: a person must
   not measure taller because they stood closer. Scan `f`, minimising the
   IQR/median of stature gains. This is the primary estimator and is fully
   self-contained.
4. **Metric scale** — assume a population mean stature of 1.70 m.

**Depth model as an independent check, and why it lost.**
Depth-Anything-V2-Metric-Indoor gave `f = 790 px` with a 14.7° plane-vs-horizon
disagreement, and under it stature drifted with range: 1.80 m at 0–1 m,
1.64 m at 1–2 m, 1.45 m at 2–5 m. That is a *systematic* bias, not noise — at
the wrong `f`, stature encodes **position rather than person**, which silently
destroys it as an identity cue. The stature-consistency estimator gives
`f ≈ 1172–1354 px` (FOV 71–79°), halves the stature spread
(IQR/med 0.128 → 0.064) and removes the range bias (near/far 1.153 → 1.003).
Depth is retained as an optional cross-check only.

**Two-way validation:**

| quantity | stature-scaled | depth model | agreement |
|---|---|---|---|
| camera height | **2.88 m** | 2.70 m | **94 %** |
| floor extent (2–98 %) | 4.2 × 2.7 m | — | matches the visible room |
| stature p10/p50/p90 | 1.59 / 1.70 / 1.82 m | — | physically plausible |

### A. Reachability in metres vs pixels (`bench/06_eval_metric_geometry.py`)

Step 6 asked "does this birth have *a* plausible predecessor?" and got 26/30.
That is the wrong question — a loose budget makes everything feasible. What
association needs is a **small candidate set**.

| budget | ≥1 candidate | **exactly 1** | mean candidates |
|---|---|---|---|
| pixel (box-height scaled) | 86.7 % | **10.0 %** | 3.03 |
| metric 1.8 m/s on the floor | 86.7 % | **23.3 %** | 3.07 |

Metric geometry **more than doubles** unambiguous links but barely shrinks the
mean candidate set. Sensitivity over the gap window explains why: the
unambiguous rate peaks near a **~5 s** gap (px 43.3 %, metric 46.7 %) and
decays after. In a 4 × 3 m room, a 1.8 m/s limit over 20 s reaches every point
in the room — the constraint is vacuous at long gaps regardless of units.
**Use a short reachability window (~5 s); beyond that, geometry stops
discriminating and appearance/face must carry the link.**

### B. Metric stature as a ReID cue — negative, with a diagnosed cause

Same pair construction as step 02, so directly comparable:

| cue | AUC | EER |
|---|---|---|
| face (insightface) | 0.907 | 0.127 |
| CLIP-ReID body | 0.842 | 0.227 |
| **metric stature** | **0.565** | **0.467** |

Same-person `|Δh|` median **8.9 cm** vs different-person **11.4 cm** — noise
swamps signal. Several tracklets read implausibly short (1.36–1.41 m).

**Cause, tested and confirmed.** Hypothesis: *in a cluttered room the bottom of
the bounding box is furniture, not feet* — the person is then localised as if
standing further away and measures short. Gridding the image by foot position
and taking per-cell median stature (`occluder_map.png`):

- spatial spread of per-cell median stature: **1.15 m**
- **within-track** stature range across cells: median **0.19 m** over 14 tracks

The same person cannot change height, so that 0.19 m is *pure position
artefact* — and it is **larger than the 11.4 cm between-person signal**. The
low cells coincide visually with the desk/monitor bank and the cabinet fronts.

This turns the negative into a positive architectural result: **the stature-error
map _is_ an occluder map, derived from data rather than drawn by hand.** And
occlusion-aware foot estimation is a **prerequisite**, not a refinement — it
gates stature as a cue *and* the metric floor positions that step A depends on.

### Where this leaves the roadmap

1. **Occlusion-aware localisation first.** Either learn the per-cell floor
   offset from the stature-error map and correct the foot point, or localise
   from the **head** point (rarely occluded from a ceiling camera) plus the
   ground plane. Everything metric is downstream of this.
2. **Short-window metric reachability (~5 s)** as a hard gate on association —
   already worth 10 % → 23 % unambiguous, and it should improve once foot
   points are corrected.
3. **Re-test stature after correction.** The 0.19 m artefact is removable; if
   it drops below ~5 cm, stature becomes a real back-facing, uniform-immune cue.
4. Face stays the cross-session arbiter (23 % coverage — an arbiter, not a
   carrier). Multi-day validation remains the one thing no amount of scene
   geometry can substitute for.

## Depth-based scene understanding (step 8) — run 2026-08-14

Step 7 ended on a blocker: stature failed as a cue (AUC 0.565) because the
bounding-box bottom is often furniture rather than feet, producing a 0.19 m
*within-track* positional artefact — larger than the 11.4 cm that separates
two different people. Step 8 removes that blocker using a metric depth model
for what it is actually good at: **static scene structure**, not scale.

### The two new components

**`scene_depth.py` — `SceneModel`.** Lifts a monocular metric depth map of the
people-free background into the calibrated ground-plane frame, giving *height
above the floor at every pixel*. The depth model's absolute scale is **not**
trusted (step 7 caught it returning f = 790 px against a true 1354 px); it is
rescaled against floor points we already have for free — the observed foot
positions of tracked people — and the residual of that rescaling is reported
rather than assumed away. Fitted scale ×0.949, floor residual 18 cm (1σ).

Sheet `05_occluder_map.png` is the payoff: the walkable tile floor comes out
green and every cabinet, rack, desk, chair and floor-bag comes out red. **This
is the occluder map, derived from footage rather than drawn.** It is exactly
the "drawn zone" capability, without the drawing, so it transfers to a new
site by running a script.

**A sign bug this exposed.** `GroundPlane` is fitted only up to a global sign —
stature and distance are both invariant under `X → −X`, so nothing in steps
6–7 could notice that `floor_xy` was reconstructing the floor *behind* the
camera. Harmless for everything that only measures lengths (all previously
reported numbers stand, and were re-verified), fatal for depth fusion and for
drawing geometry back into the image. `physical_frame()` now resolves the
branch explicitly. Round-trip checks: floor→pixel→floor exact, stature
round-trips to 1.7000, back-projected verticals agree with the fitted vertical
vanishing point to 0.00°.

### Why stature really failed — two causes, not one

Occlusion gating alone barely moved the artefact. Comparing on the **same**
tracks (the earlier before/after compared different surviving populations,
which measured nothing):

| | within-track stature range |
|---|---|
| raw | 20 cm |
| + occlusion gating (feet visible, 56 % of obs) | 20 cm |
| + position-bias field | **9 cm** |

Gating fixes *some* tracks dramatically (trk 4: 18.7 → 3.6 cm; trk 60's median
stature 1.41 → 1.52 m) and leaves others untouched — so a **second** error
source of 15–20 cm survives even when the feet are plainly visible.

**`StatureField`.** That second source is systematic, not noise. Fitting
`log(stature) = person + cell` by median polish separates the person term (the
cue) from a per-cell term (a lens/plane/detector artefact masquerading as one).
The decisive test is reproducibility: fitting the field independently on two
**disjoint halves of the tracklets** gives **r = +0.934** across shared cells
(sheet 09). Noise does not reproduce across a track split. The field spans
about ±10 %, i.e. ±17 cm at 1.7 m, which is the whole artefact.

It consumes no labels — the "person" term is keyed on tracker output — so it
is fitted on site from unlabelled footage, like the rest of the calibration.

### Result: stature becomes a usable cue

Same pair construction as step 02. The bias field is fitted on one fold of
tracklets and scored on the other, so it cannot leak:

| variant | AUC | EER | same-person | diff-person |
|---|---|---|---|---|
| 1. raw foot-box stature | 0.570 | 0.459 | 8.7 cm | 11.7 cm |
| 2. + occlusion gating | 0.598 | 0.435 | 8.8 cm | 12.8 cm |
| **3. + position-bias field (held-out)** | **0.853** | **0.200** | **4.3 cm** | **15.2 cm** |
| *reference:* CLIP body | 0.842 | 0.227 | | |
| *reference:* face | 0.907 | 0.127 | | |

Same-person spread halves while different-person spread grows — the signal was
there all along, buried under a geometry error.

**Do not over-read the comparison with CLIP.** Only 119+/121− pairs survive,
from ~12 tracklets. Bootstrapping over *tracks* (pairs share tracklets and are
not independent) gives a 95 % CI of **[0.572, 0.972]**. P(AUC > 0.70) = 0.94,
so the improvement over raw is solid; P(AUC > CLIP's 0.842) = 0.61, a coin
flip. The honest claim is **"stature went from useless to genuinely useful",
not "stature beats appearance"** — and it needs more footage to sharpen.

What makes it worth having anyway is that it is *orthogonal*: it is immune to
uniforms, works from behind, and survives a coat coming off — precisely the
three failure modes of body appearance and face on this deployment.

### Commissioning tool

`scripts/commission_scene.py` is the per-camera setup path. One command, no
annotation:

    python scripts/commission_scene.py --name <site> \
        --video <video.mp4> --traj <tracklets.json>

It writes `calib/<site>/`: the ground plane, the scene depth model, the
stature field, a `commission.md` with 10 pass/warn checks, and an image pack
`01..09_*.png`. **Sheet 08 is the acceptance test** — a 1 m floor grid and
1.70 m human silhouettes drawn back into the image. If the squares look square
on the floor and the sticks match real people, the metre scale is real; every
downstream number inherits it.

Current status on `gunsan_test`: **8/10 checks pass**. The two warnings are
the 18 cm depth-vs-floor residual (depth model precision, not correctness) and
the 9 cm residual stature artefact (target was < 8 cm).

---

## 9. The scene model finally reaches the association logic

Steps 7 and 8 built a metric floor and an automatically-derived occluder map,
and then used them for *diagnostics only*. `scripts/track_rtdetr_db.py`
imported neither module. Every rebind decision was still made by
`ghost_pool._spatial_prior`: an isotropic Gaussian around
`last_bbox_centre + velocity * elapsed`, **in pixels**, with
`sigma = 4 * sqrt(elapsed)`.

That prior is wrong in three independent ways on a fisheye ceiling camera. A
pixel is worth wildly different metres near the lens and at the far wall. The
Gaussian spreads probability straight *through* solid furniture, so someone who
stepped behind a cabinet is scored as though they could re-emerge from the
middle of it. And its sigma grows without bound, so in a 4x3 m room it goes
uniform after a couple of seconds - it stops constraining anything exactly when
appearance is weakest, which on a uniformed workforce is the whole problem.

`reachability.py` replaces it with the obvious physical statement: someone who
vanished at one spot and re-appeared at another had to **walk between them,
around the furniture**.

### The walkable map is not the depth occluder map

The natural move is to reuse `SceneModel.free_space`. On this clip that would
have been a serious mistake:

| walkable-map source | area | share of observed footfall it contains |
|---|---|---|
| depth floor mask, `free_space` | 2.6 m2 | - |
| depth floor-fraction >= 0.3 | 4.9 m2 | **25 %** |
| + cells where feet were observed | 7.6 m2 | **85 %** |

The `obstacle` grid reduces each cell by the *tallest* surface projecting into
it, so one grazing wall pixel condemns a cell, and 18 cm of depth residual
smears every furniture edge. A prior built on it would have vetoed most true
rebinds.

So the map keeps three states, and the asymmetry is deliberate - it is cheap to
call a cell free and expensive to call it blocked, because a false BLOCKED
vetoes a true rebind:

* **FREE** - depth sees floor here, *or* feet were observed here.
* **BLOCKED** - no floor pixels, mostly above 0.8 m, and no feet ever observed.
* **UNKNOWN** - everything else. Traversable at 2x cost, never free.

Keeping UNKNOWN distinct from BLOCKED is what makes this switchable-on before a
site has any footage: with no evidence every cell is UNKNOWN, the prior degrades
to a plain metric distance budget, and that is already better than the pixel
Gaussian. Observed footfall then promotes cells to FREE and the map sharpens
itself, run over run, persisted to `calib/<site>/reachability.npz`. No
annotation, at any point.

**Validation of the BLOCKED tier**: it covers 10.1 m2 in 7 connected occluders
and contains **0.0 %** of all feet-*visible* observations - while containing
~16 % of head-*inferred* ones, i.e. exactly the people standing behind the
furniture whose inferred position lands on its footprint. Query points are
therefore snapped to the nearest walkable cell, never rejected.

### Evaluation without labels

Two ground truths are free on unannotated footage. Two samples of one tracklet
are the same person; two tracklets alive **in the same frame** are different
people, since one person cannot be two boxes at once (63 such pairs here).

A pooled same-vs-different AUC turns out to be the wrong summary: the two
priors decay with gap length at completely different rates, so pooling scores a
short-gap impostor against a long-gap true match - a comparison that never
happens. What happens in deployment is one ghost, one elapsed time, several
candidates on screen, pick one. So `bench/08_eval_geodesic_prior.py` ranks the true
candidate against the distractors alive in that same frame, per gap band,
weighted by the gap distribution actually observed (median 40 frames, 2.7 s).

### Result

Top-1 ranking accuracy of the spatial prior alone:

| gap | 5-20f | 20-60f | 60-150f | 150-600f | **weighted** |
|---|---|---|---|---|---|
| pixel Gaussian (before) | **93.8** | 78.3 | 59.6 | 36.1 | 65.5 |
| geodesic floor (after) | 91.6 | 78.3 | **69.9** | **52.8** | **72.5** |

The win is concentrated exactly where it should be. Below ~20 frames the pixel
prior is genuinely sharper - the box has barely moved and metric localisation
noise is larger than the displacement being measured - so the shipped combiner
**hands over at that crossover** rather than blending, which measured worse than
either. Above it the pixel prior has decayed to noise and the geodesic one
carries the decision, by +10 points at 4-10 s and +17 points beyond.

The portal term ("went in one side of the cabinet, came out the other") is worth
about +1 point on its own and is kept.

### The walking-speed veto is a short-gap tool, and that is all it is

Separately from the prior, a candidate that could not have been *walked to* is
vetoed outright - physically impossible outranks any appearance agreement.

| gap | true continuations vetoed (cost) | impostors vetoed (benefit) |
|---|---|---|
| 5-20f | 1.5 % | **27 %** |
| 20-60f | 0.0 % | 0.4 % |
| >60f | 0.0 % | 0.0 % |

Past ~1.3 s the budget `v_max * elapsed` exceeds the room and the veto becomes
inert - confirming step 7's warning rather than escaping it. It is kept because
its purpose is to block *appearance*-driven false rebinds, which is a benefit a
spatial-only ranking cannot see: it can only show the cost. `slack_m` was raised
0.6 -> 1.0 m, which halves the false-veto rate at a third of the catch.

### The proxy was hiding how big the win is

The numbers above use a *proxy* for "same person": two samples from inside one
tracklet. That is the one case the long-gap rebind path never sees - the person
never actually left the tracker's hands, so they are still roughly where they
were, which is precisely the regime a pixel Gaussian is good at. The proxy was
therefore biased *against* the feature.

The clip was then hand-labelled per tracklet (`scripts/label_tracklets.py`
renders a contact sheet of crops; 33 labels, a few minutes of clicking). That
turns SAME into a real re-entry: tracklet *j* starts after tracklet *i* ends,
same body. Re-running with `--labels`:

| spatial prior | 5-20f | 20-60f | 60-150f | 150-600f | weighted |
|---|---|---|---|---|---|
| pixel (before) | 76.9 % | 83.0 % | 59.9 % | 29.0 % | 61.7 % |
| geodesic (ships) | 76.9 % | **87.0 %** | **79.3 %** | **85.1 %** | **81.5 %** |

+19.8 points weighted, against +7.0 under the proxy. The long-gap band goes
29 % -> 85 %, i.e. from worse-than-a-coin-flip to reliable. This is the result
the feature was built for and the proxy could not show it: a real re-entry means
the person *walked somewhere*, usually around the furniture, and walking around
furniture is the one thing a geodesic distance models and a pixel Gaussian
cannot.

The veto's measured benefit drops to ~0 here, but that is a change of
denominator, not a regression - with labels the DIFF set is any two different
people at any time offset, not only ones caught co-existing, so it is dominated
by pairs far apart in time where the veto is inert by design. Its cost also
goes to 0.0 %, which is the number that matters for keeping it.

### What the labels said about the tracker

4 people produced 33 labelled tracklets - **8x fragmentation**, and one person
alone fragmented into 18. That is the actual size of the problem this whole
subsystem exists to undo.

Checked for ID switches inside tracklets by looking for same-person tracklets
sharing frames: 17 such pairs, every one of them 1-17 frames at a handover
boundary. That is the tracker spawning the replacement track a beat before
killing the old one, not two bodies and not a switch. So the tracklets are
clean, and the within-tracklet proxy used above was at least honest.

### Honest limits

* One clip, 33 labelled tracklets, 4 people, one room. Every number here needs
  re-measuring on a second site before it is a general claim.
* 4 people is a small distractor pool. Ranking accuracy will fall in a busier
  room and the gap between the two priors may not hold its shape.
* The occluder map is **static furniture only**. Person-occludes-person is not
  in it and is likely the larger source of short gaps in a 5-person room. Same
  machinery would handle it from live tracks; not built.
* Occluder boundaries inherit the 18 cm depth residual, so they are fuzzy at
  roughly +-20 cm. Fine for region membership, marginal for exact portals.
* Within-session only. Nothing here touches the multi-day problem.

### Using it

    python scripts/track_rtdetr_db.py --input <video> --scene calib/<site> ...

`--disable-geo-prior` restores the pixel Gaussian for A/B. Without `--scene`
the pipeline is unchanged.


## 10. The clip has people changing clothes, and that breaks the embedding

The user, on seeing the fragmentation numbers: *"this is a difficult vid as
person change their clothes but in reality worker also change their clothes a
lot."* That is not an excuse for the clip, it is a statement about the deployment
domain — shift uniforms, jackets on and off, PPE — and it makes a shipped default
suspect, so it was measured rather than assumed.

The labels make the measurement possible for the first time. Same-person crop
pairs split into two regimes: pairs from *inside* one tracklet (seconds apart,
same outfit by construction) and pairs *across* two tracklets (minutes apart,
outfit unknown).

| cosine distance | n | p10 | p50 | p90 |
|---|---|---|---|---|
| SAME person, within one tracklet | 3969 | 0.086 | 0.279 | 0.478 |
| SAME person, across tracklets | 20368 | 0.284 | **0.454** | 0.665 |
| DIFFERENT people | 56264 | 0.424 | 0.671 | 0.817 |

**59 % of same-person cross-tracklet pairs are at or above the 10th percentile of
*different people*.** Worst pairs reach d = 0.98, further apart than most
strangers. As an AUC:

| embedding as a discriminator | AUC |
|---|---|
| SAME within-tracklet vs DIFFERENT | 0.944 |
| SAME across-tracklets vs DIFFERENT | **0.803** |

The 0.944 is essentially the number step 3 reported when the embedding backbone
was chosen. It was measured in the regime where the tracker has already solved
the problem. The 0.803 is the regime the gallery actually operates in. **The
benchmark that justified the appearance model was measuring the easy case.**

### The embedding veto is refusing one in four true re-entries

`ScoringWeights.embedding_veto_max_dist = 0.55` hard-kills any rebind above that
distance. Against the labels:

| veto threshold | true re-entries refused (cost) | impostors blocked (benefit) |
|---|---|---|
| 0.55 (**shipped**) | **24.5 %** | 73.2 % |
| 0.65 | 11.4 % | 55.4 % |
| 0.75 | 4.1 % | 25.3 % |
| 0.85 | 0.7 % | 5.7 % |

There is no clean operating point — the curves fall together, which is just
AUC 0.803 restated. The veto was tuned when its cost was invisible; the cost is
a quarter of the correct answers.

### It is not only clothing

Splitting cross-tracklet pairs by time apart:

| frames apart | median d | % over the 0.55 veto |
|---|---|---|
| 0-150 | 0.444 | 25.7 % |
| 150-600 | 0.418 | 17.5 % |
| 600-1500 | 0.465 | 25.8 % |
| 1500-3000 | 0.476 | 27.9 % |
| 3000+ | **0.633** | **73.5 %** |

Flat until ~200 s, then a cliff. Two different failures are stacked:

* The **cliff past 3000 frames is the clothing change** — it needs time and time
  off-camera, so it can only appear at long gaps.
* The **flat ~25 % below it is not**. Nobody changes clothes in 10 seconds. That
  is viewpoint and pose — the user's earlier point that *"majority of the time it
  is people back"*. The embedding is weak here before clothing is involved at all.

A gap-conditioned veto threshold would therefore only address the smaller of the
two problems.

### What this means for the design

It is a second, independent argument for the direction step 9 already took. The
signals that survive a clothing change are the ones that are not appearance:
where the body can physically be (geodesic reachability, now +19.8 points),
stature (`StatureField`, already metric and garment-independent), and face when
it is visible. Appearance is a tiebreaker among the physically plausible, not a
gate on them.

Concretely: the veto should move toward 0.75 and the embedding should act
through its weighted score term rather than as a hard kill. Not applied yet —
it trades impostor blocking for recall, on 4 people in one room, and that is a
deployment call rather than a benchmark one.

    python bench/09_eval_clothing_change.py --labels labels/gunsan_test.json


## 11. Pose is real, but pose-matching does not rescue appearance

The user's follow-up: *"appearance works well only if we know the pose - compared
exact pose. Generally we need to understand the space and the movement of the
human inside the vid in order to track them."*

The first half predicts a concrete, free win. Step 9 already computes metric
ground-plane position and velocity per track, so the *viewing aspect* - the
angle between where the person is walking and the direction the camera sees them
from - costs nothing extra. If the embedding is largely encoding which side of
the body is visible, then restricting comparisons to pose-matched observations
should sharpen it.

    aspect ~ 0     walking away        -> back
    aspect ~ pi/2  walking across      -> profile
    aspect ~ pi    walking toward      -> face

### It does not work across tracklets

Same person, different tracklets - the case that matters - binned by how much
the viewing aspect differs:

| aspect difference | n SAME | median d | n DIFF | median d | AUC |
|---|---|---|---|---|---|
| 0-30 deg | 2935 | 0.438 | 8654 | 0.629 | 0.762 |
| 30-60 deg | 1092 | 0.473 | 3366 | 0.633 | 0.728 |
| 60-90 deg | 739 | 0.437 | 2376 | 0.643 | 0.791 |
| 90-180 deg | 3901 | 0.455 | 10750 | 0.637 | 0.766 |

Flat. Matched pose is no better than opposite pose (AUC 0.762 vs 0.766). Gating
on pose would buy nothing.

### But the instrument is not broken - the control proves pose is real

A flat table has two readings: pose does not matter, or the aspect estimate is
noise. Within *one* tracklet the person and the outfit are fixed, so aspect is
the only variable left, and pose must show up there if it exists at all:

| aspect difference | n | median d |
|---|---|---|
| 0-30 deg | 765 | **0.275** |
| 30-60 deg | 246 | 0.300 |
| 60-90 deg | 152 | 0.346 |
| 90-180 deg | 802 | **0.362** |

correlation(aspect difference, embedding distance) = **+0.196**

Monotone, and worth 0.087 of cosine distance from front-vs-back alone. **The
user is right that the embedding is partly a pose detector.** The instrument
works.

### The synthesis

Pose is real but *second-order*. Across tracklets the clothing change is the
larger effect and buries it - which is why the first table is flat while the
control rises. Pose gating would improve short-gap matching, which is the regime
the tracker already handles, and do nothing for the re-entry case it was wanted
for.

So the half of the user's statement that survives measurement is the second
half, and it survives it emphatically. Appearance cannot be repaired by
conditioning it on pose, because the thing breaking it is not pose. What is left
is exactly what step 9 built: understanding the space and the movement. Ranked
by what actually survives a uniform change:

1. **Where the body can physically be** - geodesic reachability. Measured
   +19.8 points, and completely immune to both clothing and pose.
2. **Stature** - `StatureField`, metric, garment-independent, back-compatible.
   Built in step 6 and still under-used by the association logic.
3. **Face** - correct when visible, but the user's constraint stands: *"majority
   of the time it is people back."*
4. **Appearance** - a tiebreaker among physically plausible candidates. Not a
   gate, and not a primary key.

    python bench/10_eval_pose_conditioning.py --scene calib/gunsan_test \
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json


## 12. Metric stature does not work as an identity cue — do not wire it in

Step 11 ended by ranking stature as the second-best clothing-invariant signal and
proposing to replace the ghost pool's pixel-height term with it. Measured first,
as promised. It does not hold up.

`StatureField` gives a bias-corrected metric height per observation; per tracklet
we take the median over feet-visible frames only (an inferred foot point would
make stature circular — the head is what placed the foot).

| person | tracklets | median height | spread across own tracklets | per-tracklet IQR |
|---|---|---|---|---|
| 1 | 6 | 1.667 | **0.244** | 0.130 |
| 2 | 1 | 1.748 | - | 0.097 |
| 3 | 4 | 1.570 | 0.093 | 0.081 |
| 4 | 6 | 1.575 | 0.120 | 0.067 |

The table answers itself. Persons 3 and 4 differ by **5 mm**, while person 1
disagrees with *itself* by 244 mm across its own tracklets. The noise floor is
larger than the signal.

| set | n | p50 abs difference | p90 |
|---|---|---|---|
| SAME person, across tracklets | 36 | 0.062 | 0.141 |
| DIFFERENT people | 100 | 0.086 | 0.196 |

**Stature AUC 0.574** — barely above chance, against 0.803 for the appearance
embedding in the same regime.

### It is not even complementary

A weak cue is still worth having if it fails on *different* pairs than the strong
one. Stature is genuinely independent — correlation with appearance distance is
**-0.024**, essentially zero — but too weak to contribute anything:

| combined score | AUC |
|---|---|
| appearance alone | 0.795 |
| appearance + 0.2 x stature | 0.799 |
| appearance + 0.5 x stature | 0.797 |
| appearance + 1.0 x stature | 0.791 |
| appearance + 2.0 x stature | 0.769 |

Best case +0.004, which is noise on 36 positive pairs. And on the specific pairs
where it was wanted — true re-entries the embedding veto would kill — the median
stature difference is 0.079 against 0.086 for genuine strangers. It cannot
overturn a bad veto because it cannot tell the two apart.

### Coverage is the second problem

Only **17 of 33** labelled tracklets got a stature estimate at all; the rest had
fewer than 8 feet-visible frames. In a room where the whole motivating problem is
occlusion, the cue is missing precisely when it is needed.

### Why this generalises beyond bad luck with 4 people

It is tempting to blame the sample — two of four people happen to be the same
height. But the measurement noise is the binding constraint, not the sample.
Per-tracklet IQR runs 0.067-0.130 m and cross-tracklet disagreement reaches
0.244 m, while adult human stature has a standard deviation of roughly 0.07 m
within a gender. **The instrument is several times noisier than the population it
would have to resolve.** No amount of extra people fixes that.

Stature's honest role is a coarse *impossibility* filter of the same kind as the
walking-speed veto — refusing to match 1.55 m against 1.90 m — not a ranking
signal. At ±0.12 m it cannot do more, and the ghost pool's existing `w_height`
term is already doing approximately that job.

**Decision: the scorer is left alone.** The step 11 ranking is corrected — after
geodesic reachability there is no second strong clothing-invariant cue in this
system. That is the finding, and it makes the appearance-veto question from
step 10 more urgent rather than less, because there is nothing else to fall back
on.

    python bench/11_eval_stature_cue.py --scene calib/gunsan_test \
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json


## 13. A real movement model — and why it loses here

The user, rejecting appearance outright and naming what should replace it:

> *"tracking visuals does not work as its camera view so most of the time you
> could not see the face, and the clothes change, so totally makes sense - even
> for human like me i can not tell. But what i can do is to visualize the space
> and predict the movement and direction of the tracked human, which currently
> we don't have."*

The charge is fair. Step 9 predicts *reachability* - could they be there - not
*movement*. Its forward prediction is 0.6 s of straight-line extrapolation,
after which the belief is an isotropic blob spreading along geodesic distance.
It has no model of how people move through this particular room.

`motion_prior.MotionPrior` is that model. The state is **(cell, heading)** rather
than cell, so momentum is represented: a person entering a gap keeps going
through it. The turn distribution is learned per cell from the tracker's own
output - no annotation, so the site-generic constraint holds - and spatially
smoothed, since traffic direction is a property of a region rather than of a
10 cm square. Propagating that chain for the actual elapsed time gives
P(cell | last position, last heading, elapsed): a genuine forward prediction,
which is a thing a geodesic distance cannot express however it is reweighted.

It runs fast enough to ship - 24k states, a 130k-entry sparse chain, sub-100 ms
to fit and microseconds per cached query.

### It loses, clearly

| spatial prior | 5-20f | 20-60f | 60-150f | 150-600f | weighted |
|---|---|---|---|---|---|
| pixel | 76.9 % | 83.0 % | 59.9 % | 29.0 % | 45.4 % |
| **geodesic (ships)** | **93.8 %** | 87.0 % | **79.3 %** | **85.1 %** | **84.6 %** |
| motion | 67.7 % | 85.1 % | 58.0 % | 63.2 % | 65.9 % |

### Why: there is almost no movement to predict

| gap | median displacement between vanishing and reappearing | p90 |
|---|---|---|
| 0.3-1.3 s | 0.26 m | 1.33 m |
| 1.3-4.0 s | 0.23 m | 0.79 m |
| 4.0-10.0 s | 0.20 m | 1.38 m |
| **10-40 s** | **0.38 m** | 1.58 m |

A 0.6 m/s walker unimpeded for 40 s covers 24 m. These people cover 0.38 m.
Consistently: 53 % of all observed inter-frame speeds are below 0.15 m/s.

So the re-entry events in this clip are not *"person walks behind the shelf and
emerges on the far side"*. They are *"person stands roughly still, the detector
loses them, the detector finds them again"*. Against that, a distance-decay prior
centred on the last known position is close to optimal, and any model that lets
the walker travel is strictly worse - it spends probability mass on a room the
person never crossed.

This does not refute the user's intuition; it says the intuition describes a
situation this clip does not contain. In footage with corridors, doorways, and
people actually transiting, the ordering could plausibly reverse - the machinery
is built and benched, so re-running it is one command. It is **not wired into the
pipeline**, because shipping it here would cost 19 points.

### The correction that matters more than the result

Counting the evidence properly:

| | |
|---|---|
| cross-tracklet pairs the bench reports | 1801 |
| **independent re-entry events behind them** | **50** |
| resampling factor | **36x** |
| of those, consecutive re-entries (the deployment event) | 26 |

Each event is sampled at up to 40 offsets into the new tracklet, so the pair
count is not the evidence count. **Every accuracy figure in sections 9, 11 and 13
rests on 50 events, not on thousands of pairs**, and the confidence intervals are
correspondingly wide - a single band like `5-20f` at 4 % of the mix is a handful
of events. The step 9 headline (+19.8 points for geodesic) should be read as a
strong directional signal on one clip, not as a precise number. It was reported
without this caveat and should not have been.

    python bench/12_eval_motion_prior.py --scene calib/gunsan_test \
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json


## 14. Whole-path stitching beats greedy pairwise by 25 IDF1 points

The user's third structural point, and the sharpest:

> *"the fact that iou only track consecutive frames is really limited - it
> should track the whole motion path of one person, right? as again we need to
> understand how does that person presence in the space."*

Everything downstream of the detector here makes *local* decisions. BoT-SORT
associates frame N to frame N+1. `ghost_pool` rebinds one lost track to one new
track, greedily, at the moment the new track appears, and can never revise it.
Both answer "does this box continue that box". Neither asks whether the
resulting **path** is a plausible way for one person to have moved through the
room. Three things are structurally unavailable to a pairwise matcher:

* **Transitivity** — if A-B and B-C are both good links, A-C is implied, and it
  may be physically impossible. Nothing checks.
* **Exclusivity** — two tracklets overlapping in time cannot be one person.
  Locally that blocks one pair; globally it propagates, because ruling out A-B
  frees B for C.
* **Revision** — a match made at frame 400 is permanent, however obviously wrong
  frame 900 makes it.

`trajectory_stitch.py` replaces the greedy pass with min-cost flow over a
time-ordered tracklet graph. Each unit of flow from source to sink is one
person's path through the space - literally the object the user asked us to
model - and link costs come from the step 9 geodesic prior, so the scene model
is what holds the paths together.

| method | identities | frag | purity | IDF1 |
|---|---|---|---|---|
| no stitching (raw tracklets) | 30 | 7.50 | 100.0 % | 40.9 % |
| greedy pairwise (what ships) | 13 | 3.25 | 80.6 % | 44.8 % |
| **global min-cost flow** | **5** | **1.25** | **89.9 %** | **69.4 %** |
| (perfect) | 4 | 1.00 | 100 % | 100 % |

**+24.6 IDF1 over greedy**, and it recovers 4 people as 5 identities instead of
13. It is also *more* pure than greedy while merging far more aggressively -
which is the signature of the constraints doing real work rather than a
threshold being loosened. The result is flat across the birth-cost knob from 1.5
to 6.0, so it is not a tuned number.

This is the largest single improvement measured in this project, and it came
from changing the shape of the decision rather than from any new signal.

### Two bugs this exposed

**1. The min-cost flow needs a coverage reward.** Without a negative weight on
each tracklet's internal edge, the cheapest flow is the one that explains as few
tracklets as possible - the solver has no reason to route through a node at all.
The first implementation returned every tracklet as its own identity and looked
like a clean negative result. It was a modelling error, not a finding.

**2. The shipped prior assumed people leave, and they don't.** Link priors came
back at ~0.998 for essentially every pair: completely unable to discriminate.
The cause is that `rebind_prior` spread the position belief *ballistically*,
sigma = v_typ * elapsed. Measured unsupervised - displacement over a lag inside
tracklets, no labels needed:

| elapsed | rms displacement | ballistic (0.6 m/s) |
|---|---|---|
| 1.3 s | 0.50 m | 0.80 m |
| 5.3 s | 1.07 m | 3.20 m |
| **40 s** | **1.68 m** | **24.00 m** |

A power-law fit gives **sigma(t) = 0.43 * t^0.43** — essentially *diffusive*
(exponent 0.5), not ballistic (1.0). People in a room mill about; they do not
depart in a straight line. The ballistic model was 14x too loose at 40 s, which
flattened the prior into uselessness exactly where discrimination was needed.
This is the same physical fact step 13 found from the other direction.

`ReachParams` now defaults to `spread_a=0.45, spread_b=0.5`, with
`ballistic_spread=True` restoring the old behaviour for A/B. On the step 8
ranking this is a small net gain on its own (81.5 % -> 81.7 % weighted, with
20-60f 87.0 -> 89.9 and 150-600f 85.1 -> 88.1), but it is what makes link costs
informative enough for stitching to work at all.

### Limits

* **Offline by construction.** Revising the past needs the future. This is a
  post-pass over a recording, not a replacement for the online tracker - the
  greedy ghost pool still has to carry the live case.
* 31 tracklets, 4 people, one clip, and the same 50-event caveat from step 13
  applies to the geodesic costs underneath it.
* Purity is 89.9 %, not 100 % - it does merge two people somewhere. At this
  scale that is one mistake, and which mistake matters more than the percentage.

    python bench/13_eval_trajectory_stitch.py --scene calib/gunsan_test \
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json

## 15. A second camera, and the calibration bug it exposed

New footage: `20260819/cam2` — a different room from gunsan, 1280x720 @ 10 fps,
35 clips (18 min). The clip of interest is `20260819_145840.mp4`, in which a
person puts a hi-vis vest on and takes it off on camera: the §10 clothing-change
failure mode, filmed deliberately.

**Tracking that clip is not the test.** 4 people, 4 tracks, zero fragmentation,
zero ID switches — including across the vest change, because nobody ever leaves
frame and BoT-SORT's motion carries the association. A 30 s clip in which
everyone is continuously visible contains no re-entry events, so it cannot
score the ReID prior at all.

What it did surface is a real bug in commissioning.

### Symptom
Commissioned from this camera, sheet 08 was visibly wrong — the 1.70 m sticks
leaned about 20 deg off vertical and the floor grid did not follow the floor.
The operator caught it from the sheet, which is what the sheet is for.

| | 30 s clip | full session | + fix |
|---|---|---|---|
| checks | 6/10 | 7/10 | **8/10** |
| camera height | 1.43 m | 1.78 m | **1.82 m** |
| horizon slope | — | +27.4 deg | **+8 deg** |
| depth vs floor | 42 cm | 79 cm | 37 cm |

### Cause
`fit_height_field` RANSACs the apparent-height plane h(u,v) and selects the
model with the most inliers. Inliers are *observations*, not *viewpoints*. This
room is an office: people sit. Three static seated tracks supplied 41% of all
5478 observations and ten supplied 81%, so the vote was decided by dwell time.
The winning plane fit somebody's chair and tilted the horizon 27 deg.

gunsan never showed this because its subjects walk through and leave.

### Fix
`dwell_weights()` bins observations by (track, floor cell) and weights each by
1/(bin population), so a track contributes roughly equal weight per *place it
was seen* rather than per second it lingered. `fit_height_field` takes the
weights in both the RANSAC vote and the IRLS refinement.

Discarding duplicates outright was tried first and rejected: it fixed cam2 but
cut gunsan from 764 to 369 observations and cost **7.5 points** of downstream
ranking accuracy (81.7% -> 74.2%). Sample-starved cameras cannot afford to
throw data away; weighting degrades gracefully where dedup does not.

### Effect on the shipped number
Re-running §9 on gunsan with labels, calibration only:

| prior | 5-20f | 20-60f | 60-150f | 150-600f | weighted |
|---|---|---|---|---|---|
| pixel | 76.9% | 83.0% | 59.9% | 29.0% | 61.7% |
| geodesic (before) | — | 89.9% | 76.9% | 88.1% | 81.7% |
| geodesic (after) | 76.9% | 85.1% | 86.1% | 90.2% | **84.9%** |

+3.2 points on a camera that had no visible problem, and gunsan's stature bias
field reproducibility rose r=+0.934 -> +0.951. The bug was quietly costing
accuracy everywhere; it was only *visible* where people sit still.

### Still open on cam2
`bias field reproducible` fails (r=+0.009 over 41 cells) and depth-vs-floor is
37 cm. Both look like properties of the room rather than of the method: 3.4 m2
of standable floor in a cluttered office means few people ever cross the same
cell twice. Worth revisiting if a less cluttered camera is commissioned.

## 16. A third camera, real re-entries, and the first clean test of whole-path stitching

Source: `20260529_OfficeCCTV_Raw/cam1` on the backup server — a third room, PPE
domain (hard hats, hi-vis vests), 1280x720 @ 10 fps. 13 clips over 40 min, one
of which (`20260529_143423.mp4`) is corrupt (truncated, no moov atom).

Commissioned `calib/office_cam1` from the 12 usable clips: 8/10 checks, camera
height 1.98 m, near/far bias 1.005, sheet 08 clean. Monocular depth is off by
2.5x (152 cm residual), so the occluder map here is not trustworthy even though
the ground plane is.

Tracked the **contiguous** 15.5 min block 14:35:21-14:50:50. The 14:29->14:35
gap was deliberately *not* spliced: concatenating across it would present a
six-minute absence as instantaneous and manufacture re-entry events.

Unlike everything benched before, people here leave and come back. The ghost
pool fired for real: 34 rebinds and 23 reachability vetoes.

### Ground truth for free
The operator states the footage contains **two people**. That is checkable
without labels. Two tracklets that share a frame cannot be one body, so the
temporal-overlap graph constrains the answer:

    edges: 1-2, 1-3, 1-4, 4-5, 5-6, 6-7

This is a path, and a path has exactly one 2-colouring: {1,5,7} and {2,3,4,6}.
So if there are two people, the partition is *forced*. No annotation needed.

### Result

| method | identities | correct |
|---|---|---|
| raw tracker (ships today) | 7 | no - 3.5x over-count |
| greedy rebinding (ghost pool) | 4 | **no - groups 2+3+5+7, mixing both people** |
| global stitch, birth_cost >= 4 | **2** | **yes - exactly {1,5,7} / {2,3,4,6}** |

Three independent routes agree on the same partition: the operator's count, the
overlap bipartition, and min-cost flow.

The greedy failure is the important half. It does not merely under-merge; it
merges *across* people, which is the error that actually costs an operator
trust. A pairwise matcher has no way to notice that binding 2->5 forces a
contradiction later at 5-6; the flow formulation cannot make that mistake
because exclusivity is a constraint on the whole assignment rather than a
tie-breaker at one edge.

This is the first evidence for the whole-path argument from footage that
contains the events it is about. On gunsan the stitcher was scored against
labels on 50 independent re-entry events; here it is scored against a
combinatorially forced answer, and it is exact.

### Caveat on the knob
`birth_cost` defaults to 2.0, which yields 3 identities here (and merges
1+6, a cross-person error). It needs >= 4.0 on this camera, whereas gunsan was
flat across 1.5-6.0. bc = 4-6 satisfies both, but a single default has now been
shown not to transfer for free, and the knob should be set per camera or, better,
derived from the observed link-probability distribution.
