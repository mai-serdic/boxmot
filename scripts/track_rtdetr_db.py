"""
RT-DETRv4 (ONNX) + BoxMOT BoT-SORT + Persistent Person Database
----------------------------------------------------------------
Two-tier pipeline:
  1. Short-term: BoT-SORT does frame-to-frame association with its own trk_id.
  2. Long-term:  IdentityResolver maps trk_id → persistent Person_XXX via a
     gallery-backed PersonDB that survives across runs/sessions.

Embeddings are extracted once per frame (OSNet x1.0 by default) and fed into
BoT-SORT via the `embs=` parameter, so we don't double-compute them.

Usage:
    python track_rtdetr_db.py \\
        --onnx  models/20260504_rtv4_hgnetv2_m.onnx \\
        --reid  osnet_x1_0_msmt17.pt \\
        --input videos/gunsan_test.mp4 \\
        --output runs/track/rtdetr_db.mp4 \\
        --gallery gallery/persons.npz
"""

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable so `boxmot` and `reid` resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import onnxruntime as ort

from boxmot.trackers.botsort.botsort import BotSort
from boxmot.reid.core import ReID

from reid.person_db import PersonDB, IdentityResolver
from reid.ghost_pool import GhostPool, GhostTrack, NewTrackInfo, ScoringWeights
from reid.reachability import Reachability, ReachParams
from reid.scene_geometry import GroundPlane
from reid.scene_depth import SceneModel, floor_from_boxes
from reid.floor_plan import FloorPlan
# face_anchor is imported lazily in run() — insightface is only needed with --face


# ─────────────────────────────────────────────────────────────────────────────
# ONNX detector wrapper (same as track_rtdetr.py, kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────
class RTDetrOnnx:
    INPUT_SIZE = (640, 640)

    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        active = self.session.get_providers()
        print(f"[INFO] ORT providers active: {active}")
        if "cuda" in device and "CUDAExecutionProvider" not in active:
            print("[WARN] Requested CUDA but ORT fell back to CPU.")
        self.input_name = self.session.get_inputs()[0].name
        self.size_name = self.session.get_inputs()[1].name

    def predict(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4, person_class: int = 0):
        orig_h, orig_w = frame_bgr.shape[:2]
        h, w = self.INPUT_SIZE

        img = cv2.resize(frame_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)

        labels, boxes, scores = self.session.run(
            None, {self.input_name: img, self.size_name: orig_size}
        )
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]

        mask = (labels == person_class) & (scores >= conf_thresh)
        if mask.sum() == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack(
            [boxes[mask], scores[mask], labels[mask].astype(float)]
        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Containment suppression
# ─────────────────────────────────────────────────────────────────────────────
def suppress_contained(dets: np.ndarray, thresh: float) -> np.ndarray:
    """Drop a detection that sits (almost) entirely inside a larger one.

    One person changing a coat, or half-hidden behind furniture, makes the
    detector fire twice: once on the whole body and once on the part that still
    looks like a person. The two boxes carry different embeddings, so the
    gallery happily enrols the inner one as a new person. Plain NMS misses this
    -- IoU between a box and a box nested inside it can be low -- so the test is
    intersection over the *smaller* box's own area.
    """
    if len(dets) < 2 or thresh <= 0:
        return dets
    b = dets[:, :4]
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    order = np.argsort(-area)                      # keep the larger box
    keep = np.ones(len(dets), bool)
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1:]:
            if not keep[j] or area[j] <= 0:
                continue
            iw = max(0.0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            ih = max(0.0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            if iw * ih / area[j] >= thresh:
                keep[j] = False
    return dets[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Color palette (deterministic per person_id)
# ─────────────────────────────────────────────────────────────────────────────
def color_for(pid: int):
    rng = np.random.default_rng(pid * 9_973 + 17)
    c = rng.integers(64, 255, size=3, dtype=np.int64)
    return int(c[0]), int(c[1]), int(c[2])


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    device_str = args.device
    torch_device = torch.device(device_str)

    # ── 1. Detector ──────────────────────────────────────────────────────────
    pid_frames: dict = {}
    n_suppressed = 0

    print(f"[INFO] Loading ONNX detector: {args.onnx}")
    detector = RTDetrOnnx(args.onnx, device=device_str)

    # ── 2. ReID ──────────────────────────────────────────────────────────────
    reid_path = Path(args.reid)
    if not reid_path.is_absolute():
        reid_path = PROJECT_ROOT / args.reid
    print(f"[INFO] Loading ReID model: {reid_path}")
    reid_backend = ReID(weights=reid_path, device=torch_device, half=False).get_backend()

    # ── 3. Video I/O ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video {orig_w}x{orig_h} @ {fps:.1f} fps  ({total} frames)")

    # ── 4. Tracker (short-term) ──────────────────────────────────────────────
    tracker = BotSort(
        reid_model=reid_backend,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        track_buffer=600,
        match_thresh=0.8,
        proximity_thresh=0.9,
        appearance_thresh=0.4,
        cmc_method="ecc",
        frame_rate=int(round(fps)),
        with_reid=True,
    )

    # ── 5. Person DB + resolver (long-term) ─────────────────────────────────
    gallery_path = Path(args.gallery)
    if not gallery_path.is_absolute():
        gallery_path = PROJECT_ROOT / args.gallery
    db = PersonDB(
        gallery_path,
        k_per_person=args.k_per_person,
        match_threshold=args.match_threshold,
        face_match_threshold=args.face_tau,
    )
    resolver = IdentityResolver(
        db,
        decision_frames=args.decision_frames,
        quality_min_conf=args.quality_min_conf,
        ratio_threshold=args.ratio_threshold,
        gallery_update_max_dist=args.gallery_update_max_dist,
    )
    print(
        f"[INFO] DB loaded with {db.n_persons} persons "
        f"({db.n_embeddings} embeddings); next_id={db.next_id}"
    )

    # ── 5a. Face anchor (clothing-independent identity cue, optional) ───────
    face_anchor = None
    if args.face:
        from reid.face_anchor import FaceAnchor

        face_anchor = FaceAnchor(
            det_score_min=args.face_det_min,
            min_face_px=args.face_min_px,
        )
        print(
            f"[INFO] FaceAnchor enabled: tau={args.face_tau} "
            f"det_min={args.face_det_min} min_px={args.face_min_px} "
            f"every={args.face_every}f min_h={args.face_min_track_h}px"
        )

    # ── 5b. Ghost pool (within-session rebinding of lost tracks) ────────────
    ghost_pool = None
    reach = reach_params = None
    scene_gp = scene_model = reach_cache = None
    reach_auto_tier = None
    if not args.disable_ghost_pool:
        weights = ScoringWeights(
            w_spatial=args.ghost_w_spatial,
            w_embedding=args.ghost_w_embedding,
            w_time=args.ghost_w_time,
            w_height=args.ghost_w_height,
            w_hat=args.ghost_w_hat,
            embedding_veto_max_dist=args.ghost_emb_veto,
        )
        # Geodesic floor prior. Without a commissioned scene the pool keeps
        # its pixel-space Gaussian, which is why this is optional rather than
        # required — but on any camera that has been through
        # scripts/commission_scene.py it is strictly the better prior.
        if args.scene and not args.disable_geo_prior:
            _sdir = Path(args.scene)
            _gp = GroundPlane.load(_sdir / "scene.json")
            _sc = SceneModel.load(_sdir / "scene_depth.npz", _gp)
            _cache = _sdir / "reachability.npz"
            if _cache.exists():
                reach = Reachability.load(_cache)
                print(f"[INFO] Reachability loaded from {_cache}")
            else:
                reach = Reachability.build(_sc)
                print("[INFO] Reachability built from depth only "
                      "(will sharpen as feet are observed)")
            # A hand-drawn plan overrides the depth-derived tiers. On a big
            # room the monocular depth is not good enough to separate floor
            # from furniture (office_cam1: 152 cm of error on known floor,
            # against a 20 cm threshold), which leaves ~89% of the map UNKNOWN
            # and the geodesic prior with nothing to say. Two minutes of
            # drawing replaces that with a real boundary.
            _fp = Path(args.floor_plan) if args.floor_plan else _sdir / "floor_plan.json"
            if args.floor_plan or _fp.exists():
                if not _fp.exists():
                    sys.exit(f"--floor-plan {_fp} does not exist")
                # Stamp a working copy. The auto tier is what we write back
                # at the end -- otherwise deleting the JSON would leave the
                # drawing baked into reachability.npz forever.
                reach_auto_tier = reach.tier.copy()
                _plan = FloorPlan.load(_fp)
                _plan.apply_to(reach)
                print(f"[INFO] floor plan applied from {_fp}")
                for i, ht in enumerate(_plan.obstacle_heights(_sc)):
                    if ht["cls"] == "unseen":
                        print(f"[INFO]   blocked polygon {i}: depth never saw this cell "
                              f"({ht['n_cells']} cells) -- height unknown")
                    else:
                        print(f"[INFO]   blocked polygon {i}: {ht['cls']}  "
                              f"(p50={ht['height_p50_m']:.2f}m p90={ht['height_p90_m']:.2f}m "
                              f"max={ht['height_max_m']:.2f}m, {ht['n_seen']}/{ht['n_cells']} cells seen)")
            reach_params = ReachParams(v_max=args.ghost_vmax,
                                       slack_m=args.ghost_reach_slack)
            print(f"[INFO] {reach.summary()}  v_max={args.ghost_vmax} m/s")
            scene_gp, scene_model, reach_cache = _gp, _sc, _cache
        ghost_pool = GhostPool(
            ttl_frames=args.ghost_ttl_frames,
            threshold=args.ghost_threshold,
            weights=weights,
            reach=reach,
            fps=fps,
            reach_params=reach_params,
        )
        print(
            f"[INFO] GhostPool enabled: ttl={args.ghost_ttl_frames}f "
            f"thr={args.ghost_threshold} emb_veto={args.ghost_emb_veto} "
            f"weights(spatial={weights.w_spatial} emb={weights.w_embedding} "
            f"time={weights.w_time} height={weights.w_height} hat={weights.w_hat})"
        )
    # Per-trk_id rolling state, used to populate GhostTracks when a track dies
    track_state: dict[int, dict] = {}
    prev_active_trk_ids: set[int] = set()

    # ── 6. Output writer ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    # ── 7. Main loop ────────────────────────────────────────────────────────
    print("[INFO] Tracking with persistent person DB …")
    frame_idx = 0
    frame_shape = (orig_h, orig_w, 3)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = detector.predict(
                frame, conf_thresh=args.conf, person_class=args.person_class
            )
            n_raw = len(dets)
            dets = suppress_contained(dets, args.contain_thresh)
            n_suppressed += n_raw - len(dets)

            # Pre-extract embeddings ONCE (BoT-SORT will use these via embs=)
            if len(dets) > 0:
                embs = reid_backend.get_features(dets[:, :4], frame)
                embs = np.asarray(embs, dtype=np.float32)
            else:
                embs = np.empty((0, 0), dtype=np.float32)

            tracks = tracker.update(dets, frame, embs=embs if len(dets) else None)
            # tracks: (M, 8) → x1 y1 x2 y2 trk_id conf cls det_ind

            active_trk_ids = {int(t[4]) for t in tracks}

            # Metric floor position of every live track, computed once per
            # frame and shared by the ghost pool and the rolling track state.
            # Feet occluded by furniture fall back to a head-based estimate
            # (see scene_depth.localize), and only the feet-*visible* ones are
            # trusted to teach the walkable map where the floor really is.
            trk_xy: dict[int, tuple[float, float]] = {}
            if reach is not None and len(tracks):
                _xy, _vis = floor_from_boxes(scene_gp, scene_model, tracks[:, :4])
                for _k, _t in enumerate(tracks):
                    if np.all(np.isfinite(_xy[_k])):
                        trk_xy[int(_t[4])] = (float(_xy[_k][0]), float(_xy[_k][1]))
                _good = _vis & np.all(np.isfinite(_xy), axis=1)
                if _good.any():
                    reach.observe(_xy[_good])

            # ── GHOST POOL: harvest just-died, rebind newly-born ────────────
            # A "newly-born" tracker trk_id is checked against recently-lost
            # tracks (the ghost pool). If a confident multi-signal match
            # exists, we transfer that ghost's resolved identity (if any)
            # to the new trk_id so the PPE / display state stays continuous.
            if ghost_pool is not None:
                just_died = prev_active_trk_ids - active_trk_ids
                newly_born = active_trk_ids - prev_active_trk_ids

                for dead_id in just_died:
                    st = track_state.pop(dead_id, None)
                    if st is None or st["n_obs"] < 2 or not st.get("bank"):
                        continue
                    # Push the track's K-best historical embeddings, not just
                    # the most-recent (which is often a back-view at the
                    # moment of disappearance).
                    bank_embs = [emb for emb, _q in st["bank"]]
                    ghost_pool.add(GhostTrack(
                        trk_id=dead_id,
                        last_frame=frame_idx - 1,
                        last_bbox=st["last_bbox"],
                        last_velocity=st["last_velocity"],
                        embeddings=bank_embs,
                        avg_height=st["height_sum"] / st["n_obs"],
                        avg_width=st["width_sum"] / st["n_obs"],
                        last_xy=st.get("last_xy"),
                        last_vel_xy=st.get("last_vel_xy"),
                    ))

                bound_ghost_ids: set[int] = set()
                for t in tracks:
                    trk_id = int(t[4])
                    if trk_id not in newly_born:
                        continue

                    # Same-id "rebind" — BoT-SORT briefly lost then recovered
                    # the same trk_id. Resolver state for trk_id was never
                    # cleared, so nothing to transfer. Just remove the stale
                    # ghost from the pool and move on without going through
                    # scoring (which would be a no-op anyway).
                    if trk_id in ghost_pool.ghosts:
                        ghost_pool.remove(trk_id)
                        continue

                    # Cross-id rebind — this is the dangerous case. Risk of
                    # cascading-error if the rebind is wrong. Goes through
                    # full multi-signal scoring + tighter embedding veto.
                    det_ind = int(t[7])
                    if det_ind < 0 or det_ind >= len(embs):
                        continue
                    x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                    emb_v = np.asarray(embs[det_ind], dtype=np.float32).reshape(-1)
                    emb_v = emb_v / (np.linalg.norm(emb_v) + 1e-12)
                    info = NewTrackInfo(
                        bbox=(x1, y1, x2, y2), embedding=emb_v,
                        height=float(y2 - y1), width=float(x2 - x1),
                        xy=trk_xy.get(trk_id),
                    )
                    match = ghost_pool.best_match(
                        info, frame_idx, exclude_trk_ids=bound_ghost_ids
                    )
                    if match is None:
                        continue
                    ghost_id, score, br = match
                    # Belt-and-suspenders: stricter veto on cross-id rebinds
                    # to prevent embedding-uncertain matches from cascading
                    # an identity to the wrong physical person.
                    if br["emb_dist"] > args.ghost_cross_id_emb_max:
                        print(
                            f"  [GHOST] frame={frame_idx} new trk={trk_id} → "
                            f"REFUSED cross-id rebind to lost trk={ghost_id}  "
                            f"(emb_d={br['emb_dist']:.3f} > "
                            f"{args.ghost_cross_id_emb_max:.2f})"
                        )
                        continue

                    bound_ghost_ids.add(ghost_id)
                    # Transfer resolved identity from the ghost trk_id to
                    # this freshly-born trk_id, so resolver continues with
                    # the same Person_XXX assignment.
                    if ghost_id in resolver.trk_to_person:
                        resolver.trk_to_person[trk_id] = \
                            resolver.trk_to_person.pop(ghost_id)
                        if ghost_id in resolver.last_update_frame:
                            resolver.last_update_frame[trk_id] = \
                                resolver.last_update_frame.pop(ghost_id)
                        if ghost_id in resolver.last_match_dist:
                            resolver.last_match_dist[trk_id] = \
                                resolver.last_match_dist.pop(ghost_id)
                    print(
                        f"  [GHOST] frame={frame_idx} new trk={trk_id} → "
                        f"rebind lost trk={ghost_id}  "
                        f"(score={score:.3f}, emb_d={br['emb_dist']:.3f})"
                    )

                if frame_idx % 30 == 0:
                    ghost_pool.cleanup(frame_idx)

            # Mutual-exclusion set: which Person_XXX are already attached to
            # *other* still-active tracks on this frame? A pending tracklet
            # cannot resolve to one of these — that would put two boxes with
            # the same label on screen at once.
            in_use_pids: set[int] = {
                pid for tid, pid in resolver.trk_to_person.items()
                if tid in active_trk_ids
            }

            # Face crops must come from the un-annotated frame — boxes/labels
            # are drawn onto `frame` per track inside this same loop.
            clean_frame = frame.copy() if face_anchor is not None else frame

            for t in tracks:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                trk_id = int(t[4])
                conf = float(t[5])
                det_ind = int(t[7])

                if det_ind < 0 or det_ind >= len(embs):
                    pid = resolver.trk_to_person.get(trk_id)
                else:
                    # Opportunistic face extraction, throttled: every N-th
                    # frame (offset by trk_id so co-present tracks don't all
                    # pay the cost on the same frame), tall-enough boxes only.
                    face_obs = None
                    if (
                        face_anchor is not None
                        and (frame_idx + trk_id) % args.face_every == 0
                        and (y2 - y1) >= args.face_min_track_h
                    ):
                        cx1, cy1 = max(0, x1), max(0, y1)
                        crop = clean_frame[cy1:max(cy1, y2), cx1:max(cx1, x2)]
                        face_obs = face_anchor.extract(crop)

                    # Exclude pids in use by *other* trks; this trk's own pid
                    # (if already resolved) must remain in scope so the
                    # resolved-track gallery-update path still works.
                    own_pid = resolver.trk_to_person.get(trk_id)
                    excl = in_use_pids - ({own_pid} if own_pid is not None else set())
                    pid = resolver.resolve(
                        trk_id,
                        embs[det_ind],
                        (x1, y1, x2, y2),
                        conf,
                        frame_idx,
                        frame_shape,
                        exclude_pids=excl,
                        face_obs=face_obs,
                    )

                # If this call just committed a new mapping, lock that pid
                # for the rest of this frame's iteration.
                if pid is not None:
                    in_use_pids.add(pid)

                if pid is not None:
                    pid_frames.setdefault(frame_idx, []).append(
                        [int(pid), int(x1), int(y1), int(x2), int(y2)]
                    )
                    color = color_for(pid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    # Big ID, sized with the box so distant people don't get a
                    # label that covers their neighbours.
                    label = str(pid)
                    fs = float(
                        min(max((y2 - y1) / 190.0, 1.0), 3.2)
                    ) * args.id_scale
                    thk = max(2, int(round(fs * 2.6)))
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, fs, thk
                    )
                    tx = int(min(max((x1 + x2) // 2 - tw // 2, 2), orig_w - tw - 2))
                    ty = y1 - 10 if y1 - 10 - th > 0 else min(y2 + th + 8, orig_h - 4)
                    cv2.rectangle(
                        frame, (tx - 6, ty - th - 8), (tx + tw + 6, ty + 8),
                        (15, 15, 15), -1,
                    )
                    cv2.putText(
                        frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), thk + 4, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        color, thk, cv2.LINE_AA,
                    )
                else:
                    # Unresolved track: gray bbox only, no label.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

            # HUD: gallery summary
            hud = (
                f"frame {frame_idx}/{total}  "
                f"persons={db.n_persons}  "
                f"enrolled={resolver.n_enrolled}  matched={resolver.n_matched}  "
                f"refused_amb={resolver.n_refused_ambiguous}"
            )
            if face_anchor is not None:
                hud += (
                    f"  face_anchor={resolver.n_face_anchored}"
                    f"  face_reanchor={resolver.n_face_reanchored}"
                )
            cv2.rectangle(frame, (0, 0), (orig_w, 28), (0, 0, 0), -1)
            cv2.putText(
                frame, hud, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # ── GHOST POOL: update per-track rolling state for next frame ──
            # Per-track K-best embedding bank:
            #   - "bank" is a list of (embedding, quality_score) tuples, K-best
            #     by quality. quality_score = bbox_area * confidence.
            #   - Before adding a new embedding, we check it against the bank's
            #     running centroid; if the new embedding is a large view change
            #     (cosine distance > BANK_CONSISTENCY_MAX), we skip it. This
            #     stops back-view crops from polluting the bank when the track
            #     was mostly observed front-on (and vice versa).
            #   - Until BANK_WARMUP_FRAMES we add unconditionally so the bank
            #     can ever populate.
            if ghost_pool is not None:
                BANK_K = 10
                BANK_WARMUP_FRAMES = 3
                BANK_CONSISTENCY_MAX = 0.30   # cosine distance gate

                def _bank_add(bank: list, emb: np.ndarray, q: float, k: int) -> None:
                    if len(bank) < k:
                        bank.append((emb, q))
                        return
                    worst_idx = min(range(len(bank)), key=lambda i: bank[i][1])
                    if q > bank[worst_idx][1]:
                        bank[worst_idx] = (emb, q)

                def _bank_centroid(bank: list):
                    if not bank:
                        return None
                    c = np.mean([e for e, _ in bank], axis=0)
                    n = np.linalg.norm(c)
                    return c / n if n > 1e-12 else None

                for t in tracks:
                    trk_id = int(t[4])
                    det_ind = int(t[7])
                    x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                    bbox_t = (x1, y1, x2, y2)

                    st = track_state.get(trk_id)
                    if st is None:
                        st = {
                            "last_bbox": bbox_t,
                            "last_velocity": (0.0, 0.0),
                            "bank": [],                # K-best (emb, quality)
                            "n_obs": 1,
                            "height_sum": float(y2 - y1),
                            "width_sum": float(x2 - x1),
                            "last_xy": trk_xy.get(trk_id),
                            "last_vel_xy": (0.0, 0.0),
                        }
                        track_state[trk_id] = st
                    else:
                        prev_cx = 0.5 * (st["last_bbox"][0] + st["last_bbox"][2])
                        prev_cy = 0.5 * (st["last_bbox"][1] + st["last_bbox"][3])
                        new_cx = 0.5 * (x1 + x2)
                        new_cy = 0.5 * (y1 + y2)
                        v_inst = (new_cx - prev_cx, new_cy - prev_cy)
                        a = 0.3
                        st["last_velocity"] = (
                            a * v_inst[0] + (1 - a) * st["last_velocity"][0],
                            a * v_inst[1] + (1 - a) * st["last_velocity"][1],
                        )
                        # Metric velocity in m/s, smoothed the same way as
                        # the pixel one. This is what makes "kept walking that
                        # way" mean something physical rather than something
                        # that depends on distance from the lens.
                        xy_now = trk_xy.get(trk_id)
                        if xy_now is not None:
                            xy_prev = st.get("last_xy")
                            if xy_prev is not None:
                                vx = (xy_now[0] - xy_prev[0]) * fps
                                vy = (xy_now[1] - xy_prev[1]) * fps
                                pv = st.get("last_vel_xy") or (0.0, 0.0)
                                st["last_vel_xy"] = (a * vx + (1 - a) * pv[0],
                                                     a * vy + (1 - a) * pv[1])
                            st["last_xy"] = xy_now
                        st["last_bbox"] = bbox_t
                        st["n_obs"] += 1
                        st["height_sum"] += float(y2 - y1)
                        st["width_sum"] += float(x2 - x1)

                    if 0 <= det_ind < len(embs):
                        e = np.asarray(embs[det_ind], dtype=np.float32).reshape(-1)
                        e = e / (np.linalg.norm(e) + 1e-12)
                        bbox_area = float((x2 - x1) * (y2 - y1))
                        det_conf = float(t[5])
                        q = bbox_area * det_conf

                        bank = st["bank"]
                        # Consistency gate (skip until warmup so bank can fill)
                        if len(bank) >= BANK_WARMUP_FRAMES:
                            c = _bank_centroid(bank)
                            if c is not None:
                                d = 1.0 - float(np.dot(c, e))
                                if d > BANK_CONSISTENCY_MAX:
                                    # likely view change — skip without poisoning
                                    continue
                        _bank_add(bank, e, q, BANK_K)

                prev_active_trk_ids = active_trk_ids

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = frame_idx / total * 100 if total else 0
                ghost_stat = (
                    f" ghost={len(ghost_pool)} rebound={ghost_pool.n_rebound}"
                    f" reachveto={ghost_pool.n_reach_veto}"
                    if ghost_pool is not None else ""
                )
                print(
                    f"  frame {frame_idx}/{total} ({pct:.1f}%)  "
                    f"persons={db.n_persons} "
                    f"(enrolled={resolver.n_enrolled} matched={resolver.n_matched})"
                    f"{ghost_stat}"
                )
    finally:
        cap.release()
        writer.release()
        db.save()
        # Persist what this run learned about where the floor actually is, so
        # the next run on this camera starts with a sharper map. This is the
        # whole no-annotation story: the site commissions itself by being
        # walked through.
        if reach is not None and reach_cache is not None:
            if reach_auto_tier is not None:
                from reid.reachability import FREE, MIN_FOOT_OBS
                _saved = reach_auto_tier.copy()
                _saved[reach.foot_obs >= MIN_FOOT_OBS] = FREE
                reach.tier = _saved
                reach._fields.clear()
                reach._region = None
            reach.save(reach_cache)
            print(f"[DONE] {reach.summary()} -> {reach_cache}")

    if args.dump_json:
        import json as _json
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_json).write_text(_json.dumps(
            {"W": orig_w, "H": orig_h, "frames": frame_idx,
             "pid_frames": {str(k): v for k, v in sorted(pid_frames.items())}}))
        print(f"[DONE] Saved per-frame person IDs: {args.dump_json}")

    print(f"[DONE] Suppressed {n_suppressed} contained detections "
          f"(thresh={args.contain_thresh})")

    print(f"\n[DONE] Saved video: {args.output}")
    print(
        f"[DONE] DB final state: {db.n_persons} persons, "
        f"{db.n_embeddings} embeddings  ({db.db_path})"
    )

    fixed = args.output.replace(".mp4", "_h264.mp4")
    os.system(
        f'ffmpeg -y -i "{args.output}" -vcodec libx264 -crf 23 "{fixed}" '
        f'-loglevel quiet'
    )
    if os.path.exists(fixed):
        print(f"[DONE] H.264 copy: {fixed}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RT-DETRv4 + BoT-SORT + persistent person DB"
    )
    p.add_argument("--onnx", default="models/20260504_rtv4_hgnetv2_m.onnx")
    # CLIP-ReID is the production embedding: it built the 1280-dim gallery and
    # benchmarks clearly ahead of OSNet on this footage (AUC 0.86 vs 0.77,
    # separating co-present workers much better — see bench/REPORT.md). The old
    # osnet_x1_0 default also pointed at a repo-root file that no longer exists.
    p.add_argument("--reid", default="models/clip_market1501.pt")
    p.add_argument("--input", default="videos/gunsan_test.mp4")
    p.add_argument("--output", default="runs/track/rtdetr_db.mp4")
    p.add_argument("--gallery", default="gallery/persons.npz")
    p.add_argument("--contain-thresh", type=float, default=0.9,
                   help="drop a detection this fraction contained inside a "
                        "larger one (0 disables)")
    p.add_argument("--id-scale", type=float, default=1.0,
                   help="multiplier on the on-screen person-ID text size")
    p.add_argument("--dump-json", default=None,
                   help="write per-frame resolved person IDs + boxes here")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--person-class", type=int, default=0)
    # DB / resolver knobs
    p.add_argument("--k-per-person", type=int, default=10)
    p.add_argument(
        "--match-threshold", type=float, default=0.35,
        help="cosine distance ceiling for declaring a match (lower = stricter). "
             "0.25 was far too strict for CLIP on this footage — same-person "
             "views median ~0.36 apart, so ~75%% of returning workers spawned a "
             "new ID. 0.35 roughly halves that; the Lowe ratio test below keeps "
             "merges controlled. Sweep 0.30-0.42 per site — see bench/REPORT.md.",
    )
    p.add_argument(
        "--ratio-threshold", type=float, default=0.85,
        help="Lowe-style ratio: best/second-best must be < this to commit a match",
    )
    p.add_argument(
        "--gallery-update-max-dist", type=float, default=0.20,
        help="per-frame gallery updates are skipped if cur_dist > this",
    )
    p.add_argument("--decision-frames", type=int, default=8)
    p.add_argument("--quality-min-conf", type=float, default=0.7)
    # Ghost pool (within-session track-rebinding) knobs
    p.add_argument(
        "--scene", type=str, default=None,
        help="commissioned calib dir (scene.json + scene_depth.npz) enabling "
             "the geodesic walkable-floor prior for ghost rebinding",
    )
    p.add_argument(
        "--floor-plan", type=str, default=None,
        help="hand-drawn walkable/obstacle polygons from "
             "scripts/annotate_floor.py; overrides the depth-derived walkable "
             "map. Defaults to <scene>/floor_plan.json when that file exists",
    )
    p.add_argument(
        "--disable-geo-prior", action="store_true",
        help="keep the legacy pixel-space Gaussian spatial prior even when "
             "--scene is given (for A/B)",
    )
    p.add_argument(
        "--ghost-vmax", type=float, default=2.2,
        help="m/s walking-speed bound for the reachability veto",
    )
    p.add_argument(
        "--ghost-reach-slack", type=float, default=0.6,
        help="metres of localisation slack added to the reachability budget",
    )
    p.add_argument(
        "--ghost-ttl-frames", type=int, default=9000,
        help="how many frames a lost track stays rebind-eligible (default ~10min@15fps)",
    )
    p.add_argument(
        "--ghost-threshold", type=float, default=0.55,
        help="combined-signal score required to rebind a new track to a ghost",
    )
    p.add_argument(
        "--ghost-emb-veto", type=float, default=0.55,
        help="cosine distance above which embedding mismatch vetoes a rebind",
    )
    p.add_argument(
        "--ghost-cross-id-emb-max", type=float, default=0.30,
        help="cross-id rebinds (different trk_id) require emb_dist below this "
             "tighter ceiling — prevents borderline matches from cascading the "
             "wrong identity onto a different physical person",
    )
    p.add_argument(
        "--ghost-w-spatial", type=float, default=0.30,
        help="ghost-pool score weight: spatial-prior (predicted position match)",
    )
    p.add_argument(
        "--ghost-w-embedding", type=float, default=0.45,
        help="ghost-pool score weight: body-embedding similarity",
    )
    p.add_argument(
        "--ghost-w-time", type=float, default=0.10,
        help="ghost-pool score weight: time-decay (younger ghosts preferred)",
    )
    p.add_argument(
        "--ghost-w-height", type=float, default=0.10,
        help="ghost-pool score weight: bbox-height consistency",
    )
    p.add_argument(
        "--ghost-w-hat", type=float, default=0.05,
        help="ghost-pool score weight: hat-color match (auxiliary)",
    )
    p.add_argument(
        "--disable-ghost-pool", action="store_true",
        help="turn off ghost-pool rebinding (baseline comparison)",
    )
    # Face anchoring (clothing-independent identity cue) knobs
    p.add_argument(
        "--face", action="store_true",
        help="enable pose-gated face anchoring (confirm-only: a strict face "
             "match decides/corrects identity; a face mismatch never blocks a "
             "body match — see bench/05_eval_face_anchor.py)",
    )
    p.add_argument(
        "--face-tau", type=float, default=0.50,
        help="ArcFace cosine-distance ceiling for a face match. Kept well "
             "below the face EER point (~0.70) because face matches override "
             "body appearance and must be merge-safe",
    )
    p.add_argument(
        "--face-det-min", type=float, default=0.5,
        help="min SCRFD detection confidence for the pose gate",
    )
    p.add_argument(
        "--face-min-px", type=float, default=24.0,
        help="min face min-side in pixels for the pose gate",
    )
    p.add_argument(
        "--face-every", type=int, default=3,
        help="run face extraction on a track every N-th frame (cost throttle)",
    )
    p.add_argument(
        "--face-min-track-h", type=int, default=100,
        help="skip face extraction for person boxes shorter than this (px)",
    )
    args = p.parse_args()
    run(args)
