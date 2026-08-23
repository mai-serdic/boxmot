"""Offline whole-path stitch over a tracklet dump.

The live tracker (BoT-SORT + ghost pool) still writes greedy IDs. This post-pass
rewrites them: tracklets become nodes, min-cost flow is one person per path.
Needs the future, so it runs on a recording, not frame by frame.

    python scripts/stitch_traj.py \
        --scene calib/office_cam1 \
        --traj  runs/traj/office_cam1_block2.json \
        --fps 10 \
        --out   runs/traj/office_cam1_block2_stitched.json

`--floor-plan` defaults to <scene>/floor_plan.json when that file exists.
`--compare-floor-plan` runs with and without the drawing so you can see whether
it changed the assignment, not just the printed map.
`--birth` is optional; omit it and the knob is taken from the observed
link-probability distribution (see `reid.trajectory_stitch.suggest_birth_cost`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid.crossing_events import (
    candidate_split_frames,
    detect_crossing_events,
    unresolved_crossings,
)
from reid.floor_plan import FloorPlan
from reid.joint_crossing import apply_crossing_resolutions, named_path_identities
from reid.occupancy_partition import bipartite_room_partition
from reid.reachability import Reachability, ReachParams
from reid.room_memory import RoomMemory, groups_from_memory, stitch_memory_groups
from reid.scene_depth import SceneModel, floor_from_boxes
from reid.scene_geometry import GroundPlane
from reid.trajectory_stitch import (
    Tracklet,
    number_unassigned_tracklets,
    path_frames_from_traj,
    run_stitch,
)


def dump_to_traj(d: dict, allow_gallery: bool = False) -> dict[int, list]:
    """BoT-SORT tracklets from `traj`. Gallery pid_frames is not identity."""
    if "traj" in d:
        return {int(k): [list(r) for r in v] for k, v in d["traj"].items()}
    if not allow_gallery:
        raise SystemExit(
            "dump has no 'traj' — whole-path identity needs BoT-SORT tracklets, "
            "not gallery pid_frames. Run track_rtdetr_db.py --dump-json, or pass "
            "--allow-gallery-fragments only for debugging."
        )
    import warnings

    warnings.warn(
        "stitching gallery fragments from pid_frames — not a spatial tracklet graph",
        stacklevel=2,
    )
    traj: dict[int, list] = {}
    for f, rows in d["pid_frames"].items():
        fi = int(f)
        for row in rows:
            tid = int(row[0])
            traj.setdefault(tid, []).append([fi, *row[1:5]])
    for rows in traj.values():
        rows.sort(key=lambda r: r[0])
    return traj


def build_tracklets(
    traj: dict, gp, scene, min_obs: int = 10, embs: dict | None = None
) -> list[Tracklet]:
    out = []
    embs = embs or {}
    for tid, rows in traj.items():
        a = np.asarray(rows, float)
        if len(a) < min_obs:
            continue
        xy, _vis = floor_from_boxes(gp, scene, a[:, 1:5])
        emb = embs.get(tid) or embs.get(str(tid))
        if emb is not None:
            e = np.asarray(emb, dtype=np.float32).reshape(-1)
            e = e / (np.linalg.norm(e) + 1e-12)
        else:
            e = None
        out.append(Tracklet(int(tid), a[:, 0].astype(int), xy, emb=e))
    out.sort(key=lambda t: t.t0)
    return out


def build_reach(scene, tracklets, plan: FloorPlan | None = None) -> Reachability:
    reach = Reachability.build(scene)
    for t in tracklets:
        ok = np.all(np.isfinite(t.xy), axis=1)
        if ok.any():
            reach.observe(t.xy[ok])
    if plan is not None:
        plan.apply_to(reach)
    return reach


def fmt_groups(groups: dict) -> str:
    parts = []
    for tids in groups.values():
        parts.append("{" + ",".join(str(t) for t in tids) + "}")
    return "  ".join(parts)


def report(tag: str, r: dict) -> None:
    print(f"\n=== {tag} ===")
    print(
        f"  raw tracklets     {r['n_raw']}   (max simultaneous {r['max_simultaneous']})"
    )
    print(f"  greedy            {r['n_greedy']}  {fmt_groups(r['groups_greedy'])}")
    print(f"  global stitch     {r['n_path']}  {fmt_groups(r['groups_path'])}")
    print(f"  birth_cost        {r['birth_cost']:.3f}")
    s = r.get("suggest") or {}
    if s:
        print(
            f"  link dist         feasible={s['n_feasible']}  "
            f"with_pred={s['n_with_pred']}  "
            f"best_nats p50={s['best_nats_p50']} p85={s['best_nats_p85']}"
        )


def result_json(r: dict, meta: dict, traj: dict) -> dict:
    pf = path_frames_from_traj(traj, r["owner_path"], unassigned_id=0)
    return {
        **meta,
        "identity": "path_through_space",
        "birth_cost": r["birth_cost"],
        "suggest": r["suggest"],
        "max_simultaneous": r["max_simultaneous"],
        "n_raw": r["n_raw"],
        "n_greedy": r["n_greedy"],
        "n_path": r["n_path"],
        "groups_path": {str(k): v for k, v in r["groups_path"].items()},
        "owner_path": {str(k): int(v) for k, v in r["owner_path"].items()},
        "owner_trk": {str(k): int(v) for k, v in r["owner"].items()},
        "path_frames": {str(k): v for k, v in sorted(pf.items())},
        "unassigned_tracklets": sorted(
            int(tid) for tid in traj if int(tid) not in r["owner_path"]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="default: dump['fps'] if present, else 15",
    )
    ap.add_argument("--floor-plan", default=None)
    ap.add_argument(
        "--no-floor-plan",
        action="store_true",
        help="ignore <scene>/floor_plan.json even if it exists",
    )
    ap.add_argument(
        "--compare-floor-plan",
        action="store_true",
        help="run twice, with and without the drawing",
    )
    ap.add_argument(
        "--birth",
        type=float,
        default=None,
        help="omit to derive from occupancy and p_floor hops",
    )
    ap.add_argument(
        "--w-emb",
        type=float,
        default=0.0,
        help="optional appearance soft weight on link costs (default 0 = geometry "
        "only). Cannot veto a feasible walk.",
    )
    ap.add_argument(
        "--allow-gallery-fragments",
        action="store_true",
        help="allow pid_frames-only dumps (debug; not spatial tracklets)",
    )
    ap.add_argument(
        "--min-obs",
        type=int,
        default=10,
        help="minimum observations admitted to core path optimization; shorter fragments are numbered afterward",
    )
    ap.add_argument(
        "--room-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use stateful room-memory + global revision",
    )
    ap.add_argument("--room-memory-min-link", type=float, default=0.18)
    ap.add_argument("--room-memory-batch-frames", type=int, default=20)
    ap.add_argument("--room-memory-global-birth", type=float, default=6.0)
    ap.add_argument(
        "--occupancy-partition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="collapse fragments only when sustained co-presence proves a bipartite room",
    )
    ap.add_argument(
        "--crossing-resolutions",
        default=None,
        help="authoritative reviewed crossing-state JSON; ambiguous events stay unresolved",
    )
    args = ap.parse_args()

    sdir = Path(args.scene)
    dump = json.loads(Path(args.traj).read_text())
    fps = args.fps or dump.get("fps") or 15.0
    gp = GroundPlane.load(sdir / "scene.json")
    scene = SceneModel.load(sdir / "scene_depth.npz", gp)
    traj = dump_to_traj(dump, allow_gallery=args.allow_gallery_fragments)
    crossing_events = detect_crossing_events(traj)
    embs = dump.get("trk_embs") or {}
    tracklets = build_tracklets(traj, gp, scene, min_obs=args.min_obs, embs=embs)
    print(
        f"[INFO] {len(traj)} tracklets in dump, {len(tracklets)} with "
        f">={args.min_obs} obs, fps={fps}, w_emb={args.w_emb}"
    )

    fp_path = None
    if not args.no_floor_plan:
        fp_path = Path(args.floor_plan) if args.floor_plan else sdir / "floor_plan.json"
        if not fp_path.exists():
            if args.floor_plan:
                sys.exit(f"--floor-plan {fp_path} does not exist")
            fp_path = None
    plan = FloorPlan.load(fp_path) if fp_path else None
    if plan:
        print(
            f"[INFO] floor plan {fp_path}  "
            f"({sum(1 for p in plan.polygons if p['kind'] == 'walkable')} walkable, "
            f"{sum(1 for p in plan.polygons if p['kind'] == 'blocked')} blocked)"
        )

    par = ReachParams()
    reach = build_reach(scene, tracklets, plan)
    print(f"[INFO] {reach.summary()}")
    r = run_stitch(reach, tracklets, fps, par, w_emb=args.w_emb, birth_cost=args.birth)
    memory_owner = None
    memory_verdicts = None
    memory_global_owner = None
    memory_states = None
    if args.room_memory:
        memory = RoomMemory(
            reach,
            fps,
            par,
            min_link_prob=args.room_memory_min_link,
            batch_window_frames=args.room_memory_batch_frames,
        )
        memory_owner = memory.observe(tracklets)
        memory_verdicts = memory.verdicts
        memory_states = memory.snapshot()
        memory_global_owner = stitch_memory_groups(
            reach,
            tracklets,
            memory_owner,
            fps,
            par,
            birth_cost=args.room_memory_global_birth,
        )
        memory_global_owner, short_fragments = number_unassigned_tracklets(
            memory_global_owner, traj
        )
        print(
            f"  room memory     {len(set(memory_owner.values()))}  "
            f"{groups_from_memory(memory_owner)}"
        )
        print(
            f"  room global     {len(set(memory_global_owner.values()))}  "
            f"{groups_from_memory(memory_global_owner)}"
        )
    report("with floor plan" if plan else "depth map only", r)

    r_nplan = None
    if args.compare_floor_plan:
        if plan is None:
            sys.exit("--compare-floor-plan needs a floor_plan.json")
        reach_auto = build_reach(scene, tracklets, plan=None)
        print(f"[INFO] auto map {reach_auto.summary()}")
        r_nplan = run_stitch(
            reach_auto, tracklets, fps, par, w_emb=args.w_emb, birth_cost=args.birth
        )
        report("without floor plan", r_nplan)
        same = r["groups_path"] == r_nplan["groups_path"]
        print("\n  assignment changed:", "no" if same else "YES")
        if r["n_path"] != r_nplan["n_path"]:
            print(f"  identity count {r_nplan['n_path']} -> {r['n_path']}")
        if same:
            print(
                "  (map still changed: see free/unknown/blocked above. "
                "At this birth_cost exclusivity already forces the split; "
                "the drawing matters more at lower birth_cost and for live "
                "ghost-pool vetoes.)"
            )

    out = (
        Path(args.out)
        if args.out
        else Path(args.traj).with_name(Path(args.traj).stem + "_stitched.json")
    )
    payload = result_json(
        r,
        {
            "W": dump.get("W"),
            "H": dump.get("H"),
            "frames": dump.get("frames"),
            "fps": fps,
            "scene": str(sdir),
            "floor_plan": str(fp_path) if fp_path else None,
            "source": str(args.traj),
            "w_emb": args.w_emb,
        },
        traj,
    )
    payload["path_identity"] = {
        str(pid): {
            "person_id": None,
            "status": "unresolved",
            "confidence": 0.0,
            "evidence": [],
        }
        for pid in sorted(r["groups_path"])
    }
    payload["crossing_events"] = [e.__dict__ for e in crossing_events]
    payload["candidate_split_frames"] = {
        str(k): v for k, v in candidate_split_frames(crossing_events).items()
    }
    payload["uncertain_events"] = unresolved_crossings(crossing_events)
    if memory_owner is not None:
        payload["room_memory_owner"] = {str(k): int(v) for k, v in memory_owner.items()}
        payload["room_memory_groups"] = {
            str(k): v for k, v in groups_from_memory(memory_owner).items()
        }
        payload["room_memory_verdicts"] = memory_verdicts
        payload["path_states"] = {str(k): v for k, v in memory_states.items()}
        payload["room_memory_global_owner"] = {
            str(k): int(v) for k, v in memory_global_owner.items()
        }
        payload["room_memory_global_groups"] = {
            str(k): v for k, v in groups_from_memory(memory_global_owner).items()
        }
        payload["short_fragment_tracklets"] = short_fragments
        room_pf = path_frames_from_traj(traj, memory_global_owner, unassigned_id=0)
        payload["room_path_frames"] = {str(k): v for k, v in sorted(room_pf.items())}
        payload["room_path_identity"] = {
            str(pid): {
                "person_id": None,
                "status": "unresolved",
                "confidence": 0.0,
                "evidence": [],
            }
            for pid in sorted(set(memory_global_owner.values()))
        }
        payload["baseline_owner_path"] = payload["owner_path"]
        payload["baseline_groups_path"] = payload["groups_path"]
        payload["baseline_path_frames"] = payload["path_frames"]
        payload["owner_path"] = payload["room_memory_global_owner"]
        payload["groups_path"] = payload["room_memory_global_groups"]
        payload["path_frames"] = payload["room_path_frames"]
        payload["n_path"] = len(payload["room_memory_global_groups"])
        payload["path_identity"] = payload["room_path_identity"]
        payload["unassigned_tracklets"] = []
    if r_nplan is not None:
        payload["without_floor_plan"] = {
            "n_path": r_nplan["n_path"],
            "groups_path": {str(k): v for k, v in r_nplan["groups_path"].items()},
            "birth_cost": r_nplan["birth_cost"],
        }
    if args.occupancy_partition:
        primary_owner = {int(k): int(v) for k, v in payload["owner_path"].items()}
        partition_owner, partition_diagnostic = bipartite_room_partition(
            traj, primary_owner
        )
        payload["occupancy_partition"] = partition_diagnostic
        if partition_diagnostic["status"] == "applied":
            payload["pre_occupancy_owner_path"] = payload["owner_path"]
            payload["pre_occupancy_groups_path"] = payload["groups_path"]
            payload["pre_occupancy_path_frames"] = payload["path_frames"]
            partition_groups = groups_from_memory(partition_owner)
            partition_pf = path_frames_from_traj(
                traj, partition_owner, compact=False, unassigned_id=0
            )
            payload["owner_path"] = {str(k): int(v) for k, v in partition_owner.items()}
            payload["groups_path"] = {str(k): v for k, v in partition_groups.items()}
            payload["path_frames"] = {
                str(k): v for k, v in sorted(partition_pf.items())
            }
            payload["n_path"] = len(partition_groups)
            payload["path_identity"] = {
                str(path_id): {
                    "person_id": None,
                    "status": "unresolved",
                    "confidence": 0.0,
                    "evidence": [],
                }
                for path_id in sorted(partition_groups)
            }
            print(
                f"[INFO] occupancy partition {partition_diagnostic['baseline_paths']} "
                f"-> {payload['n_path']} paths; baseline consistency "
                f"{partition_diagnostic['baseline_consistency']:.3f}"
            )
    if args.crossing_resolutions:
        resolution_path = Path(args.crossing_resolutions)
        resolution_doc = json.loads(resolution_path.read_text())
        primary_owner = {int(k): int(v) for k, v in payload["owner_path"].items()}
        frame_owner, applied, resolved, bindings = apply_crossing_resolutions(
            crossing_events,
            traj,
            primary_owner,
            resolution_doc,
            include_bindings=True,
        )
        corrected_pf = path_frames_from_traj(
            traj,
            primary_owner,
            compact=False,
            unassigned_id=0,
            frame_owner=frame_owner,
        )
        payload["path_frames"] = {str(k): v for k, v in sorted(corrected_pf.items())}
        if "room_path_frames" in payload:
            payload["room_path_frames"] = payload["path_frames"]
        payload["crossing_resolution_source"] = str(resolution_path)
        payload["crossing_resolutions_applied"] = applied
        payload["reviewed_path_labels"] = {
            label: int(path_id) for label, path_id in bindings.items()
        }
        payload["path_identity"].update(named_path_identities(bindings, resolution_doc))
        payload["uncertain_events"] = [
            item
            for item in payload["uncertain_events"]
            if item["event_id"] not in resolved
        ]
        print(
            f"[INFO] applied {len(applied)} authoritative crossing resolutions "
            f"to {len(frame_owner)} observations; {len(resolved)} events closed"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
