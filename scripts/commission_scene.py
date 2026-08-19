"""Commission a new camera: calibrate it, model its scene, and produce the
evidence a human needs to accept or reject the result.

    python scripts/commission_scene.py --name <site> --input <video-or-frames-dir>

That is the whole procedure. Input is footage - a video file or a directory of
frames - and nothing else. Tracklets are built automatically if they do not
already exist. Nothing is annotated, no checkerboard is walked around the room,
no zones are drawn: a per-site setup ritual is not a product.

Output is `calib/<name>/`:

    scene.json            calibrated ground plane
    scene_depth.npz       metric static scene (per-pixel height above floor)
    stature_field.json    per-cell stature bias correction
    commission.md         printed checks, in writing
    01..09_*.png          the image pack

The image pack exists because a calibration that cannot be *seen* to be right
will eventually be trusted when it is wrong. Sheet 08 is the one that settles
it: a 1 m floor grid and 1.7 m human silhouettes drawn back into the image. If
the squares look square and the silhouettes match the people, the metre scale
is real; if they do not, no summary statistic should talk anybody out of it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid.scene_depth import (  # noqa: E402
    STEP_OVER_M,
    SceneModel,
    StatureField,
    cam_point,
    localize,
)
from reid.scene_geometry import (  # noqa: E402
    GroundPlane,
    RadialModel,
    background_frame,
    collect_observations,
    estimate_radial_k1,
    dwell_weights,
    fit_height_field,
    focal_from_stature_consistency,
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'WARN'}] {name:<28} {detail}")
    return ok


def metric_depth(img_bgr: np.ndarray) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = net(**{k: v.to(dev) for k, v in proc(images=rgb, return_tensors="pt").items()})
    d = torch.nn.functional.interpolate(
        out.predicted_depth[None], size=img_bgr.shape[:2],
        mode="bicubic", align_corners=False)[0, 0]
    return d.cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# projection helpers (undistorted image space, where the depth map lives)
# --------------------------------------------------------------------------


def project(gp: GroundPlane, X: np.ndarray) -> np.ndarray:
    """Camera-coords metres -> undistorted pixel coords."""
    X = np.atleast_2d(X)
    s = max(gp.width, gp.height) / 2.0
    z = np.where(np.abs(X[:, 2]) > 1e-9, X[:, 2], np.nan)
    return np.stack([
        gp.f * X[:, 0] / z * s + gp.width / 2.0,
        gp.f * X[:, 1] / z * s + gp.height / 2.0,
    ], axis=1)


def floor_to_cam(gp: GroundPlane, xy: np.ndarray, h_m: float = 0.0) -> np.ndarray:
    return cam_point(gp, xy, h_m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--input", "--video", dest="input", required=True,
                    help="video file OR directory of frames")
    ap.add_argument("--traj", default=None,
                    help="tracklet JSON; built automatically if omitted")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mean-stature", type=float, default=1.70)
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--onnx", default="models/20260504_rtv4_hgnetv2_m.onnx")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--depth-npy", default=None,
                    help="reuse a cached depth map instead of running the model")
    args = ap.parse_args()

    # Fail on a bad path before spending a minute loading models.
    if not Path(args.input).exists():
        raise SystemExit(
            f"--input {args.input} does not exist.\n"
            "Pass a real video file or a directory of frames, e.g.\n"
            "  python scripts/commission_scene.py --name gunsan "
            "--input videos/gunsan_test.mp4")

    out = Path(args.out_dir or f"calib/{args.name}")
    out.mkdir(parents=True, exist_ok=True)

    # Tracklets are the only input, and building them is part of commissioning
    # rather than a step the operator has to know about.
    traj_path = Path(args.traj) if args.traj else Path(f"runs/traj/{args.name}.json")
    if not traj_path.exists():
        print(f"[0/5] tracklets -> {traj_path}")
        from make_tracklets import build
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(json.dumps(
            build(args.input, args.onnx, fps=args.fps)))
    d = json.loads(traj_path.read_text())
    W, H = d["W"], d["H"]
    traj = {int(k): sorted(v) for k, v in d["traj"].items()}
    print(f"\n=== commissioning '{args.name}' :: {W}x{H}, {len(traj)} tracklets ===\n")

    # -- 1. geometry --------------------------------------------------------
    print("[1/5] ground plane")
    bg = background_frame(args.input)
    if args.k1 is not None:
        k1, curve = args.k1, []
    else:
        k1, curve = estimate_radial_k1(bg)
    model = RadialModel(k1, W, H)
    bg_u = model.undistort_image(bg)

    obs = collect_observations(traj, model)
    n_tracks = len(set(obs.track_id.tolist()))
    check("observations", len(obs) >= 200,
          f"{len(obs)} standing full-body obs from {n_tracks} tracklets")

    # The horizon fit is a RANSAC vote, so it is decided by how many
    # observations agree - not by how many *distinct* viewpoints agree. In an
    # office most people sit still, and one person seated for twenty minutes
    # supplies thousands of near-identical samples that win the vote outright
    # and tilt the horizon to fit their chair. Weighting each observation by
    # 1/(samples in its track's floor cell) removes that dwell-time bias while
    # keeping every sample, which matters on cameras that are sample-starved to
    # begin with - discarding duplicates outright fixed the office camera but
    # cost a sparser camera 7 points of downstream ranking accuracy.
    w = dwell_weights(obs)
    print(f"      dwell weighting: {len(obs)} obs -> {w.sum():.0f} effective")

    horizon, hmask, rel_rms = fit_height_field(obs, weights=w)
    check("height-field fit", rel_rms <= 0.06, f"rel-rms {rel_rms:.1%} (want <=6%)")

    f, spread, (f_lo, f_hi), nf = focal_from_stature_consistency(obs, horizon, mask=hmask)
    s = max(W, H) / 2.0
    check("focal length", abs(nf - 1.0) <= 0.05,
          f"f={f*s:.0f}px  fov={2*np.degrees(np.arctan(1.0/f)):.0f}deg  "
          f"near/far bias {nf:.3f} (want 1.00+-0.05)")

    gp = GroundPlane.from_horizon(horizon, f, obs, model,
                                  mean_stature_m=args.mean_stature, mask=hmask)
    gp.save(out / "scene.json")
    cam_h = abs(gp.cam_height_m)
    check("camera height", 1.8 <= cam_h <= 6.0, f"{cam_h:.2f} m")

    st_all = gp.stature_m(obs.foot[hmask], obs.head[hmask])
    ok_st = np.isfinite(st_all)
    p10, p50, p90 = np.percentile(st_all[ok_st], [10, 50, 90])
    check("stature spread", (p90 - p10) < 0.35,
          f"p10/p50/p90 = {p10:.2f}/{p50:.2f}/{p90:.2f} m (want p90-p10 < 0.35)")

    # -- 2. depth -----------------------------------------------------------
    print("\n[2/5] metric depth")
    if args.depth_npy and Path(args.depth_npy).exists():
        depth = np.load(args.depth_npy).astype(np.float64)
        print(f"      reusing {args.depth_npy}")
    else:
        depth = metric_depth(bg_u)
        np.save(out / "depth_raw.npy", depth.astype(np.float32))
    foot_px = obs.foot[hmask][:, :2] / obs.foot[hmask][:, 2:] * s + np.array([W / 2, H / 2])
    scene = SceneModel.build(depth, gp, foot_px)
    scene.save(out / "scene_depth.npz")
    check("depth vs floor", scene.floor_resid_m <= 0.12,
          f"1-sigma {scene.floor_resid_m*100:.0f} cm on known floor points "
          f"(mode '{scene.depth_mode}', scale x{scene.depth_scale:.3f})")

    fm = scene.floor_mask
    check("floor coverage", 0.05 <= fm.mean() <= 0.75,
          f"{100*fm.mean():.0f}% of pixels are floor")
    free = scene.free_space
    seen = np.isfinite(scene.obstacle)
    area = free.sum() * scene.cell_m ** 2
    check("free floor area", area > 2.0,
          f"{area:.1f} m^2 standable of {seen.sum()*scene.cell_m**2:.1f} m^2 observed")

    # -- 3. does the depth scene explain the stature artefact? --------------
    print("\n[3/5] occlusion audit")
    obs_all = collect_observations(traj, model, stride=2)
    fpx = obs_all.foot[:, :2] / obs_all.foot[:, 2:] * s + np.array([W / 2, H / 2])
    xy_c, st_c, vis = localize(gp, scene, fpx, obs_all.foot, obs_all.head,
                               args.mean_stature)
    st_raw = gp.stature_m(obs_all.foot, obs_all.head)
    good_raw = np.isfinite(st_raw) & (st_raw > 0.6) & (st_raw < 2.6)
    print(f"      feet visible in {100*vis.mean():.0f}% of {len(vis)} observations")

    def within_track_range(st, tid, gx, gy, nb=24):
        """Spread of one person's measured height across the places they stood.
        Keyed by track so the three variants can be compared over the *same*
        people - they survive gating at different rates, and a before/after
        computed over different populations would measure that instead."""
        rs = {}
        for t in np.unique(tid):
            m = (tid == t) & np.isfinite(st)
            if m.sum() < 30:
                continue
            cell = gy[m] * nb + gx[m]
            meds = [np.median(st[m][cell == c]) for c in np.unique(cell)
                    if (cell == c).sum() >= 5]
            if len(meds) >= 2:
                rs[int(t)] = float(max(meds) - min(meds))
        return rs

    NB = 24
    gx = np.clip((fpx[:, 0] / W * NB).astype(int), 0, NB - 1)
    gy = np.clip((fpx[:, 1] / H * NB).astype(int), 0, NB - 1)

    # Compare on the *same* tracks: gating changes which tracks survive, and a
    # before/after over different populations measures nothing.
    r_raw = within_track_range(np.where(good_raw, st_raw, np.nan), obs_all.track_id, gx, gy)
    r_gate = within_track_range(st_c, obs_all.track_id, gx, gy)
    common = sorted(set(r_raw) & set(r_gate))

    field = StatureField.fit(fpx, st_c, obs_all.track_id, gp)
    st_f = field.apply(fpx, st_c, gp)
    r_fld = within_track_range(st_f, obs_all.track_id, gx, gy)
    common = sorted(set(common) & set(r_fld))
    m_raw = float(np.median([r_raw[t] for t in common])) if common else np.nan
    m_gate = float(np.median([r_gate[t] for t in common])) if common else np.nan
    m_fld = float(np.median([r_fld[t] for t in common])) if common else np.nan
    check("stature position artefact", np.isfinite(m_fld) and m_fld < 0.08,
          f"within-track range {m_raw*100:.0f} -> {m_gate*100:.0f} (gated) -> "
          f"{m_fld*100:.0f} cm (bias field), over {len(common)} common tracks")

    # Split-half reproducibility: the field must be scene structure, not a fit
    # to noise. Two disjoint halves of the tracklets must agree.
    uid = np.unique(obs_all.track_id)
    half = np.isin(obs_all.track_id, uid[::2])
    try:
        fA = StatureField.fit(fpx[half], st_c[half], obs_all.track_id[half], gp)
        fB = StatureField.fit(fpx[~half], st_c[~half], obs_all.track_id[~half], gp)
        both = fA.seen & fB.seen
        r = (np.corrcoef(fA.bias[both], fB.bias[both])[0, 1] if both.sum() > 3
             else float("nan"))
    except RuntimeError:
        r, both = float("nan"), np.zeros(1, bool)
    check("bias field reproducible", np.isfinite(r) and r > 0.4,
          f"split-half correlation r={r:+.3f} over {int(both.sum())} shared cells "
          "(want > 0.4)")
    (out / "stature_field.json").write_text(json.dumps(field.to_dict()))

    # -- 4. images ----------------------------------------------------------
    print("\n[4/5] image pack")
    rgb = cv2.cvtColor(bg_u, cv2.COLOR_BGR2RGB)

    # 01 distortion
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); ax[0].set_title("raw")
    ax[1].imshow(rgb); ax[1].set_title(f"undistorted  k1={k1:+.3f}")
    for a in ax:
        a.axis("off")
    fig.suptitle("01  Distortion - straight edges of the room should now be straight")
    plt.tight_layout(); plt.savefig(out / "01_distortion.png", dpi=110); plt.close()

    # 02 distortion score curve
    if curve:
        plt.figure(figsize=(6, 3.2))
        plt.plot([c[0] for c in curve], [c[1] for c in curve], ".-")
        plt.axvline(k1, color="r", ls="--", label=f"k1={k1:+.3f}")
        plt.xlabel("division-model k1"); plt.ylabel("straight-line length$^2$")
        plt.title("02  Plumb-line distortion fit"); plt.legend()
        plt.tight_layout(); plt.savefig(out / "02_distortion_fit.png", dpi=110); plt.close()

    # 03 horizon + observations
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(rgb)
    xs = np.linspace(-1.6, 1.6, 200)
    if abs(horizon[1]) > 1e-6:
        ys = -(horizon[0] * xs + horizon[2]) / horizon[1]
        ax.plot(xs * s + W / 2, ys * s + H / 2, "c-", lw=2, label="horizon")
    fp = obs.foot[:, :2] / obs.foot[:, 2:] * s + np.array([W / 2, H / 2])
    hp = obs.head[:, :2] / obs.head[:, 2:] * s + np.array([W / 2, H / 2])
    ax.plot(fp[hmask][:, 0], fp[hmask][:, 1], "g.", ms=2, label="feet (inlier)")
    ax.plot(hp[hmask][:, 0], hp[hmask][:, 1], "y.", ms=2, label="heads")
    ax.plot(fp[~hmask][:, 0], fp[~hmask][:, 1], "r.", ms=2, label="rejected")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"03  Horizon + observations  (rel-rms {rel_rms:.1%})"); ax.axis("off")
    plt.tight_layout(); plt.savefig(out / "03_horizon.png", dpi=110); plt.close()

    # 04 depth + height above floor
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    im0 = ax[0].imshow(depth * scene.depth_scale, cmap="turbo")
    ax[0].set_title("depth, rescaled to the floor (m)")
    plt.colorbar(im0, ax=ax[0], fraction=0.03)
    im1 = ax[1].imshow(scene.height, cmap="coolwarm", vmin=-0.5, vmax=2.5)
    ax[1].set_title("height above floor (m)")
    plt.colorbar(im1, ax=ax[1], fraction=0.03)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"04  Metric depth  (floor residual {scene.floor_resid_m*100:.0f} cm)")
    plt.tight_layout(); plt.savefig(out / "04_depth.png", dpi=110); plt.close()

    # 05 the derived occluder map
    ov = rgb.copy().astype(np.float32)
    tint = np.zeros_like(ov)
    tint[fm] = (0, 220, 0)
    occ = scene.height > STEP_OVER_M
    tint[occ] = (230, 40, 40)
    ov = np.clip(ov * 0.55 + tint * 0.45, 0, 255).astype(np.uint8)
    plt.figure(figsize=(10, 5.6)); plt.imshow(ov); plt.axis("off")
    plt.title("05  Derived occluder map - green = floor a foot can rest on, "
              "red = obstacle\n(no zones were drawn; this comes from the depth model)")
    plt.tight_layout(); plt.savefig(out / "05_occluder_map.png", dpi=110); plt.close()

    # 06 top-down
    ext = [scene.origin[0], scene.origin[0] + scene.obstacle.shape[1] * scene.cell_m,
           scene.origin[1] + scene.obstacle.shape[0] * scene.cell_m, scene.origin[1]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    im = ax[0].imshow(scene.obstacle, extent=ext, cmap="magma", vmin=0, vmax=2.0)
    plt.colorbar(im, ax=ax[0], label="obstacle height (m)")
    ax[0].set_title("obstacle height, top-down")
    ax[1].imshow(free.astype(float), extent=ext, cmap="Greens", vmin=0, vmax=1.4)
    tid = obs_all.track_id
    for t in np.unique(tid):
        m = (tid == t) & np.isfinite(xy_c).all(axis=1)
        if m.sum() > 5:
            ax[1].plot(xy_c[m, 0], xy_c[m, 1], ".", ms=1.5)
    ax[1].set_title("free space (green) + walked paths")
    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("m"); a.set_ylabel("m")
    fig.suptitle("06  Metric floor plan")
    plt.tight_layout(); plt.savefig(out / "06_floorplan.png", dpi=110); plt.close()

    # 07 stature before/after occlusion gating
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].hist(st_raw[good_raw], bins=50, alpha=.65, label=f"all feet (n={good_raw.sum()})")
    fin = np.isfinite(st_c)
    ax[0].hist(st_c[fin], bins=50, alpha=.65, label=f"feet visible (n={fin.sum()})")
    ax[0].axvline(args.mean_stature, color="r", ls="--")
    ax[0].set_xlabel("stature (m)"); ax[0].legend(); ax[0].set_title("stature distribution")
    if common:
        ax[1].boxplot([[r_raw[t] * 100 for t in common],
                       [r_gate[t] * 100 for t in common],
                       [r_fld[t] * 100 for t in common]],
                      tick_labels=["raw", "+gated", "+bias field"])
    ax[1].set_ylabel("within-track stature range (cm)")
    ax[1].set_title("same person, different places\n(any spread here is a geometry error)")
    fig.suptitle("07  Does occlusion gating fix stature?")
    plt.tight_layout(); plt.savefig(out / "07_stature.png", dpi=110); plt.close()

    # 08 the metre check: 1 m grid + 1.7 m silhouettes back-projected
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.imshow(rgb)
    lo = scene.origin
    hi = lo + np.array(scene.obstacle.shape[::-1]) * scene.cell_m
    g0 = np.ceil(lo)
    for gxv in np.arange(g0[0], hi[0], 1.0):
        pts = np.stack([np.full(60, gxv), np.linspace(lo[1], hi[1], 60)], axis=1)
        uv = project(gp, floor_to_cam(gp, pts))
        ax.plot(uv[:, 0], uv[:, 1], "-", color="deepskyblue", lw=1, alpha=.9)
    for gyv in np.arange(np.ceil(lo[1]), hi[1], 1.0):
        pts = np.stack([np.linspace(lo[0], hi[0], 60), np.full(60, gyv)], axis=1)
        uv = project(gp, floor_to_cam(gp, pts))
        ax.plot(uv[:, 0], uv[:, 1], "-", color="deepskyblue", lw=1, alpha=.9)
    # silhouettes on actually-walked ground
    walked = xy_c[np.isfinite(xy_c).all(axis=1)]
    if len(walked) > 8:
        sel = walked[np.linspace(0, len(walked) - 1, 7).astype(int)]
        for p in sel:
            b = project(gp, floor_to_cam(gp, p[None], 0.0))[0]
            t = project(gp, floor_to_cam(gp, p[None], args.mean_stature))[0]
            if not np.isfinite([b, t]).all():
                continue
            ax.plot([b[0], t[0]], [b[1], t[1]], "-", color="yellow", lw=3, alpha=.85)
            ax.plot(b[0], b[1], "o", color="yellow", ms=5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"08  METRE CHECK - 1 m floor grid + {args.mean_stature:.2f} m people.\n"
                 "Accept the calibration only if the squares look square on the floor "
                 "and the yellow sticks match real people's height.")
    plt.tight_layout(); plt.savefig(out / "08_metre_check.png", dpi=115); plt.close()

    # 09 the fitted bias field, over the scene that causes it
    fld = np.where(field.seen, np.exp(field.bias), np.nan).reshape(field.nb, field.nb)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))
    ax[0].imshow(rgb)
    im = ax[0].imshow(np.kron(fld, np.ones((H // field.nb + 1, W // field.nb + 1)))
                      [:H, :W], alpha=0.6, cmap="RdYlGn_r", vmin=0.85, vmax=1.15,
                      extent=[0, W, H, 0])
    plt.colorbar(im, ax=ax[0], label="measured / true stature")
    ax[0].set_title("where the geometry lies about height"); ax[0].axis("off")
    if np.isfinite(r):
        ax[1].plot(np.exp(fA.bias[both]), np.exp(fB.bias[both]), "o", ms=4)
        lim = [0.8, 1.2]
        ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
        ax[1].set_xlabel("field from tracklet half A"); ax[1].set_ylabel("half B")
        ax[1].set_aspect("equal")
    ax[1].set_title(f"reproducible across disjoint tracks?  r={r:+.3f}")
    fig.suptitle("09  Stature bias field - fitted from unlabelled tracklets, "
                 "validated by split-half agreement")
    plt.tight_layout(); plt.savefig(out / "09_stature_field.png", dpi=110); plt.close()

    # -- 5. write it down ---------------------------------------------------
    print("\n[5/5] report")
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    lines = [f"# Commissioning report - {args.name}", "",
             f"input: `{args.input}`  ", f"tracklets: `{traj_path}`  ",
             f"result: **{n_pass}/{len(CHECKS)} checks pass**", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'WARN'} | {dt} |" for n, ok, dt in CHECKS]
    lines += ["", "## Calibration", "```json",
              json.dumps(json.loads((out / "scene.json").read_text()), indent=2), "```",
              "", "## Image pack", ""]
    lines += [f"- `{p.name}`" for p in sorted(out.glob("0*.png"))]
    lines += ["", "Sheet 08 is the acceptance test: if the 1 m grid looks square on "
              "the floor and the silhouettes match real people, the metre scale is "
              "correct. Everything downstream (speed limits, reachability, stature) "
              "inherits that scale."]
    (out / "commission.md").write_text("\n".join(lines))
    print(f"\n=== {n_pass}/{len(CHECKS)} checks pass -> {out}/ ===")
    for p in sorted(out.glob("*.png")):
        print(f"    {p}")


if __name__ == "__main__":
    main()
