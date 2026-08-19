"""Calibrate a fixed camera's metric ground plane from its own footage.

    python scripts/calibrate_scene.py --video videos/gunsan_test.mp4 \
        --traj runs/traj/gunsan_test.json --out calib/gunsan_test.json

Requires only a video and the motion-only tracklets from it. Produces a
`GroundPlane` JSON plus diagnostics (undistortion before/after, horizon fit,
floor plan, stature distribution) so the fit can be judged rather than
trusted blindly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reid.scene_geometry import (  # noqa: E402
    GroundPlane,
    RadialModel,
    background_frame,
    collect_observations,
    estimate_radial_k1,
    fit_height_field,
    focal_from_depth,
    focal_from_stature_consistency,
    vertical_vanishing_point,
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def metric_depth(img_bgr: np.ndarray) -> np.ndarray:
    """Metric depth (metres) for one image. Imported lazily so that the rest of
    the calibration works on machines without the depth model."""
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
        mode="bicubic", align_corners=False,
    )[0, 0]
    return d.cpu().numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--traj", required=True, help="JSON: {W,H,frames,traj{tid:[[f,x1,y1,x2,y2]..]}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-dir", default=None)
    ap.add_argument("--mean-stature", type=float, default=1.70)
    ap.add_argument("--k1", type=float, default=None, help="skip distortion estimation")
    ap.add_argument("--no-depth", action="store_true", help="skip the depth cross-check")
    args = ap.parse_args()

    diag = Path(args.diag_dir) if args.diag_dir else Path(args.out).parent / "diag"
    diag.mkdir(parents=True, exist_ok=True)

    d = json.loads(Path(args.traj).read_text())
    W, H = d["W"], d["H"]
    traj = {int(k): sorted(v) for k, v in d["traj"].items()}
    print(f"[scene] {W}x{H}, {len(traj)} tracklets")

    # 1. radial distortion ---------------------------------------------------
    bg = background_frame(args.video)
    cv2.imwrite(str(diag / "background.jpg"), bg)
    if args.k1 is not None:
        k1, curve = args.k1, []
        print(f"[dist] using supplied k1={k1:+.4f}")
    else:
        k1, curve = estimate_radial_k1(bg)
        ks = np.array([c[0] for c in curve])
        ss = np.array([c[1] for c in curve])
        rel = (ss.max() - ss.min()) / max(ss.max(), 1e-9)
        print(f"[dist] k1={k1:+.4f}  (score contrast {rel:.1%}"
              f"{'  -- WEAK, few straight edges' if rel < 0.05 else ''})")
        plt.figure(figsize=(6, 3.2))
        plt.plot(ks, ss, ".-")
        plt.axvline(k1, color="r", ls="--", label=f"k1={k1:+.4f}")
        plt.xlabel("division-model k1"); plt.ylabel("straight-line length$^2$")
        plt.title("Plumb-line distortion fit"); plt.legend(); plt.tight_layout()
        plt.savefig(diag / "distortion_fit.png", dpi=120); plt.close()

    model = RadialModel(k1, W, H)
    cv2.imwrite(str(diag / "background_undistorted.jpg"), model.undistort_image(bg))

    # 2. observations --------------------------------------------------------
    obs = collect_observations(traj, model)
    print(f"[obs ] {len(obs)} standing full-body observations "
          f"from {len(set(obs.track_id.tolist()))} tracklets")

    # 3. conditioning check on the classical vertical-VP route ---------------
    _, vmask_vp = vertical_vanishing_point(obs)
    fv = obs.foot[:, :2] / obs.foot[:, 2:] - obs.head[:, :2] / obs.head[:, 2:]
    spread = float(np.ptp(np.percentile(np.degrees(np.arctan2(fv[:, 0], fv[:, 1])), [5, 95])))
    print(f"[vp   ] head->foot angular spread {spread:.1f}deg "
          f"({'ill-conditioned, using height field' if spread < 15 else 'usable'}); "
          f"vp inliers {vmask_vp.sum()}/{len(obs)}")

    # 4. horizon from the apparent-height field ------------------------------
    horizon, hmask, rel_rms = fit_height_field(obs)
    print(f"[hrzn] {horizon.round(4).tolist()}  inliers {hmask.sum()}/{len(obs)}  "
          f"rel-rms {rel_rms:.1%}"
          f"{'  -- POOR, planar assumption suspect' if rel_rms > 0.06 else ''}")

    # 5. focal length --------------------------------------------------------
    # Primary estimator: stature must not depend on where a person stands.
    # Self-contained and, on this footage, sharper than the depth route below.
    s = max(W, H) / 2.0
    f, spread, (f_lo, f_hi), nf = focal_from_stature_consistency(obs, horizon, mask=hmask)
    print(f"[focal] f={f*s:.0f}px (fov {2*np.degrees(np.arctan(1.0/f)):.0f}deg)  "
          f"stature IQR/med {spread:.3f}  near/far bias {nf:.3f}  "
          f"5%-band {f_lo*s:.0f}-{f_hi*s:.0f}px")
    if abs(nf - 1.0) > 0.05:
        print("[focal] WARNING residual range bias >5%: stature will encode position")

    # Independent cross-check from a metric depth model (optional).
    cam_h_depth = None
    if not args.no_depth:
        try:
            depth = metric_depth(model.undistort_image(bg))
            foot_norm = obs.foot[hmask][:, :2] / obs.foot[hmask][:, 2:]
            f_d, _, cam_h_depth, ang = focal_from_depth(depth, foot_norm, horizon, model)
            np.save(diag / "depth_background.npy", depth.astype(np.float32))
            print(f"[depth] cross-check: f={f_d*s:.0f}px, plane-vs-horizon {ang:.1f}deg, "
                  f"camera height {cam_h_depth:.2f}m (not used for the fit)")
        except Exception as e:  # depth is a nicety, not a dependency
            print(f"[depth] cross-check unavailable: {e}")

    # 6. ground plane --------------------------------------------------------
    gp = GroundPlane.from_horizon(horizon, f, obs, model,
                                  mean_stature_m=args.mean_stature, mask=hmask)
    ch = abs(gp.cam_height_m)
    msg = f"[plane] camera height (stature-scaled) {ch:.2f}m"
    if cam_h_depth:
        msg += (f"  vs depth {cam_h_depth:.2f}m  -> agreement "
                f"{100*min(ch, cam_h_depth)/max(ch, cam_h_depth):.0f}%")
    print(msg)
    print(f"[plane] up={gp.r3.round(3).tolist()}")
    gp.save(args.out)
    print(f"[save ] {args.out}")

    # diagnostics ------------------------------------------------------------
    vmask = hmask
    xy = gp.floor_xy(obs.foot[vmask])
    st = gp.stature_m(obs.foot[vmask], obs.head[vmask])
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(st)
    tid = obs.track_id[vmask][ok]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for t in np.unique(tid):
        m = tid == t
        ax[0].plot(xy[ok][m, 0], xy[ok][m, 1], ".", ms=3, label=f"{t}")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("m"); ax[0].set_ylabel("m")
    ax[0].set_title("Floor plan - tracklet foot positions (metres)")
    ax[1].hist(st[ok], bins=40)
    ax[1].axvline(args.mean_stature, color="r", ls="--", label="assumed mean")
    ax[1].set_xlabel("stature (m)"); ax[1].set_title("Recovered stature")
    ax[1].legend(); plt.tight_layout()
    plt.savefig(diag / "floorplan_stature.png", dpi=120); plt.close()

    span = np.nanpercentile(xy[ok], [2, 98], axis=0)
    print(f"[diag ] floor extent ~{span[1,0]-span[0,0]:.1f}m x {span[1,1]-span[0,1]:.1f}m")
    print(f"[diag ] stature p10/p50/p90 = "
          f"{np.percentile(st[ok],10):.2f}/{np.percentile(st[ok],50):.2f}/"
          f"{np.percentile(st[ok],90):.2f} m")
    print(f"[diag ] wrote {diag}/")


if __name__ == "__main__":
    main()
