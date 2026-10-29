#!/usr/bin/env python
"""CAMOS end-to-end inference: pre-operative face STL -> predicted post-operative face STL.

Example
-------
python run_camos.py --stl patient/Face.stl --out results/patient \
    --mvqvae MVQVAE_PATH --cvar CAMOS_PATH
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camos import landmarks as lm            # noqa: E402
from camos import align, preprocess, surface  # noqa: E402
from camos.indices import all_idx            # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stl", required=True, help="pre-operative facial surface (STL, mm, face toward -y, +z up)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--mvqvae", required=True, help="multi-scale VQ-VAE checkpoint (best.pt)")
    p.add_argument("--cvar", required=True, nargs="+",
                   help="fine-tuned CAMOS checkpoint(s); several checkpoints are ensembled by averaging")
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")
    p.add_argument("--seed", type=int, default=0)
    # landmark extraction
    p.add_argument("--landmarks", default=None,
                   help="skip rendering: 3D Slicer markup json with the 478 MediaPipe landmarks on the mesh")
    p.add_argument("--sparse-views", action="store_true", help="25 views instead of 81 (faster, less accurate)")
    p.add_argument("--c-dist", type=float, default=3.0, help="camera distance factor (x face width)")
    p.add_argument("--window", type=int, default=800, help="render size in pixels")
    p.add_argument("--keep-images", action="store_true", help="keep the rendered views in the work folder")
    # geometry
    p.add_argument("--align", action="store_true",
                   help="re-align the face from its landmarks (off by default: the STL is assumed to be "
                        "in the standard frame already, like the training data)")
    p.add_argument("--crop", action="store_true", help="also save the cropped facial region")
    p.add_argument("--tps-lambda", type=float, default=0.001)
    # expert / reproduction options
    p.add_argument("--stats-json", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.stl))[0]
    work = os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)

    # ------------------------------------------------------------------ 0. load
    mesh_in = trimesh.load(a.stl, force="mesh")
    log(f"input mesh: {len(mesh_in.vertices)} vertices, bbox {np.round(mesh_in.bounds, 1).tolist()}")
    mesh = mesh_in.copy()
    stl_work = a.stl

    # ------------------------------------------------------------------ 1. landmarks
    if a.landmarks:
        log(f"using landmarks from {a.landmarks}")
        land478 = lm.read_markups(a.landmarks)
    else:
        log("1/5 extracting dense landmarks (render + MediaPipe + back-projection)")
        raw, land478 = lm.extract_landmarks(stl_work, os.path.join(work, "views"), sparse_views=a.sparse_views,
                                            window_size=a.window, c_dist=a.c_dist, keep_images=a.keep_images, log=log)
    lm.write_markups(land478, os.path.join(a.out, f"{name}_landmarks478_pre.mrk.json"))
    for w in align.orientation_report(land478):
        log(f"WARNING: {w}")

    # ------------------------------------------------------------------ 2. alignment
    M = align.canonical_transform(land478) if a.align else np.eye(4)
    land478_al = align.apply_transform(land478, M)
    mesh_al = mesh.copy()
    mesh_al.apply_transform(M)
    log("2/5 alignment " + (f"done (rotation {np.degrees(np.arccos(np.clip((np.trace(M[:3,:3])-1)/2,-1,1))):.2f} deg)"
                            if a.align else "skipped (input assumed to be in the standard frame)"))

    # ------------------------------------------------------------------ 3. preprocessing
    pre282 = preprocess.select_282(land478_al)
    if a.stats_json:
        with open(a.stats_json) as f:
            stats = {k: float(v) for k, v in json.load(f).items() if k in ("x_mean", "y_mean", "z_mean", "all_std")}
    else:
        stats = preprocess.normalization_stats(pre282)
    cond = preprocess.normalize(pre282, stats)
    log(f"3/5 preprocessing done (all_std {stats['all_std']:.2f} mm)")

    # ------------------------------------------------------------------ 4. prediction
    from camos import predict
    preds = []
    for ck in a.cvar:
        mvqvae, cvar = predict.load_models(a.mvqvae, ck, device=a.device)
        preds.append(predict.predict_normalized(mvqvae, cvar, cond, device=a.device, seed=a.seed))
        log(f"4/5 prediction with {os.path.basename(ck)} done")
    pred_norm = np.mean(preds, axis=0)
    pred282_al = preprocess.denormalize(pred_norm, stats)

    # ------------------------------------------------------------------ 5. surface
    warped_al = surface.warp_mesh(mesh_al, pre282, pred282_al, lambda_=a.tps_lambda)
    M_back = np.linalg.inv(M)                    # back to the input coordinate frame
    warped = warped_al.copy()
    warped.apply_transform(M_back)
    pred282 = align.apply_transform(pred282_al, M_back)
    out_stl = os.path.join(a.out, f"{name}_predicted.stl")
    warped.export(out_stl)
    lm.write_markups(pred282, os.path.join(a.out, f"{name}_landmarks282_predicted.mrk.json"),
                     labels=[f"mp{i}" for i in all_idx])
    lm.write_markups(align.apply_transform(pre282, M_back), os.path.join(a.out, f"{name}_landmarks282_pre.mrk.json"),
                     labels=[f"mp{i}" for i in all_idx])
    if a.crop:
        crop_in = align.crop_face(mesh_al, land478_al)
        crop_in_w = surface.warp_mesh(crop_in, pre282, pred282_al, lambda_=a.tps_lambda)
        for m_, tag in ((crop_in, "pre_crop"), (crop_in_w, "predicted_crop")):
            m_ = m_.copy()
            m_.apply_transform(M_back)
            m_.export(os.path.join(a.out, f"{name}_{tag}.stl"))
    disp = np.linalg.norm(pred282_al - pre282, axis=1)
    with open(os.path.join(a.out, f"{name}_camos.json"), "w") as f:
        json.dump({"input": os.path.abspath(a.stl), "stats": stats, "align_matrix": M.tolist(),
                   "checkpoints": [os.path.abspath(c) for c in a.cvar],
                   "mvqvae": os.path.abspath(a.mvqvae), "views": "25" if a.sparse_views else "81",
                   "landmark_displacement_mm": {"mean": float(disp.mean()), "max": float(disp.max())}}, f, indent=2)
    log(f"5/5 surface written: {out_stl}  (landmark displacement mean {disp.mean():.2f} mm, max {disp.max():.2f} mm)")


if __name__ == "__main__":
    main()
