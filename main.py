# main.py
import argparse
import subprocess
import os
import sys

def run(cmd):
    print("\n✅ Running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run_all")
    p.add_argument("--input_videos", required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--engineered_csv", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--num_frames", type=int, default=30)
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 27, 42])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--no_shap", action="store_true")
    p.add_argument("--fps_json", default=None)

    args = ap.parse_args()

    if args.cmd == "run_all":
        seg_dir = os.path.join(args.work_dir, "segmentation")
        norm_dir = os.path.join(args.work_dir, "normalized2")
        feat_csv = os.path.join(args.work_dir, "FEATURE_BANK_FULL.csv")
        clf_dir  = os.path.join(args.work_dir, "classification")

        os.makedirs(args.work_dir, exist_ok=True)

        # 1) YOLO segmentation
        run([
            sys.executable, "Yolov8segmentation.py",
            "--input_videos", args.input_videos,
            "--weights", args.yolo_weights,
            "--out_dir", seg_dir,
            "--num_frames", str(args.num_frames),
        ])

        # 2) Optical flow
        run([
            sys.executable, "TV-L1 OPTICAL FLOW.py",
            "--input_videos", args.input_videos,
            "--masks_dir", seg_dir,
            "--out_dir", norm_dir,
            "--num_frames", str(args.num_frames),
        ])

        # 3) Feature bank
        cmd_fb = [
            sys.executable, "Feature-bank.py",
            "--engineered_csv", args.engineered_csv,
            "--normalized_dir", norm_dir,
            "--out_csv", feat_csv,
        ]
        if args.fps_json:
            cmd_fb += ["--fps_json", args.fps_json]
        run(cmd_fb)

        # 4) Classification
        cmd_clf = [
            sys.executable, "classification.py",
            "--feature_bank", feat_csv,
            "--normalized_dir", norm_dir,
            "--output_dir", clf_dir,
            "--n_splits", str(args.n_splits),
            "--seeds", *[str(s) for s in args.seeds],
        ]
        if args.no_shap:
            cmd_clf += ["--no_shap"]
        run(cmd_clf)

        print("\n🎉 DONE. Outputs in:", args.work_dir)

if __name__ == "__main__":
    main()
