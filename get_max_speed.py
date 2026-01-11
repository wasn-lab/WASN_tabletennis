# python ver4\get_max_speed.py --csv 048\C0045_predict_speed.csv --output 048
import os
import re
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="frame-level csv")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--min_dx", type=float, default=0.0)
    parser.add_argument("--max_frame_gap", type=int, default=1)
    parser.add_argument("--min_run_len", type=int, default=2)
    parser.add_argument("--use_valid_speed_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --------------------------------------------------
    # read csv
    # --------------------------------------------------
    df = pd.read_csv(args.csv)

    need_cols = ["frame_num", "x", "in_zone", "speed_km_hr"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df = df.sort_values("frame_num").reset_index(drop=True)

    # --------------------------------------------------
    # decide right-moving frames
    # --------------------------------------------------
    dx = df["x"].diff()
    right_mask = (df["in_zone"] == 1) & (dx > args.min_dx)
    right_mask = right_mask.fillna(False)

    frames = df["frame_num"].to_numpy()

    # --------------------------------------------------
    # group frames into strokes
    # --------------------------------------------------
    strokes = []
    current = []

    for i in range(len(df)):
        if not right_mask.iloc[i]:
            if len(current) >= args.min_run_len:
                strokes.append(current.copy())
            current = []
            continue

        if not current:
            current = [i]
        else:
            prev_i = current[-1]
            if frames[i] - frames[prev_i] <= args.max_frame_gap:
                current.append(i)
            else:
                if len(current) >= args.min_run_len:
                    strokes.append(current.copy())
                current = [i]

    if len(current) >= args.min_run_len:
        strokes.append(current.copy())

    # --------------------------------------------------
    # build output table
    # --------------------------------------------------
    rows = []

    for sid, idxs in enumerate(strokes, start=1):
        sub = df.iloc[idxs]

        speeds = sub["speed_km_hr"]
        if args.use_valid_speed_only and "valid_speed" in sub.columns:
            speeds = speeds[sub["valid_speed"] == True]

        max_speed = speeds.max()

        rows.append({
            "start_frame": int(sub["frame_num"].iloc[0]),
            "end_frame": int(sub["frame_num"].iloc[-1]),
            "stroke_id": sid,
            "max_speed": float(max_speed)
        })

    out_df = pd.DataFrame(rows)

    # --------------------------------------------------
    # add mean row
    # --------------------------------------------------
    mean_speed = out_df["max_speed"].mean() if len(out_df) > 0 else np.nan

    out_df.loc[len(out_df)] = {
        "start_frame": "",
        "end_frame": "",
        "stroke_id": "mean_max_speed",
        "max_speed": mean_speed
    }

    # --------------------------------------------------
    # output filename (Cxxxx)
    # --------------------------------------------------
    base = os.path.basename(args.csv)
    video_id = re.match(r"(C\d+)", base).group(1)
    out_name = f"{video_id}_right_strokes_segments.csv"

    out_path = os.path.join(args.output, out_name)
    out_df.to_csv(out_path, index=False)

    print(f"saved: {out_path}")
    print(f"strokes: {len(out_df) - 1}, mean_max_speed: {mean_speed:.2f}")


if __name__ == "__main__":
    main()
