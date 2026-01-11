#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Draw ROI JSON on the first frame of a video/image for quick visual inspection.
Usage:
    python visual_roi.py -data <video_or_image> -speedzone <speed_zone.json> --table_zone <table_zone.json>
Output:
    <video_stem>_roi_vis.jpg
"""
import json
import cv2
import numpy as np
import sys
from argparse import ArgumentParser
from pathlib import Path

def load_roi(path: Path):
    raw = json.load(open(path, encoding="utf-8"))
    pts = np.array(raw["pts"], np.int32)
    return raw.get("type", "poly"), pts

def get_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # maybe an image
        img = cv2.imread(str(video_path))
        if img is None:
            raise RuntimeError("Unable to read image")
        return img
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Unable to read the first frame of the video")
    return frame

def visualize(video_path: Path, rois: list, out_path: Path):
    canvas = get_first_frame(video_path).copy()
    for name, typ, pts in rois:
        color = (0, 255, 0) if "speed" in name.lower() else (255, 0, 0)
        # semi-transparent fill
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        # 外框
        cv2.polylines(canvas, [pts], True, color, 2)
        # 文字標籤
        x, y = pts[0]
        cv2.putText(canvas, name.replace("_zone", ""), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), canvas)
    print("Saved visualization image →", out_path.resolve())
    cv2.imshow("ROI Visualization (press any key to close)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_opt():
    parser = ArgumentParser(description="Visualize ROI JSON on the first frame of a video/image")
    parser.add_argument('--source', required=True, help='path to video or image')
    parser.add_argument('--speedzone', dest='speedzone', required=True, help='path to speed zone json')
    # parser.add_argument('--tablezone', dest='table_zone', required=False, help='path to table zone json')
    return parser.parse_args()


def main(opt):
    video_path = Path(opt.source)
    if not video_path.exists():
        print("File not found:", video_path)
        sys.exit(2)

    json_files = [Path(opt.speedzone)]
    # if getattr(opt, 'tablezone', None):
    #     json_files.append(Path(opt.table_zone))

    rois = []
    for p in json_files:
        if not p.exists():
            print("Warning: not found", p)
            continue
        typ, pts = load_roi(p)
        rois.append((p.stem, typ, pts))

    if not rois:
        print("No valid JSON files provided")
        sys.exit(2)

    out_path = video_path.parent / f"{video_path.stem}_roi_vis.jpg"
    visualize(video_path, rois, out_path)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
