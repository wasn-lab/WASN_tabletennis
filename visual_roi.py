#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 ROI JSON 畫在「影片第一幀」上方便肉眼驗收
用法：
  python visualize_roi.py your_video.mp4 speed_zone.json [table_zone.json ...]
輸出：
  your_video_roi_vis.jpg
"""
import json
import cv2
import numpy as np
import sys
from pathlib import Path

def load_roi(path: Path):
    raw = json.load(open(path, encoding="utf-8"))
    pts = np.array(raw["pts"], np.int32)
    return raw.get("type", "poly"), pts

def get_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # 可能是一張圖
        img = cv2.imread(str(video_path))
        if img is None:
            raise RuntimeError("無法讀取影像")
        return img
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("無法讀取影片第一幀")
    return frame

def visualize(video_path: Path, rois: list, out_path: Path):
    canvas = get_first_frame(video_path).copy()
    for name, typ, pts in rois:
        color = (0, 255, 0) if "speed" in name.lower() else (255, 0, 0)
        # 半透明填滿
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
    print("已儲存視覺化圖片 →", out_path.resolve())
    cv2.imshow("ROI 視覺化（任意鍵關閉）", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 3:
        print("用法: python visualize_roi.py 影片/圖片  json1 [json2 ...]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    json_files = [Path(p) for p in sys.argv[2:]]

    rois = []
    for p in json_files:
        if not p.exists():
            print("警告：找不到", p); continue
        typ, pts = load_roi(p)
        rois.append((p.stem, typ, pts))

    if not rois:
        print("沒有任何有效 JSON"); sys.exit(2)

    out_path = video_path.parent / f"{video_path.stem}_roi_vis.jpg"
    visualize(video_path, rois, out_path)

if __name__ == "__main__":
    main()
