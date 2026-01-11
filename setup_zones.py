#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI setup script (simplified)
 1. Speed-Zone  → arbitrary polygon (>= 3 points)
 2. Table-Zone  → quadrilateral (4 points)
After saving, the main program will no longer prompt.
"""
import cv2
import numpy as np
import json
import sys
from pathlib import Path
from argparse import ArgumentParser

JSON_VER = 1


# ------------------------------------------------
# Shared utilities
# ------------------------------------------------
def shoelace(pts: np.ndarray) -> float:
    return float(cv2.contourArea(pts))

# ------------------------------------------------
# Generic ROI editor (used for both Polygon and Quad)
# ------------------------------------------------
class RoiEditor:
    def __init__(self, win_name, img, save_path, need_exact=3):
        """
        need_exact: minimum number of points; 3 = polygon, 4 = quad
        """
        self.win_name   = win_name
        self.img        = img
        self.h, self.w  = img.shape[:2]
        self.save_path  = Path(save_path)
        self.need_exact = need_exact        # 3 → poly , 4 → quad
        self.pts        = []
        self.drag_idx   = -1
        self._init_shape()

    # --- Initial shape ---
    def _init_shape(self):
        margin = int(min(self.w, self.h) * 0.2)
        if self.need_exact == 4:          # quadrilateral
            self.pts = [(margin, margin),
                        (self.w - margin, margin),
                        (self.w - margin, self.h - margin),
                        (margin, self.h - margin)]
        else:                             #  polygon → start with 4 points for easier dragging
            self.pts = [(margin, margin),
                        (self.w - margin, margin),
                        (self.w - margin, self.h - margin),
                        (margin, self.h - margin)]

    # --- Draw ---
    def redraw(self):
        tmp = self.img.copy()
        n = len(self.pts)
        if n >= 3:
            pts = np.array(self.pts, np.int32)
            overlay = tmp.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, tmp, 0.7, 0, tmp)
            cv2.polylines(tmp, [pts], True, (0, 255, 255), 2)
        for i, (x, y) in enumerate(self.pts):
            color = (0, 0, 255) if i == self.drag_idx else (255, 0, 0)
            cv2.circle(tmp, (int(x), int(y)), 6, color, -1)
        cv2.imshow(self.win_name, tmp)

    # --- Mouse callback ---
    def mouse_cb(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pts:
                dists = [((x - px) ** 2 + (y - py) ** 2) ** .5
                         for px, py in self.pts]
                if min(dists) < 15:
                    self.drag_idx = int(np.argmin(dists))
                    return
            # Add point (allowed only in polygon mode)
            if self.need_exact != 4:
                self.pts.append([x, y])
                self.drag_idx = -1
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx != -1:
            self.pts[self.drag_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1
        self.redraw()

    # --- Main loop ---
    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.w // 2, self.h // 2)
        cv2.setMouseCallback(self.win_name, self.mouse_cb)
        self.redraw()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:                            # Enter
                if self._validate():
                    self._save()
                    break
            elif key == ord("r"):                    # reset
                self._init_shape()
            elif key == 27:                          # ESC
                print("User aborted")
                sys.exit(0)
        cv2.destroyWindow(self.win_name)
        return self.pts

    # --- Validation & save ---
    def _validate(self):
        n = len(self.pts)
        if n < self.need_exact:
            return False
        return shoelace(np.array(self.pts, np.float32)) > 100

    def _save(self):
        data = {"version": JSON_VER,
                "type": "poly" if self.need_exact != 4 else "quad",
                "pts": self.pts}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Saved →", self.save_path.resolve())

# ------------------------------------------------
# main flow
# ------------------------------------------------
def parse_opt():
    parser = ArgumentParser(description="Setup ROI zones from a video or image")
    parser.add_argument('--source', required=True, help='path to video or image')
    parser.add_argument('--speedzone', dest='speedzone', default=Path(".")/ "speed_zone.json" ,
                        help='path to speed zone json')
    parser.add_argument('--table_zone', dest='table_zone', default=Path(".")/ "table_zone.json" ,
                        help='path to table zone json')
    return parser.parse_args()

def main(opt):
    p = Path(opt.source)
    if not p.exists():
        print("File not found:", p)
        sys.exit(2)

    # read first frame 
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        img = cv2.imread(str(p))
        if img is None:
            print("Unable to read image"); sys.exit(2)
    else:
        ok, img = cap.read(); cap.release()
        if not ok:
            print("Unable to read the first frame of the video"); sys.exit(2)

    # check existing files (use paths supplied via CLI)
    speed_path = Path(opt.speedzone) if getattr(opt, 'speedzone', None) else SPEED_PATH
    table_path = Path(opt.table_zone) if getattr(opt, 'table_zone', None) else TABLE_PATH
    if speed_path.exists() or table_path.exists():
        if input("Existing zone files detected. Overwrite? (y/N) ").strip().lower() != "y":
            print("Keeping old files, exiting"); return

    # in sequence edition
    RoiEditor("1. Drag Speed-Zone (Polygon)", img, speed_path, need_exact=3).run()
    RoiEditor("2. Drag Table-Zone (Quad)",  img, table_path, need_exact=4).run()

    print("ROI setup complete!")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
