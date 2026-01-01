#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI 設定腳本（精簡版）
 1. Speed-Zone  → 任意多邊形（≥3 點）
 2. Table-Zone  → 四邊形（4 點）
存檔後主程式不再彈窗。
"""
import cv2
import numpy as np
import json
import sys
from pathlib import Path

JSON_VER = 1
OUT_DIR   = Path(".")
SPEED_PATH = OUT_DIR / "speed_zone.json"
TABLE_PATH = OUT_DIR / "table_zone.json"

# ------------------------------------------------
# 共用小工具
# ------------------------------------------------
def shoelace(pts: np.ndarray) -> float:
    return float(cv2.contourArea(pts))

# ------------------------------------------------
# 通用 ROI 編輯器（Polygon / Quad 都走這裡）
# ------------------------------------------------
class RoiEditor:
    def __init__(self, win_name, img, save_path, need_exact=3):
        """
        need_exact: 最低點數；=3 表示任意多邊形，=4 強制四邊形
        """
        self.win_name   = win_name
        self.img        = img
        self.h, self.w  = img.shape[:2]
        self.save_path  = Path(save_path)
        self.need_exact = need_exact        # 3 → poly , 4 → quad
        self.pts        = []
        self.drag_idx   = -1
        self._init_shape()

    # --- 初始形狀 ---
    def _init_shape(self):
        margin = int(min(self.w, self.h) * 0.2)
        if self.need_exact == 4:          # 四邊形
            self.pts = [(margin, margin),
                        (self.w - margin, margin),
                        (self.w - margin, self.h - margin),
                        (margin, self.h - margin)]
        else:                             # 多邊形 → 先給 4 點方便拉
            self.pts = [(margin, margin),
                        (self.w - margin, margin),
                        (self.w - margin, self.h - margin),
                        (margin, self.h - margin)]

    # --- 畫圖 ---
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

    # --- 滑鼠回呼 ---
    def mouse_cb(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pts:
                dists = [((x - px) ** 2 + (y - py) ** 2) ** .5
                         for px, py in self.pts]
                if min(dists) < 15:
                    self.drag_idx = int(np.argmin(dists))
                    return
            # 加點（任意多邊形模式才允許）
            if self.need_exact != 4:
                self.pts.append([x, y])
                self.drag_idx = -1
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx != -1:
            self.pts[self.drag_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1
        self.redraw()

    # --- 主迴圈 ---
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
            elif key == ord("r"):                    # 重置
                self._init_shape()
            elif key == 27:                          # ESC
                print("使用者中斷")
                sys.exit(0)
        cv2.destroyWindow(self.win_name)
        return self.pts

    # --- 防呆 & 存檔 ---
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
        print("已儲存 →", self.save_path.resolve())

# ------------------------------------------------
# 主流程
# ------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("用法: python setup_zone.py 影片/圖片路徑")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print("檔案不存在:", p)
        sys.exit(2)

    # 讀第一幀
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        img = cv2.imread(str(p))
        if img is None:
            print("無法讀取影像"); sys.exit(2)
    else:
        ok, img = cap.read(); cap.release()
        if not ok:
            print("無法讀取影片第一幀"); sys.exit(2)

    # 覆蓋提示
    if SPEED_PATH.exists() or TABLE_PATH.exists():
        if input("偵測到舊 zone 檔，要覆蓋嗎？(y/N) ").strip().lower() != "y":
            print("保留舊檔，程式結束"); return

    # 依序編輯
    RoiEditor("1. 拖曳 Speed-Zone（多邊形）", img, SPEED_PATH, need_exact=3).run()
    RoiEditor("2. 拖曳 Table-Zone（四邊形）",  img, TABLE_PATH, need_exact=4).run()

    print("ROI 設定完成！")

if __name__ == "__main__":
    main()
