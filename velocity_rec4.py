# ==========================================================
#  新版 ROI 设定：Speed & Table 皆为 polygon JSON
# ==========================================================
import torch
import torchvision

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import json

from models.tracknet import TrackNet
from utils.general import get_shuttle_position

import time
from functools import wraps

MAX_JUMP_DISTANCE = 150
MIN_FRAME_GAP = 1
MAX_FRAME_GAP = 5
MIN_REASONABLE_SPEED = 5
MAX_REASONABLE_SPEED = 150

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

ZONE_PATH  = Path("speed_zone.json")     # 改为 JSON
TABLE_PATH = Path("table_zone.json")
JSON_VER = 1

def px_to_km(px):
    return 0.2777/100000*px

def time_stamp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[START] {func.__name__}()")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[DONE] {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper
def shoelace(pts: np.ndarray) -> float:
    return float(cv2.contourArea(pts))
def point_in_poly(pts, p):
    """射线法判断点是否在多边形内"""
    inside, j = False, len(pts) - 1
    for i, (xi, yi) in enumerate(pts):
        xj, yj = pts[j]
        if ((yi > p[1]) != (yj > p[1])) and \
           (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside
def draw_roi_on(img: np.ndarray, pts: np.ndarray, color, alpha=0.3):
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, color, 2)
    return img
class RoiEditor:
    def __init__(self, win_name, img, save_path, need_exact=3, min_area=1000):
        self.win_name   = win_name
        self.img        = img
        self.h, self.w  = img.shape[:2]
        self.save_path  = Path(save_path)
        self.need_exact = need_exact
        self.min_area   = min_area
        self.pts        = []
        self.drag_idx   = -1
        self._init_shape()
    def _init_shape(self):
        margin = int(min(self.w, self.h) * 0.2)
        if self.need_exact == 4:
            self.pts = [(margin, margin),
                        (self.w - margin, margin),
                        (self.w - margin, self.h - margin),
                        (margin, self.h - margin)]
        else:
            self.pts = [(margin, margin),
                        (self.w - margin, margin),
                        (self.w - margin, self.h - margin),
                        (margin, self.h - margin)]
    def redraw(self):
        tmp = self.img.copy()
        n = len(self.pts)
        if n >= 3:
            pts_arr = np.array(self.pts, np.int32)
            color = (0, 255, 0) if "Speed" in self.win_name else (255, 0, 0)
            tmp = draw_roi_on(tmp, pts_arr, color)
        for i, (x, y) in enumerate(self.pts):
            color = (0, 0, 255) if i == self.drag_idx else (255, 0, 0)
            cv2.circle(tmp, (int(x), int(y)), 6, color, -1)
        msg = f"點數:{n}  Enter=存檔  R=重設  Esc=退出"
        if n < self.need_exact:
            msg += f" (需≥{self.need_exact}點)"
        cv2.displayStatusBar(self.win_name, msg)
        cv2.imshow(self.win_name, tmp)
    def mouse_cb(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pts:
                dists = [((x - px) ** 2 + (y - py) ** 2) ** 0.5
                         for px, py in self.pts]
                if min(dists) < 15:
                    self.drag_idx = int(np.argmin(dists))
                    return
            if self.need_exact != 4:
                self.pts.append([x, y])
                self.drag_idx = -1
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx != -1:
            self.pts[self.drag_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1
        self.redraw()
    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.w // 2, self.h // 2)
        cv2.setMouseCallback(self.win_name, self.mouse_cb)
        self.redraw()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if self._validate():
                    self._save()
                    self._visualize_confirm()
                    break
            elif key == ord("r"):
                self._init_shape()
            elif key == 27:
                print("使用者中斷"); sys.exit(0)
        cv2.destroyWindow(self.win_name)
        return self.pts
    def _validate(self):
        n = len(self.pts)
        if n < self.need_exact:
            return False
        area = shoelace(np.array(self.pts, np.float32))
        if area < self.min_area:
            print(f"面積 {area:.0f} px² 過小 (門檻 {self.min_area})，拒絕存檔")
            return False
        return True
    def _save(self):
        data = {"version": JSON_VER,
                "type": "poly" if self.need_exact != 4 else "quad",
                "pts": self.pts}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("已儲存 →", self.save_path.resolve())
    def _visualize_confirm(self):
        pts_arr = np.array(self.pts, np.int32)
        color = (0, 255, 0) if "Speed" in self.win_name else (255, 0, 0)
        vis = draw_roi_on(self.img.copy(), pts_arr, color)
        cv2.imshow("ROI 確認（任意鍵關閉）", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# ---------- 主程式專用 API ----------
def setup_zones(first_frame, w, h):
    """讀檔 or 互動拉框；回傳 (speed_poly, table_poly) 兩個 ndarray"""
    if ZONE_PATH.exists() and TABLE_PATH.exists():
        with open(ZONE_PATH)  as f: speed_poly = np.array(json.load(f)["pts"], np.int32)
        with open(TABLE_PATH) as f: table_poly = np.array(json.load(f)["pts"], np.int32)
        return speed_poly, table_poly
    print("\n=== 拉取 ROI ===")
    RoiEditor("1. Speed-Zone（多邊形，≥3 點）", first_frame, ZONE_PATH,
              need_exact=3, min_area=5000).run()
    RoiEditor("2. Table-Zone（四邊形，4 點）",  first_frame, TABLE_PATH,
              need_exact=4, min_area=1000).run()
    with open(ZONE_PATH)  as f: speed_poly = np.array(json.load(f)["pts"], np.int32)
    with open(TABLE_PATH) as f: table_poly = np.array(json.load(f)["pts"], np.int32)
    return speed_poly, table_poly

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'example_dataset/match/videos/1_10_12.mp4', help='Path to video.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.csv')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512], help='image size h,w')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Path to trained model weights.')
    parser.add_argument('--project', default=ROOT / 'runs/velocity_rec2', help='save results to project/name')
    parser.add_argument('--center-ratio', type=float, default=0.6, help='Default center ratio')
    parser.add_argument('--max-jump', type=int, default=200, help='Max jump distance in pixels')
    parser.add_argument('--max-speed', type=float, default=150.0, help='Max reasonable speed in km/h')
    parser.add_argument('--min-speed', type=float, default=0.0, help='Min reasonable speed in km/h')
    parser.add_argument('--use-multi-point', action='store_true', help='Use multi-point fitting for speed')
    opt = parser.parse_args()
    return opt

@time_stamp
def main(opt):
    global MAX_JUMP_DISTANCE, MIN_REASONABLE_SPEED, MAX_REASONABLE_SPEED
    
    MAX_JUMP_DISTANCE = opt.max_jump
    MIN_REASONABLE_SPEED = opt.min_speed
    MAX_REASONABLE_SPEED = opt.max_speed
    
    source_name = os.path.splitext(os.path.basename(opt.source))[0]
    b_save_txt = opt.save_txt
    b_view_img = opt.view_img
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    f_source = str(opt.source)
    imgsz = opt.imgsz

    source_name = f'{source_name}_predict'

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)

    img_save_path = f'{d_save_dir}/{source_name}'
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    vid_cap = cv2.VideoCapture(f_source)
    if not vid_cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {f_source}")

    ret, first_frame = vid_cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame!")

    
    

    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---------- 读 Speed & Table polygon ----------
    speed_poly, table_poly = setup_zones(first_frame, w, h)

    print(f"\n=== Speed Calculation Config ===")
    print(f"Max Jump Distance: {MAX_JUMP_DISTANCE} px")
    print(f"Frame Gap Range: {MIN_FRAME_GAP} - {MAX_FRAME_GAP}")
    print(f"Speed Range: {MIN_REASONABLE_SPEED} - {MAX_REASONABLE_SPEED} km/h")
    print(f"Multi-point Fitting: {opt.use_multi_point}")
    print(f"================================\n")

    # confirmed = first_frame.copy()
    # cv2.rectangle(confirmed, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 100, 0), 3)
    # cv2.putText(confirmed, "CONFIRMED SPEED ZONE", (roi_x1, roi_y1 - 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
    # preview_path = f"{d_save_dir}/center_zone_confirmed.jpg"
    # cv2.imwrite(preview_path, confirmed)
    

    vid_cap.release()
    vid_cap = cv2.VideoCapture(f_source)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackNet().to(device)

    from collections import OrderedDict
    state_dict = torch.load(f_weights, map_location=device)

    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{d_save_dir}/{source_name}.mp4', fourcc, fps, (w, h))

    if b_save_txt:
        f_save_txt = open(f'{d_save_dir}/{source_name}.csv', 'w')
        f_save_txt.write('frame_num,visible,x,y,in_zone,speed_km_hr,valid_speed,on_table\n')

    if b_view_img:
        cv2.namedWindow(source_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_name, (w, h))

    count = 0
    positions = []
    speed = 0.0
    last_valid_speed = 0.0
    record_list = []
    
    stats = {
        'total_detections': 0,
        'jump_filtered': 0,
        'frame_gap_filtered': 0,
        'speed_filtered': 0,
        'valid_speeds': 0
    }

    while vid_cap.isOpened():
        imgs = []
        for _ in range(3):
            ret, img = vid_cap.read()
            if not ret:
                break
            imgs.append(img)

        if len(imgs) < 3:
            break

        imgs_torch = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_torch = torchvision.transforms.ToTensor()(img).to(device)
            img_torch = torchvision.transforms.functional.resize(img_torch, imgsz, antialias=True)
            imgs_torch.append(img_torch)

        imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

        preds = model(imgs_torch)
        preds = preds[0].detach().cpu().numpy()

        y_preds = (preds > 0.5).astype(np.uint8) * 255

        for i in range(3):
            visible, cx_pred, cy_pred = get_shuttle_position(y_preds[i])
            cx = int(cx_pred * w / imgsz[1])
            cy = int(cy_pred * h / imgsz[0])
            in_zone  = point_in_poly(speed_poly, (cx, cy))
            on_table = point_in_poly(table_poly, (cx, cy))

            valid_speed = False
            filter_reason = ""

            if visible:
                stats['total_detections'] += 1
                
                if len(positions) >= 1:
                    last_f, last_x, last_y = positions[-1]
                    jump_dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                    
                    if jump_dist > MAX_JUMP_DISTANCE:
                        positions.clear()
                        speed = 0.0
                        filter_reason = f"JUMP_FILTER({jump_dist:.0f}px)"
                        stats['jump_filtered'] += 1
                
                positions.append((count, cx, cy))
                if len(positions) > 5:
                    positions.pop(0)

                if len(positions) >= 2:
                    f0, x0, y0 = positions[-2]
                    f1, x1, y1 = positions[-1]
                    frame_gap = f1 - f0
                    
                    if frame_gap < MIN_FRAME_GAP or frame_gap > MAX_FRAME_GAP:
                        speed = 0.0
                        filter_reason = f"FRAME_GAP_FILTER({frame_gap})"
                        stats['frame_gap_filtered'] += 1
                    else:
                        if opt.use_multi_point and len(positions) >= 3:
                            speed = fit_velocity_multi_points(positions, fps, min_points=3)
                        else:
                            dist_inpx = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                            elapsed_time = (f1 - f0) / fps
                            dist_inkm = px_to_km(dist_inpx)
                            elapsed_hr = elapsed_time / 3600
                            speed = dist_inkm / elapsed_hr if elapsed_hr > 0 else 0.0
                        
                        if speed < MIN_REASONABLE_SPEED or speed > MAX_REASONABLE_SPEED:
                            filter_reason = f"SPEED_FILTER({speed:.1f})"
                            stats['speed_filtered'] += 1
                            speed = 0.0
                        else:
                            valid_speed = True
                            last_valid_speed = speed
                            stats['valid_speeds'] += 1
                else:
                    speed = 0.0

                cv2.circle(imgs[i], (cx, cy), 8, (0, 0, 255), -1)
            else:
                speed = 0.0

            

            relative_time = count / fps
            cv2.putText(imgs[i], f"Time: {relative_time:.2f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            speed_color = (0, 255, 0) if valid_speed else (128, 128, 128)
            cv2.putText(imgs[i], f"Speed: {speed:.2f} km/hr", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, speed_color, 2)
            # 画 Speed-Zone
            cv2.polylines(imgs[i], [speed_poly], True, (255, 100, 0), 2)
            # 画 Table-Zone
            cv2.polylines(imgs[i], [table_poly], True, (0, 255, 255), 2)
            
            if filter_reason:
                cv2.putText(imgs[i], filter_reason, (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if b_save_txt:
                f_save_txt.write(f"{count},{visible},{cx},{cy},{in_zone},{speed:.2f},{valid_speed},{on_table}\n")

            if b_view_img:
                cv2.imwrite(f"{img_save_path}/{count}.png", imgs[i])
                cv2.imshow(source_name, imgs[i])
                cv2.waitKey(1)

            out.write(imgs[i])
            
            status = f"VALID" if valid_speed else filter_reason if filter_reason else "NO_DATA"
            print(f"{count} ---- visible: {visible}  cx: {cx}  cy: {cy}  in_zone: {in_zone}  speed: {speed:.2f} km/hr  [{status}]")

            record_list.append({
                "frame_num": count,
                "visible": visible,
                "x": cx,
                "y": cy,
                "in_zone": in_zone,
                "speed_km_hr": speed,
                "valid_speed": valid_speed,
                'on_table' : on_table
            })

            count += 1

    if b_save_txt:
        while count < video_len:
            f_save_txt.write(f"{count},0,0,0,0,0.0,False,False\n")
            count += 1
        f_save_txt.close()

    df = pd.DataFrame(record_list)
    df.to_csv(f"{d_save_dir}/{source_name}_speed.csv", index=False)

    print(f"\n=== Speed Calculation Statistics ===")
    print(f"Total Detections: {stats['total_detections']}")
    print(f"Jump Filtered: {stats['jump_filtered']}")
    print(f"Frame Gap Filtered: {stats['frame_gap_filtered']}")
    print(f"Speed Range Filtered: {stats['speed_filtered']}")
    print(f"Valid Speeds: {stats['valid_speeds']}")
    if stats['total_detections'] > 0:
        valid_ratio = stats['valid_speeds'] / stats['total_detections'] * 100
        print(f"Valid Speed Ratio: {valid_ratio:.1f}%")
    print(f"====================================\n")

    out.release()
    vid_cap.release()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)