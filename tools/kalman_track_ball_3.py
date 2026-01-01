import numpy as np
import cv2
import os
import glob
from filterpy.kalman import KalmanFilter
from math import sqrt
import time
from functools import wraps

video_name = 'C0086'
# ---------- 參數區 ----------
DETECT_TXT_DIR = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\runs\detect\{video_name}_predict_speed.csv'
VIDEO_PATH     = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\data\{video_name}.mp4'
OUT_VIDEO      = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\runs\detect\{video_name}_predict\kalman_{video_name}.avi'
TXT_OUT        = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\runs\detect\{video_name}_predict\kalman_ball.csv'

IMG_H, IMG_W   = 720, 1280   # 影片尺寸，不對就改掉
MAX_MISS       = 3           # 最大容忍失蹤
GATE_THRES     = 9.21        # 對應 Chi-square 0.95, 自由度=2 (x,y)
FPS =120
PX2M   = 0.2614 * 0.01
li=[]
# ----------------------------

def time_stamp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(time.asctime( time.localtime(time.time()) ))
        print(f"[⏱️ START] {func.__name__}()")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[✅ DONE] {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper

@time_stamp
def build_kalman():
    kf = KalmanFilter(dim_x=6, dim_z=2)
    dt = 1/120.  # 120 fps
    # 狀態轉移矩陣 (constant acceleration model)
    kf.F = np.array([
        [1,0,dt,0,0.5*dt**2,0],
        [0,1,0,dt,0,0.5*dt**2],
        [0,0,1,0,dt,0],
        [0,0,0,1,0,dt],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]])
    # 觀測矩陣
    kf.H = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0]])
    # 初始協方差
    kf.P *= 1000.
    # 過程雜訊
    q = 0.1
    kf.Q = np.eye(6)*q
    kf.Q[2:, 2:] *= 10.  # 速度/加速度雜訊放大
    # 觀測雜訊
    kf.R = np.eye(2)*5
    return kf

def read_detect_file(txt_dir):
    """
    回傳 dict {frame_id:  list_of_ (x1,y1,x2,y2)  }  # 多框
    支援 yolo-txt：cls cx cy w h   (cls=0 才要)
    """
    from pathlib import Path
    import numpy as np
    W, H = 1280, 720          # 影片寬高
    det = {}
    for p in Path(txt_dir).glob('*.txt'):
        fid = int(p.stem)     # 000123.txt → 123
        with open(p) as f:
            objs = []
            for ln in f:
                dat = ln.strip().split()
                if int(dat[0]) != 0:       # 只留類別 0 (ball)
                    continue
                cx,cy,w,h = map(float, dat[1:5])
                x1 = int((cx-w/2)*W)
                y1 = int((cy-h/2)*H)
                x2 = int((cx+w/2)*W)
                y2 = int((cy+h/2)*H)
                objs.append((x1,y1,x2,y2))
        if objs:          # 至少有一球才記錄
            det[fid] = objs
    return det

def read_detect_frame(csv_path):
    """回傳 dict {frame_id:int -> (x_pixel, y_pixel)}"""
    import pandas as pd
    raw = pd.read_csv(csv_path)
    detect = {}
    for _, row in raw.iterrows():
        fid = int(row['frame_num']) ##frame_num,visible,x,y,speed_km_hr
        x   = float(row['x'])
        y   = float(row['y'])
        # 若是 vis 欄位，可見才要
        if 'visible' in row and row['visible']==0:
            continue
        detect[fid] = (x, y)
    return detect


# ---------- 新增 ----------
SELECT_ROI = True   # 若 False 就回到全畫面

# --------------------------




def in_roi(px, py, roi_):
    """點 (px,py) 是否在 roi 矩形內"""
    return roi_[0] <= px <= roi_[2] and roi_[1] <= py <= roi_[3]

@time_stamp
def main():
    import csv
    raw_det = read_detect_frame(DETECT_TXT_DIR)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    # 0. 保證輸出目錄
    os.makedirs(os.path.dirname(TXT_OUT), exist_ok=True)

    # 1. 選 ROI（挪到最前面，先選完再開 CSV）
    # ---------- 1.  選 ROI（挪到最前面，先選完再開 CSV）----------

    roi = (0, 0, w, h)          # 預設全圖
    if SELECT_ROI:
        ret, first = cap.read()
        assert ret, "影片讀取失敗"
        clone = first.copy()
        cv2.namedWindow("select ROI", cv2.WINDOW_AUTOSIZE)
        refPt, cropping = [], False
        # ---- 關鍵：把 callback 定義在 main() 裡面，nonlocal 才合法 ----
        def click_and_drag(event, x, y, flags, param):
            nonlocal refPt, cropping, roi      # 現在 roi 真的在外層函式
            if event == cv2.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]
                cropping = True
            elif event == cv2.EVENT_MOUSEMOVE and cropping:
                tmp = first.copy()
                cv2.rectangle(tmp, refPt[0], (x, y), (255, 0, 255), 2)
                cv2.imshow("select ROI", tmp)
            elif event == cv2.EVENT_LBUTTONUP:
                refPt.append((x, y))
                cropping = False
                roi = (min(refPt[0][0], refPt[1][0]),
                       min(refPt[0][1], refPt[1][1]),
                       max(refPt[0][0], refPt[1][0]),
                       max(refPt[0][1], refPt[1][1]))
                print(f"[ROI] 選定範圍 {roi}")
                x1, y1, x2, y2 = roi
                roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                # 把最終框畫上
                cv2.rectangle(first, refPt[0], refPt[1], (0, 255, 0), 2)
                cv2.imshow("select ROI", first)
        cv2.setMouseCallback("select ROI", click_and_drag)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):          # Reset
                first = clone.copy()
                refPt, cropping = [], False
            elif key in (13, 32):        # Enter / Space
                break
        cv2.destroyWindow("select ROI")
        # =====  新增：彈出 ROI 預覽視窗  =====
        if roi != (0, 0, w, h):                 # 有用戶真正框選才秀
            x1, y1, x2, y2 = roi
            roi_crop = clone[y1:y2, x1:x2]      # 用原始畫面裁剪，避免畫完框的干擾
            cv2.namedWindow("ROI 預覽", cv2.WINDOW_NORMAL)
            cv2.imshow("ROI 預覽", roi_crop)
            cv2.waitKey(0)                      # 任意鍵關閉
            cv2.destroyWindow("ROI 預覽")
        # =====================================
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # 倒帶
    # ----------------------------------------

    # 2. 主迴圈 + 保證寫 CSV
    kf   = build_kalman()
    initiated = False
    miss_count = 0
    traj = []

    trace_buffer = []  # 存每點 (x,y) 用來算 total_displacement
    start_pt = None    # 軌跡起點
    with open(TXT_OUT, 'w', newline='') as f:
        csv_wr = csv.writer(f)
        # 1. 新表頭
        csv_wr.writerow(['frame', 'visible', 'x', 'y',
                         'right_speed_px_frame', 'total_displacement'])
        frame_id = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            frame_id += 1
            # ---------- 觀測點過濾 ROI ----------
            has_obs = frame_id in raw_det
            z = None
            if has_obs:
                zx, zy = raw_det[frame_id]
                if not in_roi(zx, zy, roi):
                    has_obs = False
                else:
                    z = np.array([zx, zy])
            # ---------- Kalman 主流程 ----------
            if not initiated:
                if has_obs:
                    kf.x[:2, 0] = z
                    initiated = True
                    miss_count = 0
                    # 新軌跡開始
                    start_pt = (float(z[0]), float(z[1]))
                    trace_buffer = [start_pt]
                else:
                    csv_wr.writerow([frame_id, 0, -1, -1, -1, -1])
                    writer.write(img)
                    continue
            else:
                kf.predict()
                if has_obs:
                    y = z - (kf.H @ kf.x)[:, 0]
                    S = kf.H @ kf.P @ kf.H.T + kf.R
                    d2 = float(y.T @ np.linalg.inv(S) @ y)
                    if d2 < GATE_THRES:
                        kf.update(z)
                        miss_count = 0
                    else:
                        miss_count += 1
                else:
                    miss_count += 1
                if miss_count > MAX_MISS:
                    initiated = False
                    csv_wr.writerow([frame_id, 0, -1, -1, -1, -1])
                    trace_buffer.clear()
                    writer.write(img)
                    continue
            x_est, y_est = kf.x[0, 0], kf.x[1, 0]
            visible = 1 if in_roi(x_est, y_est, roi) else 0
            # ----- 計算新欄位 -----
            # 1. 向右速度 (px/frame)  → kf.x[2] 是 x 方向速度
            vx = kf.x[2, 0]
            # 2. 總位移
            if start_pt is not None:
                total_d = sqrt((x_est - start_pt[0])**2 +
                               (y_est - start_pt[1])**2)
            else:
                total_d = 0.
            # 3. 更新 trace_buffer（可見才存，方便畫線）
            if visible:
                trace_buffer.append((float(x_est), float(y_est)))
                # 超過 240 點就丟掉舊的
                if len(trace_buffer) > int(2 * FPS):
                    trace_buffer.pop(0)
            # 4. 寫 CSV
            csv_wr.writerow([frame_id,
                             visible,
                             int(x_est) if visible else -1,
                             int(y_est) if visible else -1,
                             round(vx, 3) if visible else -1,
                             round(total_d, 2) if visible else -1])
            # ------------------ 視覺化（你原來的） ------------------
            if SELECT_ROI:
                cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]),
                              (255, 0, 255), 2)
            if has_obs:
                cv2.circle(img, (int(z[0]), int(z[1])), 4, (0, 0, 255), -1)
            # 畫軌跡
            if len(trace_buffer) > 1:
                pts = [(int(p[0]), int(p[1])) for p in trace_buffer]
                for i in range(1, len(pts)):
                    cv2.line(img, pts[i-1], pts[i], (0, 255, 255), 2)
            writer.write(img)
    cap.release()
    writer.release()
    print('[Info] 已輸出含「向右速度」與「總位移」之 CSV：', TXT_OUT)

if __name__ == '__main__':
    main()
