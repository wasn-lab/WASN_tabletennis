import numpy as np
import cv2
import os
import glob
from filterpy.kalman import KalmanFilter
from math import sqrt
import time
from functools import wraps

video_name = 'C0012'
# ---------- 參數區 ----------
DETECT_TXT_DIR = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\runs\detect\{video_name}_predict_speed.csv'
VIDEO_PATH     = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\data\{video_name}.mp4'
OUT_VIDEO      = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\runs\detect\{video_name}_predict\kalman_{video_name}.avi'
TXT_OUT        = fr'D:\BoTai\Workspace\TrackNetV2-pytorch-main\runs\detect\{video_name}_predict\kalman_ball.csv'

IMG_H, IMG_W   = 720, 1280   # 影片尺寸，不對就改掉
MAX_MISS       = 3           # 最大容忍失蹤
GATE_THRES     = 9.21        # 對應 Chi-square 0.95, 自由度=2 (x,y)
# ----------------------------

def time_stamp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[⏱️ START] {func.__name__}()")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[✅ DONE] {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper

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

@time_stamp
def main():
    raw_det = read_detect_frame(DETECT_TXT_DIR)
    cap  = cv2.VideoCapture(VIDEO_PATH)
    fps  = int(cap.get(cv2.CAP_PROP_FPS))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
    f = open(TXT_OUT, 'w')

    kf = build_kalman()
    initiated = False
    miss_count = 0
    traj = []

    frame_id = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_id += 1

        has_obs = frame_id in raw_det
        z = np.array(raw_det[frame_id]) if has_obs else None

        if not initiated:
            if has_obs:
                # 初始化
                kf.x[:2, 0] = z
                initiated = True
                miss_count = 0
            else:
                # 還沒開始就沒球，繼續空轉
                writer.write(img)
                continue
        else:
            # 先預測
            kf.predict()
            if has_obs:
                # Gate test
                y = z - (kf.H @ kf.x)[:,0]
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
                traj.clear()
                writer.write(img)
                continue

        # 把結果寫出
        x_est, y_est = kf.x[0,0], kf.x[1,0]
        traj.append((int(x_est), int(y_est)))
        f.write(f'{frame_id},{x_est:.2f},{y_est:.2f}\n')

        # 可視化
        if has_obs:
            cv2.circle(img, (int(z[0]), int(z[1])), 4, (0,0,255), -1)  # 原始觀測 紅
        cv2.circle(img, (int(x_est), int(y_est)), 5, (0,255,0), -1)      # 濾波後 綠
        for i in range(1, len(traj)):
            cv2.line(img, traj[i-1], traj[i], (0,255,255), 2)           # 軌跡線
        writer.write(img)

    cap.release()
    writer.release()
    f.close()
    print('kalman 軌跡完成:', OUT_VIDEO)


if __name__ == '__main__':
    main()
