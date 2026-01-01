'''
TrackNet Labeling Tool (frame_num, visible, x, y)

Usage:
    python label_tool.py <video.mp4> --csv_dir <dir>

Controls:
    n / p   : next / previous frame
    f / b   : skip forward / backward 36 frames
    z / x   : jump to first / last frame
    s / e   : mark segment start / end (save clip)
    = / -   : increase / decrease circle size
    q       : quit and save CSV

Mouse:
    Left click        : mark ball (red dot)
    Right double-click: remove ball

CSV format:
    Frame,Visible,X,Y
    0,1,123,456
    1,0,0,0

'''



from pathlib import Path
import cv2
import pandas as pd
import os
import argparse
import sys

# state 0:hidden  1:Visible
state_name = ['HIDDEN', 'Visible']

keybindings = {
    'next':          [ord('n')],
    'prev':          [ord('p')],
    'piece_start':   [ord('s')],
    'piece_end':     [ord('e')],
    'first_frame':   [ord('z')],
    'last_frame':    [ord('X')],
    'forward_frames':[ord('f')],
    'backward_frames':[ord('b')],
    'circle_grow':   [ord('='), ord('+')],
    'circle_shrink': [ord('-')],
    'quit':          [ord('q')],
}

class VideoPlayer():
    def __init__(self, opt) -> None:
        self.opt = opt
        self.jump = 36

        self.cap = cv2.VideoCapture(opt.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {opt.video_path}")

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_path = Path(opt.video_path)
        self.circle_size = 5

        # TrackNet CSV 路徑
        self.csv_path = Path(opt.csv_dir) / (self.video_path.stem + ".csv") \
                        if opt.csv_dir else self.video_path.with_suffix(".csv")

        if not self.csv_path.parent.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.window = cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', 1280, 720)

        _, self.frame = self.cap.read()
        self.Frame = 0
        self.piece_start = 0
        self.piece_end = 0

        # 嘗試讀 TrackNet CSV
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            print(f"[load] TrackNet 格式: {self.csv_path}")

            self.info = {'Frame': [], 'Visible': [], 'X': [], 'Y': []}
            for idx in range(self.frames):
                self.info['Frame'].append(idx)
                self.info['Visible'].append(0)
                self.info['X'].append(0)
                self.info['Y'].append(0)

            for _, row in df.iterrows():
                fn = int(row['Frame'])
                if 0 <= fn < self.frames:
                    if int(row['Visible']) == 1:
                        self.info['Visible'][fn] = 1
                        self.info['X'][fn] = int(row['X'])
                        self.info['Y'][fn] = int(row['Y'])
        else:
            print("[init] 建立新 TrackNet CSV 標註")
            self.info = {'Frame': [], 'Visible': [], 'X': [], 'Y': []}
            for idx in range(self.frames):
                self.info['Frame'].append(idx)
                self.info['Visible'].append(0)
                self.info['X'].append(0)
                self.info['Y'].append(0)

        cv2.setMouseCallback('Frame', self.markBall)
        self.display()

    def save_piece(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.piece_start)
        out = cv2.VideoWriter(f'{self.piece_start+1}_{self.piece_end+1}.mp4',
                              self.fourcc, self.fps, (self.width, self.height))
        frame_cnt = self.piece_start
        while frame_cnt <= self.piece_end:
            ret, frame = self.cap.read()
            out.write(frame)
            frame_cnt += 1
        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.Frame)
        print("save piece successfully!")

    def markBall(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            win_w = cv2.getWindowImageRect('Frame')[2]
            win_h = cv2.getWindowImageRect('Frame')[3]
            vx = int(x * self.width / win_w)
            vy = int(y * self.height / win_h)
            self.info['Frame'][self.Frame] = self.Frame
            self.info['X'][self.Frame] = vx
            self.info['Y'][self.Frame] = vy
            self.info['Visible'][self.Frame] = 1
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            self.info['Frame'][self.Frame] = self.Frame
            self.info['X'][self.Frame] = 0
            self.info['Y'][self.Frame] = 0
            self.info['Visible'][self.Frame] = 0

    def display(self):
        res_frame = self.frame.copy()
        res_frame = cv2.putText(res_frame, state_name[self.info['Visible'][self.Frame]],
                                (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        res_frame = cv2.putText(res_frame, f"Frame: {int(self.Frame)}/{int(self.frames)}",
                                (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        if self.info['Visible'][self.Frame]:
            x = int(self.info['X'][self.Frame])
            y = int(self.info['Y'][self.Frame])
            cv2.circle(res_frame, (x, y), max(1, self.circle_size), (0, 0, 255), -1)

        cv2.imshow('Frame', res_frame)

    def main_loop(self):
        key = cv2.waitKeyEx(1)
        if key in keybindings['first_frame']:
            self.Frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.Frame)
            ret, self.frame = self.cap.read()
            assert ret
        elif key in keybindings['last_frame']:
            self.Frame = self.frames - 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.Frame)
            ret, self.frame = self.cap.read()
            assert ret
        elif key in keybindings['next']:
            if self.Frame < self.frames - 1:
                ret, self.frame = self.cap.read()
                self.Frame += 1
                assert ret
        elif key in keybindings['prev']:
            if self.Frame > 0:
                self.Frame -= 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.Frame)
                _, self.frame = self.cap.read()
        elif key in keybindings['forward_frames']:
            if self.Frame < self.frames - 1:
                for _ in range(self.jump):
                    if self.Frame >= self.frames - 2:
                        break
                    self.cap.grab()
                    self.Frame += 1
                ret, self.frame = self.cap.read()
                self.Frame += 1
        elif key in keybindings['backward_frames']:
            if self.Frame < self.jump:
                self.Frame = 0
            else:
                self.Frame -= self.jump
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.Frame)
            ret, self.frame = self.cap.read()
            assert ret
        elif key in keybindings['circle_grow']:
            self.circle_size += 1
        elif key in keybindings['circle_shrink']:
            self.circle_size = max(1, self.circle_size - 1)
        elif key in keybindings['piece_start']:
            self.piece_start = self.Frame
        elif key in keybindings['piece_end']:
            self.piece_end = self.Frame
            self.save_piece()
        elif key in keybindings['quit']:
            self.finish()
            return
        self.display()

    def finish(self):
        self.cap.release()
        cv2.destroyAllWindows()
        df = pd.DataFrame.from_dict(self.info).sort_values(by=['Frame'], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
        print(f"[save] TrackNet 格式: {self.csv_path}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, nargs='?', default=None, help='Path to the video file.')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='Path to the directory where csv file should be saved.')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    if opt.video_path is None:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)
        p = Path(application_path)
        video_path = next(p.glob('*.mp4'))
        opt.video_path = str(video_path)

    player = VideoPlayer(opt)
    while player.cap.isOpened():
        player.main_loop()
