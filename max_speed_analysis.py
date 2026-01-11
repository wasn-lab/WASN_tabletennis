'''
python 1121_4\max_speed_analysis.py --csv 1121_4\out\member_47\C0027_predict_speed.csv
'''

import argparse
import os
import pandas as pd


def extract_right_strokes(df, max_gap=15, min_dx=1e-6):
    """
    從已經篩過 in_zone & valid_speed 的資料中，
    找出「往右擊球」的每次 max_speed。

    參數：
        df      : 已篩過的 DataFrame，需包含 frame_num, x, speed_km_hr
        max_gap : 同一次擊球允許的最大 frame 斷層（in_zone 可能短暫斷掉）
        min_dx  : 判斷往右的最小 x 位移（避免 dx=0 的噪音）

    回傳：
        result_df: 欄位 [start_frame, end_frame, stroke_id, max_speed]
    """
    if df.empty:
        return pd.DataFrame(columns=["start_frame", "end_frame", "stroke_id", "max_speed"])

    # 依 frame 排序
    df = df.sort_values("frame_num").reset_index(drop=True)
    print(df)

    # 先用 frame 斷層切成 segments（候選的單次擊球）
    segments = []
    current_indices = [0]

    for i in range(1, len(df)):
        gap = df.loc[i, "frame_num"] - df.loc[i - 1, "frame_num"]
        if gap <= max_gap:
            # 還視為同一次擊球
            current_indices.append(i)
        else:
            # 開始新的擊球
            segments.append(df.loc[current_indices].copy())
            current_indices = [i]

    # 最後一段也要加進來
    segments.append(df.loc[current_indices].copy())
    print(segments)

    # 分析每個 segment 的方向，只留下往右的
    rows = []
    stroke_id = 1

    for seg in segments:
        # 先保險一下按 frame 排序
        seg = seg.reset_index(drop=True)
        if len(seg) == 0:
            continue
        print(seg)

        # 從第一個 x 開始，一路只保留「不往左」的 prefix
        last_x = seg.loc[0, "x"]
        end_idx = 0  # 最後一個「還在往右或持平」的 index

        for i in range(1, len(seg)):
            cur_x = seg.loc[i, "x"]
            # 還在往右（或持平）=> 繼續延長這次擊球
            if cur_x >= last_x:
                end_idx = i
                last_x = cur_x
            else:
                # 一出現往左（cur_x < last_x），後面全部丟掉
                break
            
        prefix = seg.iloc[:]
        # 只取「從第一個到最後一個還在增加/持平」的那一段
        if end_idx != 0:
            prefix = seg.iloc[: end_idx + 1]

        # 如果只有一個點，或是實際位移太小，就略過
        if len(prefix) < 2:
            start_frame = int(prefix["frame_num"].min())
            end_frame = int(prefix["frame_num"].max())
            max_speed = float(prefix["speed_km_hr"].max())

            rows.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "stroke_id": stroke_id,
                    "max_speed": max_speed,
                }
            )
            stroke_id += 1
        else:
            x_first = prefix.iloc[0]["x"]
            x_last = prefix.iloc[-1]["x"]
            dx = x_last - x_first  # >0 視為向右

            # 只保留「整段淨位移往右」的球（回擊）
            if dx > min_dx:
                start_frame = int(prefix["frame_num"].min())
                end_frame = int(prefix["frame_num"].max())
                max_speed = float(prefix["speed_km_hr"].max())

                rows.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "stroke_id": stroke_id,
                        "max_speed": max_speed,
                    }
                )
                stroke_id += 1

    result_df = pd.DataFrame(rows, columns=["start_frame", "end_frame", "stroke_id", "max_speed"])
    print(result_df)
    return result_df



def main():
    parser = argparse.ArgumentParser(
        description="從 C0027_predict.csv 中抽出每次向右擊球的最大球速"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="輸入的預測檔路徑，例如 C0027_predict.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出的 CSV 檔名（預設為 <輸入檔名去掉副檔名>_right_direction_max_speeds.csv）",
    )
    parser.add_argument(
        "--max_gap",
        type=int,
        default=15,
        help="同一次擊球允許的最大 frame 斷層（預設 15，可依影片 FPS 調整）",
    )

    args = parser.parse_args()

    input_path = args.csv
    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_right_direction_max_speeds.csv"
    else:
        output_path = args.output

    # 讀檔
    df = pd.read_csv(input_path)

    # 基本欄位檢查
    required_cols = {"frame_num", "x", "in_zone", "speed_km_hr", "valid_speed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少欄位：{missing}，請確認 CSV 格式是否正確。")

    # 只保留 in_zone 且 valid_speed 的點
    df_valid = df[(df["in_zone"] == True) & (df["valid_speed"] == True)].copy()

    # 抽出向右擊球的 max_speed
    strokes_df = extract_right_strokes(df_valid, max_gap=args.max_gap)

    # 如果沒有找到任何 stroke，直接輸出空檔案
    if strokes_df.empty:
        print("沒有找到任何符合條件的向右擊球。")
        strokes_df.to_csv(output_path, index=False)
        print(f"已輸出空結果到: {output_path}")
        return

    # 四捨五入 max_speed（可依需求調整）
    strokes_df["max_speed"] = strokes_df["max_speed"].round(2)

    # 計算 mean max_speed 並加在最後一列
    mean_speed = strokes_df["max_speed"].mean()
    mean_speed = round(mean_speed, 2)

    mean_row = {
        "start_frame": None,
        "end_frame": None,
        "stroke_id": "mean_max_speed",
        "max_speed": mean_speed,
    }

    output_df = pd.concat([strokes_df, pd.DataFrame([mean_row])], ignore_index=True)

    # 存成 CSV
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已輸出結果到: {output_path}")
    print(output_df)


if __name__ == "__main__":
    main()
