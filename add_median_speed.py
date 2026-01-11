# python ver4\add_median_speed.py 048\C0045_right_strokes_segments.csv 048\C0045_predict_speed.csv
import pandas as pd
import numpy as np
import sys
import os


def is_number(x):
    """判斷 stroke_id 是否為數字（true 代表正常 stroke）"""
    try:
        float(x)
        return True
    except:
        return False


def main(max_csv_path, all_frames_csv_path, output_path=None):
    # === 讀取資料 ===
    df_max = pd.read_csv(max_csv_path, encoding="utf-8")
    df_all = pd.read_csv(all_frames_csv_path, encoding="utf-8")

    # === 設定輸出檔名 ===
    if output_path is None:
        base, ext = os.path.splitext(max_csv_path)
        output_path = base + "_with_median" + ext

    # === 若不存在 median_speed，新增在 max_speed 後面 ===
    if "median_speed" not in df_max.columns:
        cols = list(df_max.columns)
        if "max_speed" in cols:
            insert_idx = cols.index("max_speed") + 1
        else:
            insert_idx = len(cols)
        cols.insert(insert_idx, "median_speed")
        df_max = df_max.reindex(columns=cols)

    # 先填 NaN
    df_max["median_speed"] = np.nan

    # === 選出真正的 stroke（數字） ===
    stroke_mask = df_max["stroke_id"].apply(is_number)

    # === 計算每個 stroke 的 median_speed ===
    for idx, row in df_max[stroke_mask].iterrows():
        sf = row["start_frame"]
        ef = row["end_frame"]

        # 沒有範圍 → 保持空白
        if not (pd.notna(sf) and pd.notna(ef)):
            continue

        sf_i, ef_i = int(sf), int(ef)

        # 取出範圍內的所有 frame
        mask = (df_all["frame_num"] >= sf_i) & (df_all["frame_num"] <= ef_i)

        # 只取 speed_km_hr != 0 的速度（0 要跳過）
        speeds = df_all.loc[mask & (df_all["speed_km_hr"] != 0), "speed_km_hr"].dropna()

        if len(speeds) == 0:
            # 沒有任何非 0 速度 → 空白
            median_val = np.nan
        elif len(speeds) == 1:
            # 只有一個 frame → 就是那個速度
            median_val = float(speeds.iloc[0])
        else:
            # 多個 frame → 取中位數
            median_val = float(np.median(speeds))

        df_max.at[idx, "median_speed"] = median_val

    # === 計算 mean_median_speed（只算有值的 stroke） ===
    mean_median = df_max.loc[stroke_mask, "median_speed"].dropna().mean()

    # === 找到 mean_max_speed 那一列（非數字 stroke_id） ===
    summary_mask = ~stroke_mask
    summary_rows = df_max[summary_mask]

    # 新 row：所有欄位要齊，保持原本（含中文）欄位順序
    new_row = {col: np.nan for col in df_max.columns}
    new_row["stroke_id"] = "mean_median_speed"
    new_row["median_speed"] = mean_median

    if len(summary_rows) == 0:
        # 沒 summary → 直接加在最後一列
        df_max = pd.concat([df_max, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # 插在最後一個 summary（通常是 mean_max_speed）下面
        summary_idx = summary_rows.index[-1]
        top = df_max.iloc[: summary_idx + 1]
        bottom = df_max.iloc[summary_idx + 1 :]
        df_max = pd.concat([top, pd.DataFrame([new_row]), bottom], ignore_index=True)

    # === 輸出 ===
    df_max.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已輸出：{output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法：python add_median_speed_v3.py <max_speeds_csv> <all_frames_csv> [output_csv]")
        print("範例：python add_median_speed_v3.py "
              "C0091_predict_speed_right_direction_max_speeds.csv "
              "C0091_predict_speed.csv")
        sys.exit(1)

    max_csv = sys.argv[1]
    all_csv = sys.argv[2]
    out_csv = sys.argv[3] if len(sys.argv) >= 4 else None

    main(max_csv, all_csv, out_csv)
