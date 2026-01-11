import pandas as pd
import matplotlib.pyplot as plt

video_name = 'C0'
path = f"runs/e/C0_predict/C0_predict"
csv_path = f"runs/e/C0_predict/C0_right_strokes_segments_with_median.csv"


x_col = "stroke_id"
columns_to_plot = ["max_speed", "median_speed"]


def get_df_plot(df, x_col, y_col):
    """保留 stroke_id 是數字的列，排除 mean_xxx 列"""
    df2 = df.copy()
    df2["stroke_numeric"] = pd.to_numeric(df2[x_col], errors="coerce")
    df_plot = df2[df2["stroke_numeric"].notna()].copy()
    df_plot[x_col] = df_plot["stroke_numeric"].astype(int)
    return df_plot[[x_col, y_col]]


def read_mean_from_csv(df, y_col):
    """直接從 CSV 讀取 mean 值（不要自己算）"""
    if y_col == "max_speed":
        key = "mean_max_speed"
    else:
        key = "mean_median_speed"

    row = df[df["stroke_id"] == key]

    if row.empty:
        print(f"⚠️ CSV 中找不到 {key}，mean 設為 NaN")
        return float("nan")

    return float(row.iloc[0][y_col])


def plot_speed(df, x_col, y_col, video_name):
    df_plot = get_df_plot(df, x_col, y_col)
    mean_value = read_mean_from_csv(df, y_col)

    plt.figure(figsize=(12, 5))

    plt.plot(df_plot[x_col], df_plot[y_col], marker='o')

    # y 軸從 0 開始，刻度列全部值
    plt.ylim(bottom=0)
    plt.xticks(df_plot[x_col])

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} over {x_col}")
    plt.grid(True)

    # 右側顯示平均值
    text_str = f"{video_name} Mean {y_col}: {mean_value:.2f}"
    plt.text(
        1.02, 0.5, text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )

    # 每個點標數字
    for x, y in zip(df_plot[x_col], df_plot[y_col]):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)

    # 輸出檔案
    output_png = f"{path}{video_name}_{y_col}_{x_col}.png"
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"已儲存折線圖為：{output_png}")


# ============================
# 主程式
# ============================
df = pd.read_csv(csv_path, encoding="utf-8")

for y_col in columns_to_plot:
    if y_col in df.columns:
        plot_speed(df, x_col, y_col, video_name)
    else:
        print(f"⚠️ 欄位 {y_col} 不存在於 CSV，略過。")
