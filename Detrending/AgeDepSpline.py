import os
import pandas as pd

# 设置路径
input_dir = r"D:/Detrending/Detrending4"
output_file = r"D:/更长时间的标准年表/AgeDepSpline.csv"

# 创建目标年份序列
year_range = pd.Series(range(1700, 2026), name="year")
result_df = pd.DataFrame(year_range)

# 遍历CSV文件
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(filepath)
            if "year" in df.columns and "AgeDepSpline" in df.columns:
                # 提取目标列并过滤年份
                df = df[["year", "AgeDepSpline"]]
                df = df[df["year"].between(1700, 2025)]

                # 检查有效值个数
                if df["AgeDepSpline"].notna().sum() >= 20:
                    colname = os.path.splitext(filename)[0]  # 去扩展名作为列名
                    df = df.rename(columns={"AgeDepSpline": colname})
                    result_df = pd.merge(result_df, df, on="year", how="left")
                else:
                    print(f"⚠️ 文件 {filename} 的有效值不足 20 个，已跳过。")
        except Exception as e:
            print(f"❌ 处理文件 {filename} 出错: {e}")

# 保存结果
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print("✅ 数据已保存到:", output_file)
