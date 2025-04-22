import pandas as pd
import numpy as np
import os
import scipy.stats as stats

# 設置pandas顯示選項以確保所有內容都被顯示
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


## 開始進行樣本分佈分析
# 1. 匯入 "Aggr 2014-2023_Old.csv" 檔案，移除具有空值的列
df = pd.read_csv("Aggr 2014-2023_Old.csv")
df.dropna(inplace=True)

# 取得總樣本數
total_samples = len(df)
print(f"全部樣本：{total_samples}")

# 2. 參照 "Country.csv" 檔案以統計 country 的分布
countries_df = pd.read_csv("Country.csv")

# 將國家名稱轉換為首字大寫格式（title case）
countries_df['Country name'] = countries_df['Country name'].apply(lambda x: x.title())

# 建立國家代碼與國家名稱的對照表
country_mapping = dict(zip(countries_df['Country'], countries_df['Country name']))

# 將數字代碼轉換為國家名稱
df['country_name'] = df['country'].map(country_mapping)
country_counts = df['country_name'].value_counts()

print("\n各國家分佈：")
for country, count in country_counts.items():
    print(f"{country}: {count}")

# 3. 參照 "ICB code.csv" 檔案以統計 industry 的分布
icb_df = pd.read_csv("ICB code.csv")

# 修正產業名稱中的拼寫錯誤
industry_corrections = {
    'Induatrial Goods and Services': 'Industrial Goods and Services',
    'Techonology': 'Technology',
    'Reatail': 'Retail',
    'Persocal Care, Drug and Grocery Stores': 'Personal Care, Drug and Grocery Stores'
}

# 建立 ICB 代碼與產業名稱的對照表
icb_df['Industry'] = icb_df['Industry'].replace(industry_corrections)
industry_mapping = dict(zip(icb_df['ICB code'], icb_df['Industry']))

# 將數字代碼轉換為產業名稱
df['industry_name'] = df['icbcode'].map(industry_mapping)
industry_counts = df['industry_name'].value_counts()

print("\n各產業分佈：")
for industry, count in industry_counts.items():
    print(f"{industry}: {count}")

# 4. 統計 year 年份數據
year_counts = df['year'].value_counts().sort_index()

print("\n各年份分佈：")
for year, count in year_counts.items():
    print(f"{year}: {count}")

# 生成完整的分佈表格
print("\n表格 1：樣本分佈")
print(f"全部樣本：{total_samples}")

print("\n各國家：")
for country, count in country_counts.items():
    print(f"{country}: {count}")

print("\n各產業：")
for industry, count in industry_counts.items():
    print(f"{industry}: {count}")

print("\n各年份：")
for year, count in year_counts.items():
    print(f"{year}: {count}")

# 將結果存儲為 CSV 檔案
results = {
    "Category": ["Full sample"] + 
                ["Across countries"] +
                list(country_counts.index) + 
                ["Across industries"] + 
                list(industry_counts.index) + 
                ["Across years"] + 
                list(year_counts.index),
    "Obs.": [total_samples] + 
           [np.nan] +
           list(country_counts.values) + 
           [np.nan] + 
           list(industry_counts.values) + 
           [np.nan] + 
           list(year_counts.values)
}

results_df = pd.DataFrame(results)
results_df.to_csv("sample_distribution_results.csv", index=False)
print("\n結果已保存至 'sample_distribution_results.csv'")


## 開始進行敘述性統計分析
# 指定需要分析的變數 (實際欄位名稱 -> 顯示名稱)
variables = {
    'gap': 'GAP',
    'family': 'Family',
    'gov': 'Gov',
    'g': 'G',
    'size': 'Size',
    'lev': 'Lev', 
    'roa': 'ROA',
    'mtb': 'MTB',
    'kz': 'KZ',
    'legal': 'Legal'
}

# 檢查是否有缺失欄位
print("\n資料集欄位列表：")
for col in sorted(df.columns.tolist()):
    print(f"- {col}")

# 建立結果資料框
stats_results = []

# 計算每個變數的敘述性統計量
for col, display_name in variables.items():
    if col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        median_val = df[col].median()
        max_val = df[col].max()
        
        stats_results.append({
            'Variables': display_name,
            'Mean': mean_val,
            'SD': std_val,
            'Min': min_val,
            'Median': median_val,
            'Max': max_val
        })
        print(f"已處理變數：{col} -> {display_name}")
    else:
        print(f"警告：找不到欄位 '{col}'")

# 將結果轉換為 DataFrame
stats_df = pd.DataFrame(stats_results)

# 顯示結果
print("\n表格 3：敘述性統計量")
pd.options.display.float_format = '{:.3f}'.format
print(stats_df)

# 儲存為 CSV 檔案
stats_df.to_csv("summary_statistics_results.csv", index=False, float_format='%.3f')
print("\n敘述性統計結果已保存至 'summary_statistics_results.csv'")


## 開始進行相關係數分析
# 指定用於相關係數分析的變數列表（與敘述性統計分析相同）
corr_variables = variables

# 檢查欄位是否存在
missing_cols = [col for col in corr_variables.keys() if col not in df.columns]
if missing_cols:
    print(f"\n警告：以下欄位在資料集中不存在：{missing_cols}")
    # 移除不存在的欄位
    for col in missing_cols:
        del corr_variables[col]

# 創建一個新的 DataFrame 只包含我們需要的變數
corr_df = df[[col for col in corr_variables.keys()]].copy()

# 為了方便後續處理，將欄位重命名為顯示名稱
corr_df.rename(columns=corr_variables, inplace=True)

# 計算相關係數矩陣
correlation_matrix = corr_df.corr(method='pearson')

# 定義一個函數來為相關係數添加顯著性星號
def add_significance_stars(corr_df, p_value_df):
    result_df = corr_df.copy()
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            if i != j:  # 不標記對角線元素
                corr_value = corr_df.iloc[i, j]
                p_value = p_value_df.iloc[i, j]
                stars = ''
                if p_value < 0.01:
                    stars = '***'
                elif p_value < 0.05:
                    stars = '**'
                elif p_value < 0.1:
                    stars = '*'
                
                if stars:
                    result_df.iloc[i, j] = f"{corr_value:.3f}{stars}"
                else:
                    result_df.iloc[i, j] = f"{corr_value:.3f}"
            else:
                result_df.iloc[i, j] = f"{corr_df.iloc[i, j]:.3f}"
    
    return result_df

# 計算 p value 矩陣
p_values = pd.DataFrame(np.zeros_like(correlation_matrix), index=correlation_matrix.index, columns=correlation_matrix.columns)

# 計算所有變數對之間的 p value
n = len(corr_df)
for i, var1 in enumerate(correlation_matrix.index):
    for j, var2 in enumerate(correlation_matrix.columns):
        if i != j:  # 跳過對角線
            r = correlation_matrix.loc[var1, var2]
            # 使用 t 分布計算 p value
            t = r * np.sqrt((n - 2) / (1 - r**2))
            p = 2 * (1 - stats.t.cdf(abs(t), n - 2))
            p_values.loc[var1, var2] = p

# 為相關係數添加顯著性星號
styled_corr_matrix = add_significance_stars(correlation_matrix, p_values)

# 顯示結果
print("\n表格 4：相關係數矩陣")
print(styled_corr_matrix)

# 創建一個只包含下三角部分的矩陣
lower_triangle = styled_corr_matrix.copy()
for i in range(len(lower_triangle.index)):
    for j in range(len(lower_triangle.columns)):
        if j > i:  # 只清除右上三角 (不包含對角線)
            lower_triangle.iloc[i, j] = np.nan

# 將結果存儲為 CSV 檔案
lower_triangle.to_csv("correlation_matrix_results.csv")
print("\n相關係數矩陣已保存至 'correlation_matrix_results.csv'（僅包含對角線及左下三角部分）")