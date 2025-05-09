import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 設置pandas顯示選項以確保所有內容都被顯示
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


## 開始進行樣本分佈分析
# 1. 匯入檔案，移除具有空值的列
df = pd.read_csv("Aggr 2014-2023_All New.csv")
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
    'gap_e': 'GAP_E',
    'gap_s': 'GAP_S',
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

# 定義需要進行 winsorization 的變數
winsorize_vars = ['kz']

# 創建需要處理的變數副本
winsorized_df = df.copy()

# 定義 winsorization 的上下限百分比
lower_percentile = 0.01
upper_percentile = 0.99

# 只對指定變數進行 winsorization 處理
for col in winsorize_vars:
    if col in df.columns:
        # 計算上下限值
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)
        
        # 使用 scipy.stats 的 winsorize 函數進行處理
        winsorized_values = stats.mstats.winsorize(df[col], limits=[lower_percentile, 1-upper_percentile])
        
        # 將處理後的值存入新的 DataFrame
        winsorized_df[col] = winsorized_values
        
        print(f"變數 {variables[col]} 的 winsorization 處理結果：")
        print(f"原始範圍：[{df[col].min():.4f}, {df[col].max():.4f}]")
        print(f"處理後範圍：[{winsorized_df[col].min():.4f}, {winsorized_df[col].max():.4f}]")
        print(f"上下限值：[{lower_bound:.4f}, {upper_bound:.4f}]")
        print()

# 使用 winsorized_df 替換原始 df 進行後續分析
df = winsorized_df

# 繪製盒鬚圖
plt.figure(figsize=(15, 10))
for i, (col, display_name) in enumerate(variables.items(), 1):
    if col in df.columns:
        plt.subplot(3, 4, i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {display_name}')
        plt.ylabel('Value')
        
        # 計算異常值
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 計算異常值數量
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        # 在圖上顯示異常值數量
        plt.text(0.05, 0.95, f'Outliers: {outlier_count}\n({outlier_count/len(df)*100:.1f}%)',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
plt.tight_layout()
plt.show()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 輸出異常值統計資訊
print("\nOutlier Statistics for Each Variable:")
for col, display_name in variables.items():
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\n{display_name}:")
        print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        print(f"Outlier range: < {lower_bound:.2f} or > {upper_bound:.2f}")

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


## 開始進行線性迴歸分析
# 在進行迴歸分析之前，創建 year 和 industry 的虛擬變數
# 使用 pandas 的 get_dummies 函數創建虛擬變數

# 在進行 get_dummies 之前，先檢查資料類型
print("Year 欄位資料類型：", df['year'].dtype)
print("ICB code 欄位資料類型：", df['icbcode'].dtype)

# 確保資料類型正確，並移除小數點
df['year'] = df['year'].astype(int)
df['icbcode'] = df['icbcode'].astype(int)

# 檢查是否有特殊值
print("\nYear 欄位唯一值：", df['year'].unique())
print("ICB code 欄位唯一值：", df['icbcode'].unique())

# 然後再進行 get_dummies
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
industry_dummies = pd.get_dummies(df['icbcode'], prefix='industry', drop_first=True)

# 將虛擬變數與原始數據合併
df_with_dummies = pd.concat([df, year_dummies, industry_dummies], axis=1)

# 更新 required_vars 以包含所有虛擬變數
required_vars = ['gap', 'gap_e', 'gap_s', 'family', 'gov', 'g', 'size', 'lev', 'roa', 'mtb', 'kz'] + \
                list(year_dummies.columns) + list(industry_dummies.columns)

# 檢查所需的變數是否存在
missing_vars = [var for var in required_vars if var not in df_with_dummies.columns]
if missing_vars:
    print(f"\n警告：以下變數在資料集中不存在：{missing_vars}")
    print("請確認這些變數的名稱是否正確，或者是否需要先計算這些變數。")
else:
    print("\n所有需要的變數都存在於資料集中。")

# 如果所有變數都存在，則進行線性迴歸分析
# without legal
if not missing_vars:
    # 檢查數據類型
    print("\n檢查數據類型：")
    for var in ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz"]:
        print(f"{var}: {df_with_dummies[var].dtype}")
        # 檢查是否有非數值型數據
        non_numeric = df_with_dummies[var].apply(lambda x: not isinstance(x, (int, float, np.number)))
        if non_numeric.any():
            print(f"警告：{var} 包含非數值型數據")
            print(df_with_dummies[var][non_numeric].head())
    
    # 確保所有變數都是數值型
    for var in ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz"]:
        df_with_dummies[var] = pd.to_numeric(df_with_dummies[var], errors='coerce')
    
    # 檢查並處理虛擬變數
    for col in year_dummies.columns:
        df_with_dummies[col] = df_with_dummies[col].astype(float)
    for col in industry_dummies.columns:
        df_with_dummies[col] = df_with_dummies[col].astype(float)
    
    # 檢查是否有任何 NaN 值
    nan_check = df_with_dummies[["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz"]].isna().sum()
    print("\nNaN 值檢查：")
    print(nan_check)
    
    # 移除包含 NaN 的行
    df_with_dummies = df_with_dummies.dropna(subset=["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz"])
    print(f"\n移除 NaN 後的樣本數：{len(df_with_dummies)}")
    
    # 定義三個模型的變數
    models = {
        "Model 1": {
            "Y": "gap",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        },
        "Model 2": {
            "Y": "gap_e",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        },
        "Model 3": {
            "Y": "gap_s",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        }
    }
    
    # 整理成表格格式（每個變數兩列：一列係數+星號，一列t值）
    var_order = [
        "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "_cons"
    ] + list(year_dummies.columns) + list(industry_dummies.columns)
    
    var_display = {
        "family": "Family",
        "gov": "Gov",
        "g": "G",
        "size": "Size",
        "lev": "Lev",
        "roa": "ROA",
        "mtb": "MTB",
        "kz": "KZ",
        "_cons": "_cons"
    }
    # 為虛擬變數添加顯示名稱
    for col in year_dummies.columns:
        var_display[col] = col
    for col in industry_dummies.columns:
        var_display[col] = col
    
    # 收集每個模型的結果
    table_rows = []
    for v in var_order:
        coef_row = {"Variable": var_display[v]}
        tval_row = {"Variable": ""}
        for model_name, model_vars in models.items():
            X = df_with_dummies[model_vars["X"]]
            y = df_with_dummies[model_vars["Y"]]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            if v == "_cons":
                coef = model.params["const"]
                tval = model.tvalues["const"]
                pval = model.pvalues["const"]
            else:
                coef = model.params.get(v, np.nan)
                tval = model.tvalues.get(v, np.nan)
                pval = model.pvalues.get(v, np.nan)
            stars = ""
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            coef_row[model_name] = f"{coef:.4f}{stars}" if not np.isnan(coef) else ""
            tval_row[model_name] = f"({tval:.2f})" if not np.isnan(tval) else ""
        table_rows.append(coef_row)
        table_rows.append(tval_row)
    # adj. R-sq
    adjr_row = {"Variable": "adj. R-sq"}
    for model_name, model_vars in models.items():
        X = df_with_dummies[model_vars["X"]]
        y = df_with_dummies[model_vars["Y"]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        adjr_row[model_name] = f"{model.rsquared_adj:.3f}"
    table_rows.append(adjr_row)
    regression_table = pd.DataFrame(table_rows)
    regression_table.to_csv("regression_table_without_legal.csv", index=False)
    print("\n已輸出整理後的迴歸表格 regression_table_without_legal.csv（t值在係數下一列）。")

# with legal
if not missing_vars:
    # 定義三個模型的變數
    models = {
        "Model 1": {
            "Y": "gap",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "legal"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        },
        "Model 2": {
            "Y": "gap_e",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "legal"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        },
        "Model 3": {
            "Y": "gap_s",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "legal"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        }
    }
    
    # 整理成表格格式（每個變數兩列：一列係數+星號，一列t值）
    var_order = [
        "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "legal", "_cons"
    ] + list(year_dummies.columns) + list(industry_dummies.columns)
    
    var_display = {
        "family": "Family",
        "gov": "Gov",
        "g": "G",
        "size": "Size",
        "lev": "Lev",
        "roa": "ROA",
        "mtb": "MTB",
        "kz": "KZ",
        "legal": "Legal",
        "_cons": "_cons"
    }
    # 為虛擬變數添加顯示名稱
    for col in year_dummies.columns:
        var_display[col] = col
    for col in industry_dummies.columns:
        var_display[col] = col
    
    # 收集每個模型的結果
    table_rows = []
    for v in var_order:
        coef_row = {"Variable": var_display[v]}
        tval_row = {"Variable": ""}
        for model_name, model_vars in models.items():
            X = df_with_dummies[model_vars["X"]]
            y = df_with_dummies[model_vars["Y"]]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            if v == "_cons":
                coef = model.params["const"]
                tval = model.tvalues["const"]
                pval = model.pvalues["const"]
            else:
                coef = model.params.get(v, np.nan)
                tval = model.tvalues.get(v, np.nan)
                pval = model.pvalues.get(v, np.nan)
            stars = ""
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            coef_row[model_name] = f"{coef:.4f}{stars}" if not np.isnan(coef) else ""
            tval_row[model_name] = f"({tval:.2f})" if not np.isnan(tval) else ""
        table_rows.append(coef_row)
        table_rows.append(tval_row)
    # adj. R-sq
    adjr_row = {"Variable": "adj. R-sq"}
    for model_name, model_vars in models.items():
        X = df_with_dummies[model_vars["X"]]
        y = df_with_dummies[model_vars["Y"]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        adjr_row[model_name] = f"{model.rsquared_adj:.3f}"
    table_rows.append(adjr_row)
    regression_table = pd.DataFrame(table_rows)
    regression_table.to_csv("regression_table_with_legal.csv", index=False)
    print("\n已輸出整理後的迴歸表格 regression_table_with_legal.csv（t值在係數下一列）。")