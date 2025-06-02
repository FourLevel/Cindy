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

# 按照A-Z排序，但將"Other"放在最後
def sort_industries(industry_counts):
    # 將產業名稱按A-Z排序
    sorted_industries = sorted(industry_counts.index)
    
    # 如果有"Other"，將它移到最後
    if "Other" in sorted_industries:
        sorted_industries.remove("Other")
        sorted_industries.append("Other")
    
    # 根據排序後的順序重新整理industry_counts
    sorted_counts = pd.Series([industry_counts[industry] for industry in sorted_industries], 
                             index=sorted_industries)
    return sorted_counts

industry_counts = sort_industries(industry_counts)

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
    'boardSize': 'Board Size',
    'CEOdual': 'CEO Duality',
    'CSRcmte': 'CSR Committee',
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
plt.figure(figsize=(16, 12))
for i, (col, display_name) in enumerate(variables.items(), 1):
    if col in df.columns:
        plt.subplot(4, 4, i)
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

# 計算交互項
df_with_dummies['g_family'] = df_with_dummies['g'] * df_with_dummies['family']
df_with_dummies['g_gov'] = df_with_dummies['g'] * df_with_dummies['gov']

# 更新 required_vars 以包含所有虛擬變數和交互項
required_vars = ['gap', 'gap_e', 'gap_s', 'family', 'gov', 'g', 'size', 'lev', 'roa', 'mtb', 'kz', 'boardSize', 'CEOdual', 'CSRcmte', 'g_family', 'g_gov'] + \
                list(year_dummies.columns) + list(industry_dummies.columns)

# 檢查所需的變數是否存在
missing_vars = [var for var in required_vars if var not in df_with_dummies.columns]
if missing_vars:
    print(f"\n警告：以下變數在資料集中不存在：{missing_vars}")
    print("請確認這些變數的名稱是否正確，或者是否需要先計算這些變數。")
else:
    print("\n所有需要的變數都存在於資料集中。")

# 如果所有變數都存在，則進行線性迴歸分析
if not missing_vars:
    # 檢查數據類型
    print("\n檢查數據類型：")
    for var in ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"]:
        print(f"{var}: {df_with_dummies[var].dtype}")
        # 檢查是否有非數值型數據
        non_numeric = df_with_dummies[var].apply(lambda x: not isinstance(x, (int, float, np.number)))
        if non_numeric.any():
            print(f"警告：{var} 包含非數值型數據")
            print(df_with_dummies[var][non_numeric].head())
    
    # 確保所有變數都是數值型
    for var in ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"]:
        df_with_dummies[var] = pd.to_numeric(df_with_dummies[var], errors='coerce')
    
    # 檢查並處理虛擬變數
    for col in year_dummies.columns:
        df_with_dummies[col] = df_with_dummies[col].astype(float)
    for col in industry_dummies.columns:
        df_with_dummies[col] = df_with_dummies[col].astype(float)
    
    # 檢查是否有任何 NaN 值
    nan_check = df_with_dummies[["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"]].isna().sum()
    print("\nNaN 值檢查：")
    print(nan_check)
    
    # 移除包含 NaN 的行
    df_with_dummies = df_with_dummies.dropna(subset=["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"])
    print(f"\n移除 NaN 後的樣本數：{len(df_with_dummies)}")
    
    # # 定義三個模型的變數
    # models = {
    #     "Model 1": {
    #         "Y": "gap",
    #         "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
    #              list(year_dummies.columns) + list(industry_dummies.columns)
    #     },
    #     "Model 2": {
    #         "Y": "gap_e",
    #         "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
    #              list(year_dummies.columns) + list(industry_dummies.columns)
    #     },
    #     "Model 3": {
    #         "Y": "gap_s",
    #         "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
    #              list(year_dummies.columns) + list(industry_dummies.columns)
    #     }
    # }
    
    # # 整理成表格格式（每個變數兩列：一列係數+星號，一列t值）
    # var_order = [
    #     "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"
    # ]
    
    # var_display = {
    #     "family": "Family",
    #     "gov": "Gov",
    #     "g": "G",
    #     "size": "Size",
    #     "lev": "Lev",
    #     "roa": "ROA",
    #     "mtb": "MTB",
    #     "kz": "KZ",
    #     "boardSize": "Board Size",
    #     "CEOdual": "CEO Duality",
    #     "CSRcmte": "CSR Committee",
    #     "g_family": "G*Family",
    #     "g_gov": "G*Gov"
    # }
    
    # # 收集每個模型的結果
    # table_rows = []
    # for v in var_order:
    #     coef_row = {"Variable": var_display[v]}
    #     tval_row = {"Variable": ""}
    #     for model_name, model_vars in models.items():
    #         X = df_with_dummies[model_vars["X"]]
    #         y = df_with_dummies[model_vars["Y"]]
    #         X = sm.add_constant(X)
    #         model = sm.OLS(y, X).fit()
    #         coef = model.params.get(v, np.nan)
    #         tval = model.tvalues.get(v, np.nan)
    #         pval = model.pvalues.get(v, np.nan)
    #         stars = ""
    #         if pval < 0.01:
    #             stars = "***"
    #         elif pval < 0.05:
    #             stars = "**"
    #         elif pval < 0.1:
    #             stars = "*"
    #         coef_row[model_name] = f"{coef:.4f}{stars}" if not np.isnan(coef) else ""
    #         tval_row[model_name] = f"({tval:.2f})" if not np.isnan(tval) else ""
    #     table_rows.append(coef_row)
    #     table_rows.append(tval_row)
    
    # # 添加Year和Industry控制變數行
    # table_rows.append({"Variable": "Year", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"})
    # table_rows.append({"Variable": "", "Model 1": "", "Model 2": "", "Model 3": ""})
    # table_rows.append({"Variable": "Industry", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"})
    # table_rows.append({"Variable": "", "Model 1": "", "Model 2": "", "Model 3": ""})
    
    # # 添加常數項(_cons)
    # const_row = {"Variable": "_cons"}
    # const_tval_row = {"Variable": ""}
    # for model_name, model_vars in models.items():
    #     X = df_with_dummies[model_vars["X"]]
    #     y = df_with_dummies[model_vars["Y"]]
    #     X = sm.add_constant(X)
    #     model = sm.OLS(y, X).fit()
    #     coef = model.params["const"]
    #     tval = model.tvalues["const"]
    #     pval = model.pvalues["const"]
    #     stars = ""
    #     if pval < 0.01:
    #         stars = "***"
    #     elif pval < 0.05:
    #         stars = "**"
    #     elif pval < 0.1:
    #         stars = "*"
    #     const_row[model_name] = f"{coef:.4f}{stars}"
    #     const_tval_row[model_name] = f"({tval:.2f})"
    # table_rows.append(const_row)
    # table_rows.append(const_tval_row)
    
    # # adj. R-sq
    # adjr_row = {"Variable": "adj. R-sq"}
    # for model_name, model_vars in models.items():
    #     X = df_with_dummies[model_vars["X"]]
    #     y = df_with_dummies[model_vars["Y"]]
    #     X = sm.add_constant(X)
    #     model = sm.OLS(y, X).fit()
    #     adjr_row[model_name] = f"{model.rsquared_adj:.3f}"
    # table_rows.append(adjr_row)
    # regression_table = pd.DataFrame(table_rows)
    # regression_table.to_csv("regression_table.csv", index=False, encoding='utf-8-sig')
    # print("\n已輸出整理後的迴歸表格 regression_table.csv（t值在係數下一列）。")


## 線性迴歸分析（沒有交互項版本）
print("\n" + "="*60)
print("OLS線性迴歸分析 - 沒有交互項版本")
print("="*60)

# 定義三個模型的變數（無交互項）
models_no_interaction = {
    "Model 1": {
        "Y": "gap",
        "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"] + \
             list(year_dummies.columns) + list(industry_dummies.columns)
    },
    "Model 2": {
        "Y": "gap_e",
        "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"] + \
             list(year_dummies.columns) + list(industry_dummies.columns)
    },
    "Model 3": {
        "Y": "gap_s",
        "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"] + \
             list(year_dummies.columns) + list(industry_dummies.columns)
    }
}

# 整理成表格格式（無交互項）
var_order_no_interaction = [
    "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"
]

var_display_no_interaction = {
    "family": "Family",
    "gov": "Gov",
    "g": "G",
    "size": "Size",
    "lev": "Lev",
    "roa": "ROA",
    "mtb": "MTB",
    "kz": "KZ",
    "boardSize": "Board Size",
    "CEOdual": "CEO Duality",
    "CSRcmte": "CSR Committee"
}

# 收集每個模型的結果（無交互項）
table_rows_no_interaction = []
for v in var_order_no_interaction:
    coef_row = {"Variable": var_display_no_interaction[v]}
    tval_row = {"Variable": ""}
    for model_name, model_vars in models_no_interaction.items():
        X = df_with_dummies[model_vars["X"]]
        y = df_with_dummies[model_vars["Y"]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
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
    table_rows_no_interaction.append(coef_row)
    table_rows_no_interaction.append(tval_row)

# 添加Year和Industry控制變數行
table_rows_no_interaction.append({"Variable": "Year", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"})
table_rows_no_interaction.append({"Variable": "", "Model 1": "", "Model 2": "", "Model 3": ""})
table_rows_no_interaction.append({"Variable": "Industry", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"})
table_rows_no_interaction.append({"Variable": "", "Model 1": "", "Model 2": "", "Model 3": ""})

# 添加常數項(_cons)
const_row_no_interaction = {"Variable": "_cons"}
const_tval_row_no_interaction = {"Variable": ""}
for model_name, model_vars in models_no_interaction.items():
    X = df_with_dummies[model_vars["X"]]
    y = df_with_dummies[model_vars["Y"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    coef = model.params["const"]
    tval = model.tvalues["const"]
    pval = model.pvalues["const"]
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.1:
        stars = "*"
    const_row_no_interaction[model_name] = f"{coef:.4f}{stars}"
    const_tval_row_no_interaction[model_name] = f"({tval:.2f})"
table_rows_no_interaction.append(const_row_no_interaction)
table_rows_no_interaction.append(const_tval_row_no_interaction)

# adj. R-sq（無交互項）
adjr_row_no_interaction = {"Variable": "adj. R-sq"}
for model_name, model_vars in models_no_interaction.items():
    X = df_with_dummies[model_vars["X"]]
    y = df_with_dummies[model_vars["Y"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    adjr_row_no_interaction[model_name] = f"{model.rsquared_adj:.3f}"
table_rows_no_interaction.append(adjr_row_no_interaction)

regression_table_no_interaction = pd.DataFrame(table_rows_no_interaction)
regression_table_no_interaction.to_csv("regression_table_no_interaction.csv", index=False, encoding='utf-8-sig')
print("\n已輸出整理後的迴歸表格（無交互項）regression_table_no_interaction.csv（t值在係數下一列）。")

print("\n" + "="*60)
print("已完成兩個版本的OLS線性迴歸分析：")
print("1. 包含交互項版本：regression_table.csv") 
print("2. 不含交互項版本：regression_table_no_interaction.csv")
print("="*60)


## 進行 2SLS 迴歸分析 - 有交互項版本
print("\n" + "="*80)
print("2SLS 迴歸分析 - 有交互項版本")
print("="*80)

# 檢查工具變數是否存在
iv_vars = ['freeFloatShareholding', 'insiderShareholding']

print("\n檢查工具變數存在性：")
for var in iv_vars:
    if var in df_with_dummies.columns:
        print(f"{var} 存在")
    else:
        print(f"{var} 不存在")

missing_iv = [var for var in iv_vars if var not in df_with_dummies.columns]

if missing_iv:
    print(f"\n警告：以下工具變數在資料集中不存在：{missing_iv}")
    print("請確認工具變數的欄位名稱是否正確。")
else:
    print("\n工具變數檢查通過，開始進行2SLS分析（有交互項版本）...")
    
    # 確保工具變數為數值型
    for var in iv_vars:
        df_with_dummies[var] = pd.to_numeric(df_with_dummies[var], errors='coerce')
    
    # 移除包含NaN的行
    required_vars_2sls = iv_vars + ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"]
    df_2sls = df_with_dummies.dropna(subset=required_vars_2sls)
    print(f"2SLS分析樣本數（有交互項）：{len(df_2sls)}")
    
    try:
        # 導入2SLS所需的套件
        from linearmodels import IV2SLS
        import numpy as np
        
        # 重新計算交互項（因為可能有資料被移除）
        df_2sls['g_family'] = df_2sls['g'] * df_2sls['family']
        df_2sls['g_gov'] = df_2sls['g'] * df_2sls['gov']
        
        # 定義基本控制變數（包含交互項）
        base_controls = ["gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
                       list(year_dummies.columns) + list(industry_dummies.columns)
        
        # 定義三個2SLS模型（按照原始模型分類）
        models_2sls = {
            "Model 1": "gap",
            "Model 2": "gap_e", 
            "Model 3": "gap_s"
        }
        
        # 定義內生變數
        endog_vars = ['family']  # family 為內生的
        
        # 創建結果表格
        table_2sls_results = []
        
        # 定義變數顯示順序和名稱
        main_vars = ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"]
        
        var_display = {
            "family": "Family",
            "gov": "Gov",
            "g": "G",
            "size": "Size",
            "lev": "Lev",
            "roa": "ROA",
            "mtb": "MTB",
            "kz": "KZ",
            "boardSize": "Board Size",
            "CEOdual": "CEO Duality",
            "CSRcmte": "CSR Committee",
            "g_family": "G*Family",
            "g_gov": "G*Gov",
            "freeFloatShareholding": "Free float (IV1)",
            "insiderShareholding": "Insider (IV2)"
        }
        
        # 第一階段迴歸（Family作為依變數）
        print("\n=== 第一階段迴歸 (Family as dependent variable) ===")
        # 由於只有family是內生的，所有base_controls都可以包含在第一階段迴歸中
        first_stage_controls = base_controls
        first_stage_X = df_2sls[first_stage_controls + iv_vars]
        first_stage_X = sm.add_constant(first_stage_X)
        first_stage_y = df_2sls['family']
        first_stage_model = sm.OLS(first_stage_y, first_stage_X).fit(cov_type='HC1')
        
        # 第二階段迴歸（分別對三個依變數）
        second_stage_models = {}
        for model_name, dep_var in models_2sls.items():
            print(f"\n=== {model_name}: {dep_var} ===")
            
            # 外生變數（包含所有base_controls，因為只有family是內生的）
            exog_vars_list = base_controls
            
            y = df_2sls[dep_var]
            exog = df_2sls[exog_vars_list]
            exog = sm.add_constant(exog)  # 加入常數項
            endog = df_2sls[endog_vars]
            instruments = df_2sls[iv_vars]
            
            second_stage_models[model_name] = IV2SLS(y, exog, endog, instruments).fit(cov_type='robust')
        
        # 建立表格
        def get_coef_info(model, var_name):
            if var_name in model.params.index:
                coef = model.params[var_name]
                if hasattr(model, 'tstats'):
                    tstat = model.tstats[var_name]
                    pval = model.pvalues[var_name]
                else:
                    tstat = model.tvalues[var_name]
                    pval = model.pvalues[var_name]
                
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                
                return f"{coef:.4f}{stars}", f"({tstat:.2f})"
            else:
                return "", ""
        
        # 組織表格數據
        for var in main_vars + iv_vars:
            if var in var_display:
                var_name = var_display[var]
                
                # 第一階段結果
                if var in endog_vars:
                    first_stage_coef, first_stage_t = "", ""
                else:
                    first_stage_coef, first_stage_t = get_coef_info(first_stage_model, var)
                
                # 第二階段結果
                model1_coef, model1_t = get_coef_info(second_stage_models["Model 1"], var)
                model2_coef, model2_t = get_coef_info(second_stage_models["Model 2"], var)
                model3_coef, model3_t = get_coef_info(second_stage_models["Model 3"], var)
                
                # 添加係數行
                table_2sls_results.append({
                    "Variable": var_name,
                    "First Stage": first_stage_coef,
                    "Model 1": model1_coef,
                    "Model 2": model2_coef,
                    "Model 3": model3_coef
                })
                
                # 添加t值行
                table_2sls_results.append({
                    "Variable": "",
                    "First Stage": first_stage_t,
                    "Model 1": model1_t,
                    "Model 2": model2_t,
                    "Model 3": model3_t
                })
        
        # 添加控制變數和統計量
        table_2sls_results.extend([
            {"Variable": "Year", "First Stage": "Y", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"},
            {"Variable": "Industry", "First Stage": "Y", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"}
        ])
        
        # 常數項
        const_first, const_first_t = get_coef_info(first_stage_model, "const")
        const_1, const_1_t = get_coef_info(second_stage_models["Model 1"], "const")
        const_2, const_2_t = get_coef_info(second_stage_models["Model 2"], "const")
        const_3, const_3_t = get_coef_info(second_stage_models["Model 3"], "const")
        
        table_2sls_results.extend([
            {"Variable": "_cons", "First Stage": const_first, "Model 1": const_1, "Model 2": const_2, "Model 3": const_3},
            {"Variable": "", "First Stage": const_first_t, "Model 1": const_1_t, "Model 2": const_2_t, "Model 3": const_3_t}
        ])
        
        # 模型統計量
        table_2sls_results.extend([
            {
                "Variable": "N",
                "First Stage": f"{first_stage_model.nobs}",
                "Model 1": f"{second_stage_models['Model 1'].nobs}",
                "Model 2": f"{second_stage_models['Model 2'].nobs}",
                "Model 3": f"{second_stage_models['Model 3'].nobs}"
            },
            {
                "Variable": "R-sq",
                "First Stage": f"{first_stage_model.rsquared:.3f}",
                "Model 1": f"{second_stage_models['Model 1'].rsquared:.3f}",
                "Model 2": f"{second_stage_models['Model 2'].rsquared:.3f}",
                "Model 3": f"{second_stage_models['Model 3'].rsquared:.3f}"
            }
        ])
        
        # 儲存結果
        df_results_2sls = pd.DataFrame(table_2sls_results)
        df_results_2sls.to_csv("iv_2sls_results_with_interaction.csv", index=False)
        print("\n已輸出2SLS分析結果表格（有交互項）：iv_2sls_results_with_interaction.csv")
        
        # 顯示第一階段F統計量
        print(f"\n第一階段 F-statistic（有交互項）: {first_stage_model.fvalue:.2f}")
        
        # 顯示工具變數相關係數
        print("\n工具變數係數（有交互項）:")
        for iv in iv_vars:
            if iv in first_stage_model.params.index:
                coef = first_stage_model.params[iv]
                pval = first_stage_model.pvalues[iv]
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                print(f"{iv}: {coef:.4f}{stars}")
        
        print("\n" + "="*60)
        print("2SLS分析（有交互項版本）已完成！")
        print("輸出檔案：iv_2sls_results_with_interaction.csv")
        print("="*60)
        
    except ImportError:
        print("\n錯誤：需要安裝 linearmodels 套件來進行2SLS分析")
        print("請執行：pip install linearmodels")
    except Exception as e:
        print(f"\n2SLS分析（有交互項）過程中發生錯誤：{str(e)}")


## 進行 2SLS 迴歸分析 - 無交互項版本
print("\n" + "="*80)
print("2SLS 迴歸分析 - 無交互項版本")
print("="*80)

# 檢查工具變數是否存在
iv_vars = ['freeFloatShareholding', 'insiderShareholding']

print("\n檢查工具變數存在性：")
for var in iv_vars:
    if var in df_with_dummies.columns:
        print(f"{var} 存在")
    else:
        print(f"{var} 不存在")

missing_iv = [var for var in iv_vars if var not in df_with_dummies.columns]

if missing_iv:
    print(f"\n警告：以下工具變數在資料集中不存在：{missing_iv}")
    print("請確認工具變數的欄位名稱是否正確。")
else:
    print("\n工具變數檢查通過，開始進行2SLS分析（無交互項版本）...")
    
    # 確保工具變數為數值型
    for var in iv_vars:
        df_with_dummies[var] = pd.to_numeric(df_with_dummies[var], errors='coerce')
    
    # 移除包含NaN的行（無交互項版本不需要交互項變數）
    required_vars_2sls_no_int = iv_vars + ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"]
    df_2sls_no_int = df_with_dummies.dropna(subset=required_vars_2sls_no_int)
    print(f"2SLS分析樣本數（無交互項）：{len(df_2sls_no_int)}")
    
    try:
        # 導入2SLS所需的套件
        from linearmodels import IV2SLS
        import numpy as np
        
        # 定義基本控制變數（不包含交互項）
        base_controls_no_interaction = ["gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"] + \
                                     list(year_dummies.columns) + list(industry_dummies.columns)
        
        # 定義三個2SLS模型（按照原始模型分類）
        models_2sls_no_int = {
            "Model 1": "gap",
            "Model 2": "gap_e", 
            "Model 3": "gap_s"
        }
        
        # 定義內生變數
        endog_vars_no_int = ['family']  # family 為內生的
        
        # 創建結果表格（無交互項）
        table_2sls_results_no_interaction = []
        
        # 定義變數顯示順序和名稱（無交互項）
        main_vars_no_interaction = ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte"]
        
        var_display_no_interaction = {
            "family": "Family",
            "gov": "Gov", 
            "g": "G",
            "size": "Size",
            "lev": "Lev",
            "roa": "ROA",
            "mtb": "MTB",
            "kz": "KZ",
            "boardSize": "Board Size",
            "CEOdual": "CEO Duality",
            "CSRcmte": "CSR Committee",
            "freeFloatShareholding": "Free float (IV1)",
            "insiderShareholding": "Insider (IV2)"
        }
        
        # 第一階段迴歸（無交互項）
        print("\n=== 第一階段迴歸 (Family as dependent variable - 無交互項) ===")
        first_stage_controls_no_interaction = base_controls_no_interaction
        first_stage_X_no_interaction = df_2sls_no_int[first_stage_controls_no_interaction + iv_vars]
        first_stage_X_no_interaction = sm.add_constant(first_stage_X_no_interaction)
        first_stage_y_no_interaction = df_2sls_no_int['family']
        first_stage_model_no_interaction = sm.OLS(first_stage_y_no_interaction, first_stage_X_no_interaction).fit(cov_type='HC1')
        
        # 第二階段迴歸（無交互項）
        second_stage_models_no_interaction = {}
        for model_name, dep_var in models_2sls_no_int.items():
            print(f"\n=== {model_name}: {dep_var} (無交互項) ===")
            
            # 外生變數（不包含交互項）
            exog_vars_list_no_interaction = base_controls_no_interaction
            
            y = df_2sls_no_int[dep_var]
            exog = df_2sls_no_int[exog_vars_list_no_interaction]
            exog = sm.add_constant(exog)  # 加入常數項
            endog = df_2sls_no_int[endog_vars_no_int]  # 仍然只有family是內生的
            instruments = df_2sls_no_int[iv_vars]
            
            second_stage_models_no_interaction[model_name] = IV2SLS(y, exog, endog, instruments).fit(cov_type='robust')
        
        # 建立表格函數（無交互項）
        def get_coef_info_no_int(model, var_name):
            if var_name in model.params.index:
                coef = model.params[var_name]
                if hasattr(model, 'tstats'):
                    tstat = model.tstats[var_name]
                    pval = model.pvalues[var_name]
                else:
                    tstat = model.tvalues[var_name]
                    pval = model.pvalues[var_name]
                
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                
                return f"{coef:.4f}{stars}", f"({tstat:.2f})"
            else:
                return "", ""
        
        # 組織表格數據（無交互項）
        for var in main_vars_no_interaction + iv_vars:
            if var in var_display_no_interaction:
                var_name = var_display_no_interaction[var]
                
                # 第一階段結果
                if var in endog_vars_no_int:
                    first_stage_coef, first_stage_t = "", ""
                else:
                    first_stage_coef, first_stage_t = get_coef_info_no_int(first_stage_model_no_interaction, var)
                
                # 第二階段結果
                model1_coef, model1_t = get_coef_info_no_int(second_stage_models_no_interaction["Model 1"], var)
                model2_coef, model2_t = get_coef_info_no_int(second_stage_models_no_interaction["Model 2"], var)
                model3_coef, model3_t = get_coef_info_no_int(second_stage_models_no_interaction["Model 3"], var)
                
                # 添加係數行
                table_2sls_results_no_interaction.append({
                    "Variable": var_name,
                    "First Stage": first_stage_coef,
                    "Model 1": model1_coef,
                    "Model 2": model2_coef,
                    "Model 3": model3_coef
                })
                
                # 添加t值行
                table_2sls_results_no_interaction.append({
                    "Variable": "",
                    "First Stage": first_stage_t,
                    "Model 1": model1_t,
                    "Model 2": model2_t,
                    "Model 3": model3_t
                })
        
        # 添加控制變數和統計量（無交互項）
        table_2sls_results_no_interaction.extend([
            {"Variable": "Year", "First Stage": "Y", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"},
            {"Variable": "Industry", "First Stage": "Y", "Model 1": "Y", "Model 2": "Y", "Model 3": "Y"}
        ])
        
        # 常數項（無交互項）
        const_first_no_int, const_first_t_no_int = get_coef_info_no_int(first_stage_model_no_interaction, "const")
        const_1_no_int, const_1_t_no_int = get_coef_info_no_int(second_stage_models_no_interaction["Model 1"], "const")
        const_2_no_int, const_2_t_no_int = get_coef_info_no_int(second_stage_models_no_interaction["Model 2"], "const")
        const_3_no_int, const_3_t_no_int = get_coef_info_no_int(second_stage_models_no_interaction["Model 3"], "const")
        
        table_2sls_results_no_interaction.extend([
            {"Variable": "_cons", "First Stage": const_first_no_int, "Model 1": const_1_no_int, "Model 2": const_2_no_int, "Model 3": const_3_no_int},
            {"Variable": "", "First Stage": const_first_t_no_int, "Model 1": const_1_t_no_int, "Model 2": const_2_t_no_int, "Model 3": const_3_t_no_int}
        ])
        
        # 模型統計量（無交互項）
        table_2sls_results_no_interaction.extend([
            {
                "Variable": "N",
                "First Stage": f"{first_stage_model_no_interaction.nobs}",
                "Model 1": f"{second_stage_models_no_interaction['Model 1'].nobs}",
                "Model 2": f"{second_stage_models_no_interaction['Model 2'].nobs}",
                "Model 3": f"{second_stage_models_no_interaction['Model 3'].nobs}"
            },
            {
                "Variable": "R-sq",
                "First Stage": f"{first_stage_model_no_interaction.rsquared:.3f}",
                "Model 1": f"{second_stage_models_no_interaction['Model 1'].rsquared:.3f}",
                "Model 2": f"{second_stage_models_no_interaction['Model 2'].rsquared:.3f}",
                "Model 3": f"{second_stage_models_no_interaction['Model 3'].rsquared:.3f}"
            }
        ])
        
        # 儲存結果（無交互項）
        df_results_2sls_no_interaction = pd.DataFrame(table_2sls_results_no_interaction)
        df_results_2sls_no_interaction.to_csv("iv_2sls_results_no_interaction.csv", index=False)
        print("\n已輸出2SLS分析結果表格（無交互項）：iv_2sls_results_no_interaction.csv")
        
        # 顯示第一階段F統計量（無交互項）
        print(f"\n第一階段 F-statistic（無交互項）: {first_stage_model_no_interaction.fvalue:.2f}")
        
        # 顯示工具變數相關係數（無交互項）
        print("\n工具變數係數（無交互項）:")
        for iv in iv_vars:
            if iv in first_stage_model_no_interaction.params.index:
                coef = first_stage_model_no_interaction.params[iv]
                pval = first_stage_model_no_interaction.pvalues[iv]
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                print(f"{iv}: {coef:.4f}{stars}")
        
        print("\n" + "="*60)
        print("2SLS分析（無交互項版本）已完成！")
        print("輸出檔案：iv_2sls_results_no_interaction.csv")
        print("="*60)
        
    except ImportError:
        print("\n錯誤：需要安裝 linearmodels 套件來進行2SLS分析")
        print("請執行：pip install linearmodels")
    except Exception as e:
        print(f"\n2SLS分析（無交互項）過程中發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()


print("\n" + "="*80)
print("2SLS分析總結")
print("="*80)
print("已完成兩個獨立的2SLS分析版本：")
print("1. 有交互項版本：iv_2sls_results_with_interaction.csv")
print("2. 無交互項版本：iv_2sls_results_no_interaction.csv")
print("您可以根據需要分別執行這兩個部分。")
print("="*80)


## 進行 PSM (Propensity Score Matching) 分析 - 改進版包含多種參數調整選項
print("\n" + "="*80)
print("PSM (傾向分數匹配) 分析 - 改進版包含多種參數調整選項")
print("="*80)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist
    import warnings
    warnings.filterwarnings('ignore')
    
    # 確保資料完整性
    psm_vars = ['family', 'gov', 'g', 'size', 'lev', 'roa', 'mtb', 'kz', 'boardSize', 'CEOdual', 'CSRcmte', 
                'gap', 'gap_e', 'gap_s', 'year', 'icbcode', 'country']
    
    # 移除包含NaN的觀測值
    df_psm = df_with_dummies.dropna(subset=psm_vars).copy()
    print(f"PSM分析樣本數：{len(df_psm)}")
    
    # 定義改進版 PSM 函數
    def perform_enhanced_psm_analysis(
        treatment_var, 
        data, 
        control_vars, 
        outcome_vars,
        # 可調整參數
        binary_cutoff_method='median',  # 'median', 'mean', 'percentile_75', 'percentile_33'
        matching_method='nearest',      # 'nearest', 'caliper'
        matching_ratio='1:1',          # '1:1', '1:2', '1:3', '1:5'
        caliper_width=0.1,             # 當使用 caliper matching 時的容忍度
        with_replacement=False,         # 是否允許重複匹配
        trimming_alpha=0.0,            # 修剪極值的比例
        robust_se=True,                # 是否使用穩健標準誤
        balance_check=True,            # 是否進行匹配後平衡檢定
        ps_model_params=None           # 傾向分數模型參數
    ):
        """
        執行改進版 PSM 分析
        """
        print(f"\n{'='*60}")
        print(f"改進版 PSM 分析：{treatment_var}")
        print(f"二元化方法：{binary_cutoff_method}")
        print(f"匹配方法：{matching_method}")
        print(f"匹配比例：{matching_ratio}")
        if trimming_alpha > 0:
            print(f"修剪比例：{trimming_alpha*100:.1f}%")
        print(f"{'='*60}")
        
        # 1. 處理變數二元化（改進的切點選擇）
        if data[treatment_var].nunique() > 2:
            if binary_cutoff_method == 'median':
                cutoff = data[treatment_var].median()
            elif binary_cutoff_method == 'mean':
                cutoff = data[treatment_var].mean()
            elif binary_cutoff_method == 'percentile_75':
                cutoff = data[treatment_var].quantile(0.75)
            elif binary_cutoff_method == 'percentile_33':
                cutoff = data[treatment_var].quantile(0.33)
            else:
                cutoff = data[treatment_var].median()  # 預設使用中位數
            
            data[f'{treatment_var}_binary'] = (data[treatment_var] > cutoff).astype(int)
            treatment_binary = f'{treatment_var}_binary'
            print(f"{treatment_var} 為連續變數，使用 {binary_cutoff_method} 方法")
            print(f"切點：{cutoff:.4f}")
        else:
            treatment_binary = treatment_var
            print(f"{treatment_var} 已為二元變數")
        
        # 計算處理組和控制組樣本數
        treatment_counts = data[treatment_binary].value_counts()
        print(f"控制組 (0): {treatment_counts.get(0, 0)}")
        print(f"處理組 (1): {treatment_counts.get(1, 0)}")
        
        # 2. 估計傾向分數（改進的模型設定）
        X = data[control_vars]
        y = data[treatment_binary]
        
        # 標準化特徵
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 改進的邏輯迴歸參數設定
        if ps_model_params is None:
            ps_model_params = {
                'random_state': 42,
                'max_iter': 2000,
                'C': 1.0,  # 可調整正則化強度
                'solver': 'liblinear'
            }
        
        ps_model = LogisticRegression(**ps_model_params)
        ps_model.fit(X_scaled, y)
        
        # 計算傾向分數
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        data['propensity_score'] = propensity_scores
        
        print(f"傾向分數範圍：[{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
        print(f"傾向分數標準差：{propensity_scores.std():.4f}")
        
        # 3. 進行修剪 (Trimming) - 移除極端傾向分數
        if trimming_alpha > 0:
            lower_bound = np.quantile(propensity_scores, trimming_alpha)
            upper_bound = np.quantile(propensity_scores, 1 - trimming_alpha)
            
            original_n = len(data)
            data = data[(data['propensity_score'] >= lower_bound) & 
                       (data['propensity_score'] <= upper_bound)]
            trimmed_n = original_n - len(data)
            
            print(f"修剪掉 {trimmed_n} 個極端觀測值")
            
            # 重新計算處理組和控制組樣本數
            treatment_counts = data[treatment_binary].value_counts()
            print(f"修剪後 - 控制組 (0): {treatment_counts.get(0, 0)}")
            print(f"修剪後 - 處理組 (1): {treatment_counts.get(1, 0)}")
        
        # 4. 進行匹配（改進的匹配演算法）
        treated_indices = data[data[treatment_binary] == 1].index.tolist()
        control_indices = data[data[treatment_binary] == 0].index.tolist()
        
        treated_ps = data.loc[treated_indices, 'propensity_score'].values.reshape(-1, 1)
        control_ps = data.loc[control_indices, 'propensity_score'].values.reshape(-1, 1)
        
        if matching_method == 'nearest':
            # 最近鄰匹配
            distances = cdist(treated_ps, control_ps, metric='euclidean')
            
            # 決定匹配比例
            if matching_ratio == '1:1':
                n_matches = 1
            elif matching_ratio == '1:2':
                n_matches = 2
            elif matching_ratio == '1:3':
                n_matches = 3
            elif matching_ratio == '1:5':
                n_matches = 5
            else:
                n_matches = 1  # 預設1:1匹配
            
            matched_control_indices = []
            used_controls = set()
            
            for i, treated_idx in enumerate(treated_indices):
                if with_replacement:
                    # 允許重複匹配
                    best_matches = np.argsort(distances[i])[:n_matches]
                    matched_controls = [control_indices[j] for j in best_matches]
                else:
                    # 不允許重複匹配
                    available_controls = [j for j, ctrl_idx in enumerate(control_indices) 
                                        if ctrl_idx not in used_controls]
                    if len(available_controls) >= n_matches:
                        best_matches_indices = np.argsort(distances[i, available_controls])[:n_matches]
                        matched_controls = [control_indices[available_controls[j]] for j in best_matches_indices]
                        used_controls.update(matched_controls)
                    else:
                        matched_controls = []
                
                matched_control_indices.append(matched_controls)
        
        elif matching_method == 'caliper':
            # Caliper 匹配（帶容忍度）
            matched_control_indices = []
            used_controls = set()
            
            # 決定匹配比例
            if matching_ratio == '1:1':
                n_matches = 1
            elif matching_ratio == '1:2':
                n_matches = 2
            elif matching_ratio == '1:3':
                n_matches = 3
            elif matching_ratio == '1:5':
                n_matches = 5
            else:
                n_matches = 1
            
            for i, treated_idx in enumerate(treated_indices):
                treated_score = treated_ps[i][0]
                
                # 找到在 caliper 範圍內的控制組
                valid_controls = []
                for j, control_idx in enumerate(control_indices):
                    if not with_replacement and control_idx in used_controls:
                        continue
                    
                    control_score = control_ps[j][0]
                    if abs(treated_score - control_score) <= caliper_width:
                        valid_controls.append((j, control_idx, abs(treated_score - control_score)))
                
                # 按距離排序，選擇最近的
                if valid_controls:
                    valid_controls.sort(key=lambda x: x[2])  # 按距離排序
                    selected = valid_controls[:min(n_matches, len(valid_controls))]
                    matched_controls = [ctrl_idx for _, ctrl_idx, _ in selected]
                    
                    if not with_replacement:
                        used_controls.update(matched_controls)
                else:
                    matched_controls = []
                
                matched_control_indices.append(matched_controls)
            
            print(f"Caliper 寬度: {caliper_width}")
        
        # 建立匹配後的資料集
        matched_treated = []
        matched_controls = []
        
        for i, treated_idx in enumerate(treated_indices):
            if matched_control_indices[i]:
                for control_idx in matched_control_indices[i]:
                    matched_treated.append(treated_idx)
                    matched_controls.append(control_idx)
        
        print(f"成功匹配的處理組觀測值：{len(set(matched_treated))}")
        print(f"成功匹配的控制組觀測值：{len(matched_controls)}")
        print(f"總匹配對數：{len(matched_controls)}")
        
        # 建立匹配後的資料集
        matched_data = pd.concat([
            data.loc[matched_treated].assign(_weight=1, _matched_type='treated'),
            data.loc[matched_controls].assign(_weight=1, _matched_type='control')
        ])
        
        # 5. 進行匹配後的平衡檢定
        if balance_check:
            print(f"\n{'='*40}")
            print("匹配後平衡檢定")
            print(f"{'='*40}")
            
            for var in control_vars[:5]:  # 檢查前5個控制變數
                if var in matched_data.columns:
                    treated_mean = matched_data[matched_data[treatment_binary]==1][var].mean()
                    control_mean = matched_data[matched_data[treatment_binary]==0][var].mean()
                    
                    # 計算標準化偏差
                    pooled_std = matched_data[var].std()
                    if pooled_std > 0:
                        std_bias = (treated_mean - control_mean) / pooled_std
                    else:
                        std_bias = 0
                    
                    # t檢定
                    from scipy.stats import ttest_ind
                    try:
                        t_stat, p_val = ttest_ind(
                            matched_data[matched_data[treatment_binary]==1][var].dropna(),
                            matched_data[matched_data[treatment_binary]==0][var].dropna()
                        )
                    except:
                        p_val = np.nan
                    
                    print(f"{var}: 標準化偏差 = {std_bias:.3f}, t檢定 p值 = {p_val:.3f}")
        
        # 6. 使用匹配資料進行結果迴歸分析
        results = {}
        
        for outcome_var in outcome_vars:
            print(f"\n{'-'*40}")
            print(f"{outcome_var} 的匹配後迴歸結果")
            print(f"{'-'*40}")
            
            # 準備迴歸變數（簡化版：只包含處理變數）
            reg_vars = [treatment_var]
            available_vars = [var for var in reg_vars if var in matched_data.columns]
            
            X_reg = matched_data[available_vars]
            y_reg = matched_data[outcome_var]
            
            # 加入常數項
            X_reg = sm.add_constant(X_reg)
            
            # 使用權重進行迴歸
            weights = matched_data['_weight']
            
            try:
                if robust_se:
                    model = sm.WLS(y_reg, X_reg, weights=weights).fit(cov_type='HC1')
                else:
                    model = sm.WLS(y_reg, X_reg, weights=weights).fit()
                
                # 提取主要變數的係數
                if treatment_var in model.params.index:
                    coef = model.params[treatment_var]
                    tval = model.tvalues[treatment_var]
                    pval = model.pvalues[treatment_var]
                    
                    # 計算信賴區間
                    conf_int = model.conf_int().loc[treatment_var]
                    
                    stars = ""
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.1:
                        stars = "*"
                    
                    results[outcome_var] = {
                        'coefficient': coef,
                        't_value': tval,
                        'p_value': pval,
                        'significance': stars,
                        'n_obs': model.nobs,
                        'r_squared': model.rsquared,
                        'conf_int_lower': conf_int[0],
                        'conf_int_upper': conf_int[1]
                    }
                    
                    print(f"{treatment_var}: {coef:.4f}{stars} (t={tval:.2f}, p={pval:.3f})")
                    print(f"95% 信賴區間: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
                    print(f"觀測數: {model.nobs}")
                    print(f"R-squared: {model.rsquared:.3f}")
                else:
                    print(f"警告：{treatment_var} 不在迴歸結果中")
                    
            except Exception as e:
                print(f"迴歸分析錯誤：{str(e)}")
                results[outcome_var] = {'error': str(e)}
        
        return results, matched_data
    
    # 定義控制變數和結果變數
    control_vars_psm = ['g', 'size', 'lev', 'roa', 'mtb', 'kz', 'boardSize', 'CEOdual', 'CSRcmte']
    outcome_vars_psm = ['gap', 'gap_e', 'gap_s']
    
    # 定義多種規格設定進行比較
    specifications = [
        {
            'name': '基準模型（中位數切點）',
            'params': {
                'binary_cutoff_method': 'median',
                'matching_method': 'nearest',
                'matching_ratio': '1:1',
                'with_replacement': False,
                'trimming_alpha': 0.0
            }
        },
        {
            'name': '改進模型1（75百分位切點）',
            'params': {
                'binary_cutoff_method': 'percentile_75',
                'matching_method': 'nearest',
                'matching_ratio': '1:1',
                'with_replacement': False,
                'trimming_alpha': 0.05
            }
        },
        {
            'name': '改進模型2（Caliper匹配）',
            'params': {
                'binary_cutoff_method': 'median',
                'matching_method': 'caliper',
                'matching_ratio': '1:1',
                'caliper_width': 0.05,
                'with_replacement': False,
                'trimming_alpha': 0.05
            }
        },
        {
            'name': '改進模型3（1:2匹配）',
            'params': {
                'binary_cutoff_method': 'median',
                'matching_method': 'nearest',
                'matching_ratio': '1:2',
                'with_replacement': False,
                'trimming_alpha': 0.05
            }
        }
    ]
    
    # 儲存所有結果
    all_results = {}
    
    # 對每個規格設定進行分析
    for spec in specifications:
        print(f"\n{'='*80}")
        print(f"正在執行：{spec['name']}")
        print(f"{'='*80}")
        
        # Family PSM 分析
        print(f"\n{'-'*60}")
        print("Family 變數分析")
        print(f"{'-'*60}")
        
        family_results, family_matched = perform_enhanced_psm_analysis(
            treatment_var='family',
            data=df_psm.copy(),
            control_vars=control_vars_psm,
            outcome_vars=outcome_vars_psm,
            **spec['params']
        )
        
        # Gov PSM 分析
        print(f"\n{'-'*60}")
        print("Gov 變數分析")
        print(f"{'-'*60}")
        
        gov_results, gov_matched = perform_enhanced_psm_analysis(
            treatment_var='gov',
            data=df_psm.copy(),
            control_vars=control_vars_psm,
            outcome_vars=outcome_vars_psm,
            **spec['params']
        )
        
        all_results[spec['name']] = {
            'family': family_results,
            'gov': gov_results
        }
    
    # 整理比較結果為表格形式
    def create_comparison_table(all_results, outcome_vars):
        """建立多規格比較表格"""
        table_rows = []
        
        for outcome in outcome_vars:
            for spec_name, results in all_results.items():
                # Family 結果
                if outcome in results['family'] and 'coefficient' in results['family'][outcome]:
                    fr = results['family'][outcome]
                    family_result = f"{fr['coefficient']:.4f}{fr['significance']} (t={fr['t_value']:.2f})"
                else:
                    family_result = "N/A"
                
                # Gov 結果
                if outcome in results['gov'] and 'coefficient' in results['gov'][outcome]:
                    gr = results['gov'][outcome]
                    gov_result = f"{gr['coefficient']:.4f}{gr['significance']} (t={gr['t_value']:.2f})"
                else:
                    gov_result = "N/A"
                
                table_rows.append({
                    'Outcome': outcome.upper(),
                    'Specification': spec_name,
                    'Family_PSM': family_result,
                    'Gov_PSM': gov_result
                })
        
        return pd.DataFrame(table_rows)
    
    # 建立並儲存比較表格
    comparison_table = create_comparison_table(all_results, outcome_vars_psm)
    comparison_table.to_csv("psm_multiple_specifications_comparison.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*80}")
    print("PSM 多規格比較結果")
    print(f"{'='*80}")
    print(comparison_table.to_string(index=False))
    print(f"\n已輸出比較結果：psm_multiple_specifications_comparison.csv")
    
    # 也保存基準模型的詳細結果（與原來格式相容）
    baseline_results = all_results['基準模型（中位數切點）']
    
    def create_baseline_table(family_results, gov_results, outcome_vars):
        """建立基準模型結果表格（相容原格式）"""
        table_rows = []
        
        for outcome in outcome_vars:
            # Family結果
            if outcome in family_results and 'coefficient' in family_results[outcome]:
                fr = family_results[outcome]
                family_coef = f"{fr['coefficient']:.4f}{fr['significance']}"
                family_t = f"({fr['t_value']:.2f})"
            else:
                family_coef = ""
                family_t = ""
            
            # Gov結果
            if outcome in gov_results and 'coefficient' in gov_results[outcome]:
                gr = gov_results[outcome]
                gov_coef = f"{gr['coefficient']:.4f}{gr['significance']}"
                gov_t = f"({gr['t_value']:.2f})"
            else:
                gov_coef = ""
                gov_t = ""
            
            # 添加係數行
            table_rows.append({
                'Outcome': outcome.upper(),
                'Family_PSM': family_coef,
                'Gov_PSM': gov_coef
            })
            
            # 添加t值行
            table_rows.append({
                'Outcome': '',
                'Family_PSM': family_t,
                'Gov_PSM': gov_t
            })
        
        # 添加觀測數
        if outcome_vars[0] in family_results and 'n_obs' in family_results[outcome_vars[0]]:
            if outcome_vars[0] in gov_results and 'n_obs' in gov_results[outcome_vars[0]]:
                table_rows.append({
                    'Outcome': 'N',
                    'Family_PSM': f"{family_results[outcome_vars[0]]['n_obs']}",
                    'Gov_PSM': f"{gov_results[outcome_vars[0]]['n_obs']}"
                })
        
        return pd.DataFrame(table_rows)
    
    # 建立基準模型表格
    baseline_table = create_baseline_table(baseline_results['family'], baseline_results['gov'], outcome_vars_psm)
    baseline_table.to_csv("psm_results_enhanced_baseline.csv", index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("PSM 分析結果摘要（基準模型）")
    print("="*80)
    print(baseline_table)
    print("\n已輸出基準模型結果：psm_results_enhanced_baseline.csv")
    
    print("\n" + "="*80)
    print("PSM 改進分析完成！")
    print("輸出檔案：")
    print("1. 多規格比較：psm_multiple_specifications_comparison.csv")
    print("2. 基準模型結果：psm_results_enhanced_baseline.csv")
    print("\n參數調整建議：")
    print("- 使用 75 百分位切點可能會產生更極端的處理組")
    print("- Caliper 匹配可提升匹配品質")
    print("- 1:2 或更高比例匹配可增加統計檢定力")
    print("- 修剪極值可減少偏誤")
    print("="*80)
    
except ImportError as e:
    print(f"\n錯誤：缺少必要的套件 - {str(e)}")
    print("請確認已安裝所需套件：scikit-learn, scipy")
except Exception as e:
    print(f"\nPSM分析過程中發生錯誤：{str(e)}")
    import traceback
    traceback.print_exc()