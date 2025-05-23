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
    
    # 定義三個模型的變數
    models = {
        "Model 1": {
            "Y": "gap",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        },
        "Model 2": {
            "Y": "gap_e",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        },
        "Model 3": {
            "Y": "gap_s",
            "X": ["family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"] + \
                 list(year_dummies.columns) + list(industry_dummies.columns)
        }
    }
    
    # 整理成表格格式（每個變數兩列：一列係數+星號，一列t值）
    var_order = [
        "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov", "_cons"
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
        "boardSize": "Board Size",
        "CEOdual": "CEO Duality",
        "CSRcmte": "CSR Committee",
        "g_family": "G*Family",
        "g_gov": "G*Gov",
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
    regression_table.to_csv("regression_table.csv", index=False)
    print("\n已輸出整理後的迴歸表格 regression_table.csv（t值在係數下一列）。")


## 進行 2SLS 迴歸分析
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
    print("\n工具變數檢查通過，開始進行2SLS分析...")
    
    # 確保工具變數為數值型
    for var in iv_vars:
        df_with_dummies[var] = pd.to_numeric(df_with_dummies[var], errors='coerce')
    
    # 移除包含NaN的行
    required_vars_2sls = iv_vars + ["gap", "gap_e", "gap_s", "family", "gov", "g", "size", "lev", "roa", "mtb", "kz", "boardSize", "CEOdual", "CSRcmte", "g_family", "g_gov"]
    df_2sls = df_with_dummies.dropna(subset=required_vars_2sls)
    print(f"2SLS分析樣本數：{len(df_2sls)}")
    
    try:
        # 導入2SLS所需的套件
        from linearmodels import IV2SLS
        import numpy as np
        
        # 重新計算交互項（因為可能有資料被移除）
        df_2sls['g_family'] = df_2sls['g'] * df_2sls['family']
        df_2sls['g_gov'] = df_2sls['g'] * df_2sls['gov']
        
        # 定義基本控制變數
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
        df_results_2sls.to_csv("iv_2sls_results_table.csv", index=False)
        print("\n已輸出2SLS分析結果表格：iv_2sls_results_table.csv")
        
        # 顯示第一階段F統計量
        print(f"\n第一階段 F-statistic: {first_stage_model.fvalue:.2f}")
        
        # 顯示工具變數相關係數
        print("\n工具變數係數:")
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
        
    except ImportError:
        print("\n錯誤：需要安裝 linearmodels 套件來進行2SLS分析")
        print("請執行：pip install linearmodels")
    except Exception as e:
        print(f"\n2SLS分析過程中發生錯誤：{str(e)}")