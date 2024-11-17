# 載入需要的套件
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr
import statsmodels.api as sm
from openpyxl import load_workbook
# 讀取 excel 檔案
df = pd.read_excel('Data_Value_20241115.xlsx', sheet_name='ESG+FIN_Value_20241115')

# 針對每一個 No，將 Property, Plant & Equip 往前推一年
df['PROPERTY, PLANT & EQUIP - NET t-1'] = df.groupby('No')['PROPERTY, PLANT & EQUIP - NET'].shift(1)

# 將年份變成 Dummy Variable
df['Year'] = df['Year'].astype(str)
year_dummies = pd.get_dummies(df['Year'], drop_first=True)
df = pd.concat([df, year_dummies], axis=1)

# 將國家變成 Dummy Variable
df['Country'] = df['Country'].astype(str)
country_dummies = pd.get_dummies(df['Country'], drop_first=True)
df = pd.concat([df, country_dummies], axis=1)

# 檢查每一欄位缺失值的比例
missing_percentage = df.isnull().sum() / len(df) * 100
print(missing_percentage)

# 刪除 missing percentage 大於 21% 的欄位
df.dropna(axis=1, thresh=len(df) * 0.79, inplace=True)

# 刪除 missing value
df.dropna(inplace=True)

# 列出所有變數名稱
variables = df.columns.tolist()
for var in variables:
    print(var)

# 檢查 missing value
print(df.isnull().sum())

# 計算 KZ index 的函數
def calculate_kz_index(df):
    # Cash Flows
    cash_flows = df['FREE CASH FLOW']
    
    # K = Property, Plant, and Equipment
    K = df['PROPERTY, PLANT & EQUIP - NET t-1']
    
    # Q = (Market cap / Total shareholder's equity)
    Q = df['MARKET VALUE'] / df['TOTAL SHAREHOLDERS EQUITY']
    
    # Debt
    debt = df['TOTAL DEBT']
    
    # Total capital 
    total_capital = df['TOTAL CAPITAL']
    
    # Dividends
    dividends = df['CASH DIVIDENDS PAID - TOTAL']
    
    # Cash
    cash = df['CASH - GENERIC']
    
    # 計算 KZ index
    KZ_index = (-1.001909 * cash_flows / K +
                0.2826389 * Q +
                3.139193 * debt / total_capital +
                -39.3678 * dividends / K +
                -1.314759 * cash / K)
    
    return KZ_index

# 將計算結果加入 DataFrame
df['KZ Index'] = calculate_kz_index(df)

# 計算 Altman Z-Score 的函數
def calculate_altman_z_score(df):
    # A = Working Capital / Total Assets
    A = df['WORKING CAPITAL'] / df['TOTAL ASSETS']

    # B = Retained Earnings / Total Assets
    B = df['RETAINED EARNINGS'] / df['TOTAL ASSETS']
    
    # C = EBIT / Total Assets
    C = df['EARNINGS BEF INTEREST & TAXES'] / df['TOTAL ASSETS']
    
    # D = Market Value of Equity / Total Liabilities
    D = df['MARKET VALUE'] / df['TOTAL LIABILITIES']
    
    # E = Sales / Total Assets
    E = df['NET SALES OR REVENUES'] / df['TOTAL ASSETS']
    
    # 計算 Altman Z-Score (暫時不包含 A 項)
    Z_score = (1.2 * A + 
               1.4 * B + 
               3.3 * C + 
               0.6 * D + 
               1.0 * E)
    
    return Z_score

# 計算 Z-Score 並加入到原始 DataFrame
df['Altman_Z_Score'] = calculate_altman_z_score(df)

# 將 df 中的 KZ Index 設為 變數 y，其他變數設為變數 X
y_kz = df['KZ Index']
y_altman = df['Altman_Z_Score']
X_quantitative = df.drop(columns=['KZ Index', 'Altman_Z_Score', 'No', 'ISIN CODE', 'Year', 'Country', 'State Owned Enterprise (SOE)', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'INDONESIA', 'JAPAN', 'MALAYSIA', 'PHILIPPINES', 'SINGAPORE', 'SOUTH KOREA', 'TAIWAN', 'THAILAND'])
X_dummy = df[['State Owned Enterprise (SOE)', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'INDONESIA', 'JAPAN', 'MALAYSIA', 'PHILIPPINES', 'SINGAPORE', 'SOUTH KOREA', 'TAIWAN', 'THAILAND']]

# 將 X 變數進行標準化，以 std_ 開頭，另存成 std_X，並新增到 X 中
std_X_quantitative = (X_quantitative - X_quantitative.mean()) / X_quantitative.std()
std_X_quantitative.columns = ['std_' + col for col in std_X_quantitative.columns]
X_quantitative = pd.concat([X_quantitative, std_X_quantitative], axis=1)

# 將 X 合併
X = pd.concat([X_quantitative, X_dummy], axis=1)

# 將每一個 X 變數與 y_kz 和 y_altman 做迴歸分析，並分別整理成 dataframe
p_value_kz = []
p_value_altman = []
for var in X.columns:
    p_value_kz.append(pearsonr(X[var], y_kz)[1])
    p_value_altman.append(pearsonr(X[var], y_altman)[1])
p_value_kz_df = pd.DataFrame({'Variable': X.columns, 'p_value_kz': p_value_kz})
p_value_altman_df = pd.DataFrame({'Variable': X.columns, 'p_value_altman': p_value_altman})
print(p_value_kz_df)
print(p_value_altman_df)

# 計算 X 的相關係數矩陣，並匯出
corr_matrix = X_quantitative.corr()
# corr_matrix.to_excel('corr_matrix.xlsx', sheet_name='New')

# 設定要留下的 X 量化變數
X_quantitative_to_keep = ['std_Environment Pillar Score', 'std_Social Pillar Score', 'std_Governance Pillar Score', 
                          'std_Total CO2 Equivalent Emissions To Revenues USD in millions', 'std_CO2 Equivalents Emission Total', 
                          'std_Value - Board Structure/Board Diversity', 'std_Value - Compensation Policy/Board Member Compensation', 
                          'std_INCREASE/DECREASE IN CASH/SHOR', 'std_TOTAL DEBT', 'std_MARKET VALUE', 'std_WORKING CAPITAL', 'std_BOOK VALUE PER SHARE']
X_quantitative_selected = X_quantitative[X_quantitative_to_keep]

# 計算 X 的 VIF，不要用科學記號
pd.set_option('display.float_format', lambda x: '%.4f' % x)
vif = pd.DataFrame()
vif["features"] = X_quantitative_selected.columns
vif["VIF Factor"] = [variance_inflation_factor(X_quantitative_selected.values, i) for i in range(X_quantitative_selected.shape[1])]
print(vif)

# 設定要留下的 X 虛擬變數
X_dummy_to_keep = ['State Owned Enterprise (SOE)', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 
                   'INDONESIA', 'JAPAN', 'MALAYSIA', 'PHILIPPINES', 'SINGAPORE', 'SOUTH KOREA', 'TAIWAN', 'THAILAND']
X_dummy_selected = X_dummy[X_dummy_to_keep]

# 合併 X 變數
X_selected = pd.concat([X_quantitative_selected, X_dummy_selected], axis=1)

# 確保所有數據都是數值型
X_selected = X_selected.astype(float)

# 進行迴歸分析
X_selected = sm.add_constant(X_selected)
model_kz = sm.OLS(y_kz, X_selected).fit()
model_altman = sm.OLS(y_altman, X_selected).fit()
print(model_kz.summary())
print(model_altman.summary())


# 保留 model_altman 的結果
# 將模型內變數之 Coefficient, Std. Error, p-value, VIF 保存成 dataframe
coefficients = model_altman.params
std_errors = model_altman.bse
p_values = model_altman.pvalues

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_selected.columns
vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]

# Merge results
results_df = pd.DataFrame({
    "Coefficient": coefficients,
    "Std. Error": std_errors,
    "p-value": p_values
}).reset_index().rename(columns={"index": "Variable"})

# Merge VIF
results_df = results_df.merge(vif_data, on="Variable")

# Round values to four decimal places
results_df = results_df.round(4)

# Add stars based on significance
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''

results_df['p-value'] = results_df['p-value'].apply(lambda x: f"{x:.4f}{significance_stars(x)}")

# Print the table and explain the stars
print(results_df.to_markdown(index=False))
print("\nExplanation of the stars：")
print("***：Significant at the 0.1% level")
print("**：Significant at the 1% level")
print("*：Significant at the 5% level")
print(".: Significant at the 10% level")

# 將 results_df 存成 excel 檔案
results_df.to_excel('results_df_altman.xlsx', index=False)


# 保留 model_kz 的結果
# 將模型內變數之 Coefficient, Std. Error, p-value, VIF 保存成 dataframe
coefficients = model_kz.params
std_errors = model_kz.bse
p_values = model_kz.pvalues

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_selected.columns
vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]

# Merge results
results_df = pd.DataFrame({
    "Coefficient": coefficients,
    "Std. Error": std_errors,
    "p-value": p_values
}).reset_index().rename(columns={"index": "Variable"})

# Merge VIF
results_df = results_df.merge(vif_data, on="Variable")

# Round values to four decimal places
results_df = results_df.round(4)

# Add stars based on significance
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''

results_df['p-value'] = results_df['p-value'].apply(lambda x: f"{x:.4f}{significance_stars(x)}")

# Print the table and explain the stars
print(results_df.to_markdown(index=False))
print("\nExplanation of the stars：")
print("***：Significant at the 0.1% level")
print("**：Significant at the 1% level")
print("*：Significant at the 5% level")
print(".: Significant at the 10% level")

# 將 results_df 存成 excel 檔案
results_df.to_excel('results_df_kz.xlsx', index=False)