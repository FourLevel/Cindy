import pandas as pd

# 讀取 CSV
file_path = 'kz.csv'
df = pd.read_csv(file_path)

# 依據 ISIN CODE 與 Year 排序
# Year 需為 int 型態
if df['Year'].dtype != int:
    df['Year'] = df['Year'].astype(int)
df = df.sort_values(['ISIN CODE', 'Year'])

# 取得前一年 TOTAL ASSETS (A_{it-1})
df['A_prev'] = df.groupby('ISIN CODE')['TOTAL ASSETS'].shift(1)

# 計算 LEV 與 Q
# LEV = TOTAL DEBT / TOTAL SHAREHOLDERS EQUITY
# Q = MARKET VALUE / TOTAL SHAREHOLDERS EQUITY
df['LEV'] = df['TOTAL DEBT'] / df['TOTAL SHAREHOLDERS EQUITY']
df['Q'] = df['MARKET VALUE'] / df['TOTAL SHAREHOLDERS EQUITY']

# 計算 KZ index
# 需處理缺失值與型態
for col in ['FREE CASH FLOW', 'CASH DIVIDENDS PAID - TOTAL', 'CASH', 'A_prev', 'LEV', 'Q']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

kz = (
    -1.002 * (df['FREE CASH FLOW'] / df['A_prev'])
    -39.638 * (df['CASH DIVIDENDS PAID - TOTAL'] / df['A_prev'])
    -1.315 * (df['CASH'] / df['A_prev'])
    +3.139 * df['LEV']
    +0.283 * df['Q']
)
df['kz'] = kz

# 儲存結果
output_path = 'kz_with_index.csv'
df.to_csv(output_path, index=False)

print(f'已完成計算，結果已儲存至 {output_path}')
