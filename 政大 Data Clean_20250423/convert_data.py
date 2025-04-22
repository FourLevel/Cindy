import pandas as pd
import os

# 讀取檔案
raw_data = pd.read_csv('Raw Data.csv')
company_list = pd.read_csv('Company List.csv')

# 1. 分離公司名稱與財務變數
raw_data[['Company', 'Variable']] = raw_data['Name'].str.split(' - ', n=1, expand=True)

# 2. 整理年份欄位 (2018-2025)
years = [str(year) for year in range(2018, 2026)]

# 創建結果列表以存儲所有的資料列
result_rows = []

# 遍歷每一筆原始資料
for index, row in raw_data.iterrows():
    company_name = row['Company']
    
    # 在 Company List 中查找對應的 No 和 ISIN CODE
    company_info = company_list[company_list['NAME'].str.strip() == company_name.strip()]
    
    if len(company_info) == 0:
        print(f"警告: 找不到公司 '{company_name}' 的資訊")
        continue
    
    company_no = company_info['No'].values[0]
    company_isin = company_info['ISIN CODE'].values[0]
    variable_name = row['Variable']
    
    # 對每個年份的數據建立一筆新記錄
    for year in years:
        if pd.isna(row[year]) or row[year] == 'NA':
            continue
            
        # 創建包含變數值的記錄
        new_row = {
            'No': company_no,
            'ISIN CODE': company_isin,
            'Year': year,
            'Variable': variable_name,
            'Value': row[year]
        }
        
        result_rows.append(new_row)

# 將結果列表轉換為 DataFrame
result_data = pd.DataFrame(result_rows)

# 使用 pivot_table 將數據從長格式轉換為寬格式
# 其中 No、ISIN CODE 和 Year 作為索引，Variable 作為列名，Value 作為值
result_pivot = result_data.pivot_table(
    index=['No', 'ISIN CODE', 'Year'],
    columns='Variable',
    values='Value',
    aggfunc='first'
).reset_index()

# 讓列名稱更加整潔（移除 MultiIndex）
result_pivot.columns.name = None

# 儲存結果
result_pivot.to_csv('Converted Data.csv', index=False)

print("資料轉換完成，已儲存為 'Converted Data.csv'") 