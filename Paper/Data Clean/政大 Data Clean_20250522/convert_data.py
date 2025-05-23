import pandas as pd
import os

# 讀取檔案
raw_data = pd.read_csv('Raw Data.csv')
company_list = pd.read_csv('Company List.csv')

# 1. 改善分離公司名稱與財務變數的方式
# 先建立公司名稱清單，並處理可能的格式差異，同時包含 NAME 1 和 NAME 2
company_names = []
for index, row in company_list.iterrows():
    # 加入 NAME 1 欄位（如果存在且不為空）
    if 'NAME 1' in company_list.columns and pd.notna(row['NAME 1']):
        company_names.append(str(row['NAME 1']).strip())
    # 加入 NAME 2 欄位（如果存在且不為空）
    if 'NAME 2' in company_list.columns and pd.notna(row['NAME 2']):
        company_names.append(str(row['NAME 2']).strip())

# 移除重複的公司名稱
company_names = list(set(company_names))

# 創建結果列表以存儲所有的資料列
result_rows = []

# 定義一個函數來更好地分離公司名稱和變數
def extract_company_and_variable(name_string, company_names):
    """
    更安全地分離公司名稱和財務變數
    透過已知的公司清單來比對，而不是單純分割
    支援同時比對 NAME 1 和 NAME 2 欄位
    """
    if pd.isna(name_string):
        return "", ""
    
    name_string = str(name_string).strip()
    
    # 嘗試找到最長的匹配公司名稱
    best_match = None
    best_length = 0
    
    for company in company_names:
        if pd.isna(company):
            continue
        company_clean = str(company).strip()
        # 檢查是否以該公司名稱開頭
        if name_string.startswith(company_clean):
            if len(company_clean) > best_length:
                best_match = company_clean
                best_length = len(company_clean)
    
    if best_match:
        # 找到匹配的公司名稱，提取剩餘的部分作為變數名稱
        remainder = name_string[len(best_match):].strip()
        if remainder.startswith(' - '):
            variable = remainder[3:].strip()  # 移除 " - " 前綴
            return best_match, variable
        elif remainder.startswith('-'):
            variable = remainder[1:].strip()  # 移除 "-" 前綴
            return best_match, variable
    
    # 如果沒有找到匹配，回退到原來的分割方式
    if ' - ' in name_string:
        parts = name_string.split(' - ', 1)
        return parts[0].strip(), parts[1].strip()
    
    # 最後的備用方案
    return name_string, ""

# 為 raw_data 添加分離後的欄位
companies = []
variables = []

for name in raw_data['Name']:
    company, variable = extract_company_and_variable(name, company_names)
    companies.append(company)
    variables.append(variable)

raw_data['Company'] = companies
raw_data['Variable'] = variables

# 2. 整理年份欄位 (2014-2024)
years = [str(year) for year in range(2014, 2025)]

# 遍歷每一筆原始資料
for index, row in raw_data.iterrows():
    company_name = row['Company']
    
    # 在 Company List 中查找對應的 No 和 ISIN CODE
    # 同時比對 NAME 1 和 NAME 2 欄位
    company_info = pd.DataFrame()
    
    # 檢查 NAME 1 欄位
    if 'NAME 1' in company_list.columns:
        name1_match = company_list[company_list['NAME 1'].str.strip() == company_name.strip()]
        if len(name1_match) > 0:
            company_info = name1_match
    
    # 如果 NAME 1 沒有匹配到，再檢查 NAME 2 欄位
    if len(company_info) == 0 and 'NAME 2' in company_list.columns:
        name2_match = company_list[company_list['NAME 2'].str.strip() == company_name.strip()]
        if len(name2_match) > 0:
            company_info = name2_match
    
    if len(company_info) == 0:
        # 過濾掉 #ERROR 的警告訊息，因為這代表原始資料中該公司沒有該項資料
        if company_name != '#ERROR':
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