# 匯入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 匯入檔案
raw_data = pd.read_csv('Raw Data.csv')
company_list = pd.read_csv('Company List.csv')
sample_data = pd.read_csv('Sample Data.csv')

# 查看資料
print(raw_data.head())
print(company_list.head())
print(sample_data.head())






