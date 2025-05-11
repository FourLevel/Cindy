import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
from scipy import stats
from itertools import combinations

# 讀取 CSV 檔案
df = pd.read_csv('航空公司量化資料.csv')

# 定義欄位名稱對照表
column_mapping = {
    'number': 'number',
    'country': 'country',
    'year': 'year',
    'ROA': 'ROA',
    'ROE': 'ROE',
    'EPS': 'EPS',
    'debt/equity': 'debt_equity_ratio',
    '營業中': 'operating_status',
    '破產&破產保護': 'bankruptcy_status',
    '全球影響力': 'global_influence',
    '金融危機': 'financial_crisis',
    '油價': 'oil_price',
    '政治': 'political_factors',
    '疫情': 'pandemic',
    '政府政策': 'government_policy',
    '同業競爭': 'industry_competition',
    '內部制度': 'internal_system',
    '資金週轉不靈': 'cash_flow_problems',
    '管理決策不佳': 'poor_management',
    '勞資雙方不合': 'labor_disputes',
    '公會': 'union',
    '管理層缺乏應變能力': 'management_adaptability',
    '飛安事故數量': 'safety_incidents'
}

# 重命名欄位
df = df.rename(columns=column_mapping)

# 顯示資料基本資訊
print("\n資料基本資訊：")
print(df.info())

# 移除空值
df = df.dropna()

# 顯示移除空值後資料基本資訊
print("\n移除空值後資料基本資訊：")
print(df.info())

# 設定 y 為 bankruptcy_status
y = df['bankruptcy_status']

# 設定 X 為其他特徵，移除高相關性變數
X = df.drop(columns=['number', 'country', 'year', 'bankruptcy_status', 'union', 'management_adaptability'])

# 將數值型變數標準化
numeric_columns = ['ROA', 'ROE', 'EPS', 'debt_equity_ratio']
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# 計算 x 的相關係數矩陣
correlation_matrix = X.corr()

# 找出相關係數絕對值大於 0.7 的變數組合
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Variable1': correlation_matrix.columns[i],
                'Variable2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

# 將結果轉換為 DataFrame 並排序
high_corr_df = pd.DataFrame(high_corr_pairs)
if not high_corr_df.empty:
    high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
    print("\n相關係數絕對值大於 0.7 的變數組合：")
    print(high_corr_df)
else:
    print("\n沒有相關係數絕對值大於 0.7 的變數組合")

# 繪製相關係數矩陣熱圖
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, 
            annot=True,  # 顯示數值
            cmap='coolwarm',  # 使用藍紅配色
            center=0,  # 將顏色中心設為0
            fmt='.2f',  # 數值格式為兩位小數
            square=True)  # 保持正方形

plt.title('Correlation Matrix')
plt.tight_layout()  # 自動調整布局
plt.show()

def try_logistic_regression_combinations(X, y):
    # 獲取所有特徵
    features = X.columns.tolist()
    successful_combinations = []
    
    # 嘗試所有可能的特徵組合（從5個特徵到所有特徵）
    for r in range(5, len(features) + 1):
        for combo in combinations(features, r):
            try:
                # 選擇當前組合的特徵
                X_subset = X[list(combo)]
                
                # 添加常數項
                X_subset = sm.add_constant(X_subset)
                
                # 進行羅吉斯迴歸
                model = sm.Logit(y, X_subset)
                results = model.fit(maxiter=35)  # 增加最大迭代次數
                
                # 檢查是否達到最大迭代次數
                if results.mle_retvals['warnflag'] == 1:
                    print(f"組合 {combo} 達到最大迭代次數，已排除")
                    continue
                
                # 如果成功運行，記錄這個組合
                successful_combinations.append({
                    'features': combo,
                    'aic': results.aic,
                    'bic': results.bic,
                    'llf': results.llf,
                    'pvalues': results.pvalues,
                    'params': results.params,
                    'iterations': results.mle_retvals['iterations']
                })
                
            except Exception as e:
                continue
    
    return successful_combinations

# 執行羅吉斯迴歸分析
print("\n開始嘗試不同的變數組合...")
successful_models = try_logistic_regression_combinations(X, y)

# 根據 AIC 排序結果
if successful_models:
    successful_models.sort(key=lambda x: x['aic'])
    
    # 準備儲存到 CSV 的資料
    csv_data = []
    for model in successful_models:
        # 將特徵組合轉換為字串
        features_str = ', '.join(model['features'])
        
        # 將係數和P值轉換為字串
        coef_str = ', '.join([f"{var}: {coef:.4f}" for var, coef in model['params'].items()])
        pval_str = ', '.join([f"{var}: {pval:.4f}" for var, pval in model['pvalues'].items()])
        
        csv_data.append({
            '特徵組合': features_str,
            '特徵數量': len(model['features']),
            'AIC': model['aic'],
            'BIC': model['bic'],
            'Log-Likelihood': model['llf'],
            '迭代次數': model['iterations'],
            '係數': coef_str,
            'P值': pval_str
        })
    
    # 建立 DataFrame 並儲存為 CSV
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv('logistic_regression_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n所有結果已儲存至 'logistic_regression_results.csv'")
    
    # 顯示前10個最佳模型
    print("\n前10個最佳模型（根據 AIC）：")
    print("-" * 80)
    print(f"{'排名':<5}{'AIC':<10}{'BIC':<10}{'特徵數量':<10}{'迭代次數':<10}{'特徵組合'}")
    print("-" * 80)
    
    for i, model in enumerate(successful_models[:10], 1):
        features = model['features']
        print(f"{i:<5}{model['aic']:<10.2f}{model['bic']:<10.2f}{len(features):<10}{model['iterations']:<10}{features}")
    
    # 顯示最佳模型的詳細結果
    best_model = successful_models[0]
    print("\n最佳模型的詳細結果：")
    print("-" * 80)
    print(f"特徵組合: {best_model['features']}")
    print(f"AIC: {best_model['aic']:.2f}")
    print(f"BIC: {best_model['bic']:.2f}")
    print(f"Log-Likelihood: {best_model['llf']:.2f}")
    print(f"迭代次數: {best_model['iterations']}")
    
    print("\n係數和P值：")
    print("-" * 80)
    print(f"{'變數':<30}{'係數':<15}{'P值':<15}")
    print("-" * 80)
    for var, coef in best_model['params'].items():
        pval = best_model['pvalues'][var]
        print(f"{var:<30}{coef:<15.4f}{pval:<15.4f}")
else:
    print("\n沒有找到成功的模型組合")