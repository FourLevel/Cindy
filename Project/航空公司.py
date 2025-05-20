import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)
import statsmodels.api as sm
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
plt.figure(figsize=(16, 16), dpi=120)
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
                
                # 檢查 p-values 是否包含 1 或 nan
                pvalues = results.pvalues
                if any(pd.isna(pvalues)) or any(pvalues == 1):
                    print(f"組合 {combo} 包含無效的 p-value，已排除")
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

## 找到最佳模型
# 選擇 ROA, ROE, debt_equity_ratio, global_influence, financial_crisis, political_factors, industry_competition, internal_system, poor_management, safety_incidents 作為模型變數
model_1_X = df[['ROA', 'ROE', 'debt_equity_ratio', 'global_influence', 'financial_crisis', 
                'political_factors', 'industry_competition', 'internal_system', 
                'poor_management', 'safety_incidents']]

# 將數值型變數標準化
numeric_columns = ['ROA', 'ROE', 'debt_equity_ratio', 'global_influence', 'financial_crisis', 
                  'political_factors', 'industry_competition', 'internal_system', 
                  'poor_management', 'safety_incidents']
scaler = StandardScaler()
model_1_X[numeric_columns] = scaler.fit_transform(model_1_X[numeric_columns])

# 添加常數項
model_1_X = sm.add_constant(model_1_X)

model_1_y = df['bankruptcy_status']

# 進行羅吉斯迴歸，使用 statsmodels 進行迴歸分析
model_1 = sm.Logit(model_1_y, model_1_X)
results = model_1.fit(maxiter=35)

# 顯示模型摘要
print("\n模型摘要：")
print(results.summary())

# 進行預測
pred_proba = results.predict(model_1_X)

# 將機率值轉換為二元預測（使用 0.5 作為閾值）
pred = (pred_proba > 0.5).astype(int)

# 計算 ROC 曲線
fpr, tpr, thresholds = roc_curve(model_1_y, pred_proba)
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
plt.figure(figsize=(10, 8), dpi=120)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 計算混淆矩陣
conf_matrix = confusion_matrix(model_1_y, pred)
print("\n混淆矩陣：")
print(conf_matrix)

# 繪製混淆矩陣熱圖
plt.figure(figsize=(7, 6), dpi=400)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 計算評估指標
accuracy = accuracy_score(model_1_y, pred)
precision = precision_score(model_1_y, pred)
recall = recall_score(model_1_y, pred)
f1 = f1_score(model_1_y, pred)

# 顯示評估指標
print("\n模型評估指標：")
print(f" Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f" F1 score: {f1:.4f}")

# 使用公式解釋評估指標
print("\n評估指標說明：")
print(" Accuracy: 正確預測的數量 / 總預測數量")
print("Precision: 正確預測為正的數量 / 預測為正的數量")
print("   Recall: 正確預測為正的數量 / 實際為正的數量")
print(" F1 score: 2 * (Precision * Recall) / (Precision + Recall)")