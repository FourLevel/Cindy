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

# 在這裡切分訓練集和測試集（80% 訓練，20% 測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 計算訓練集的相關係數矩陣
correlation_matrix = X_train.corr()

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

plt.title('Correlation Matrix (Training Set)')
plt.tight_layout()  # 自動調整布局
plt.show()

def try_logistic_regression_combinations(X_train, y_train):
    # 獲取所有特徵
    features = X_train.columns.tolist()
    successful_combinations = []
    
    # 嘗試所有可能的特徵組合（從5個特徵到所有特徵）
    for r in range(5, len(features) + 1):
        for combo in combinations(features, r):
            try:
                # 選擇當前組合的特徵
                X_subset = X_train[list(combo)]
                
                # 添加常數項
                X_subset = sm.add_constant(X_subset)
                
                # 進行羅吉斯迴歸
                model = sm.Logit(y_train, X_subset)
                results = model.fit(maxiter=35)  # 增加最大迭代次數
                
                # 檢查是否達到最大迭代次數
                if results.mle_retvals['warnflag'] == 1:
                    print(f"組合 {combo} 達到最大迭代次數，已排除")
                    continue
                
                # 檢查 p-values 是否包含 1.0000 或 nan
                pvalues = results.pvalues
                if any(pd.isna(pvalues)) or any(pvalues == 1.0000):
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

# 執行羅吉斯迴歸分析（僅使用訓練集）
print("\n開始嘗試不同的變數組合...")
successful_models = try_logistic_regression_combinations(X_train, y_train)

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
    results_df.to_csv('logistic_regression_results_train.csv', index=False, encoding='utf-8-sig')
    print(f"\n所有結果已儲存至 'logistic_regression_results_train.csv'")
    
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
    
    # 使用最佳模型特徵進行最終模型訓練和評估
    best_features = list(best_model['features'])
else:
    print("\n沒有找到成功的模型組合")
    # 如果沒有找到成功的模型組合，則使用預設特徵組合
    best_features = ['ROA', 'ROE', 'debt_equity_ratio', 'global_influence', 'financial_crisis', 
                    'political_factors', 'government_policy', 'industry_competition', 'poor_management', 'labor_disputes', 'safety_incidents']

## 使用最佳模型特徵訓練最終模型
# 選擇最佳特徵
model_X_train = X_train[best_features]
model_X_test = X_test[best_features]

# 添加常數項
model_X_train = sm.add_constant(model_X_train)
model_X_test = sm.add_constant(model_X_test)

# 進行羅吉斯迴歸，使用 statsmodels 進行迴歸分析
final_model = sm.Logit(y_train, model_X_train)
results = final_model.fit(maxiter=35)

# 顯示模型摘要
print("\n模型摘要：")
print(results.summary())

# 進行預測（訓練集和測試集）
train_pred_proba = results.predict(model_X_train)
test_pred_proba = results.predict(model_X_test)

# 將機率值轉換為二元預測（使用 0.5 作為閾值）
train_pred = (train_pred_proba > 0.5).astype(int)
test_pred = (test_pred_proba > 0.4).astype(int)

# 計算訓練集的 ROC 曲線
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_pred_proba)
train_roc_auc = auc(train_fpr, train_tpr)

# 計算測試集的 ROC 曲線
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_pred_proba)
test_roc_auc = auc(test_fpr, test_tpr)

# 繪製 ROC 曲線
plt.figure(figsize=(10, 8), dpi=120)
plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, 
         label=f'Training ROC curve (area = {train_roc_auc:.2f})')
plt.plot(test_fpr, test_tpr, color='green', lw=2, 
         label=f'Testing ROC curve (area = {test_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 計算訓練集的混淆矩陣
train_conf_matrix = confusion_matrix(y_train, train_pred)
print("\n訓練集混淆矩陣：")
print(train_conf_matrix)

# 計算測試集的混淆矩陣
test_conf_matrix = confusion_matrix(y_test, test_pred)
print("\n測試集混淆矩陣：")
print(test_conf_matrix)

# 繪製訓練集混淆矩陣熱圖
plt.figure(figsize=(7, 6), dpi=400)
sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Training Set Confusion Matrix')
plt.tight_layout()
plt.show()

# 繪製測試集混淆矩陣熱圖
plt.figure(figsize=(7, 6), dpi=400)
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Testing Set Confusion Matrix')
plt.tight_layout()
plt.show()

# 計算訓練集的評估指標
train_accuracy = accuracy_score(y_train, train_pred)
train_precision = precision_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

# 計算測試集的評估指標
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

# 顯示訓練集的評估指標
print("\n訓練集評估指標：")
print(f" Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"   Recall: {train_recall:.4f}")
print(f" F1 score: {train_f1:.4f}")

# 顯示測試集的評估指標
print("\n測試集評估指標：")
print(f" Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")
print(f" F1 score: {test_f1:.4f}")

# 使用公式解釋評估指標
print("\n評估指標說明：")
print(" Accuracy: 正確預測的數量 / 總預測數量")
print("Precision: 正確預測為正的數量 / 預測為正的數量")
print("   Recall: 正確預測為正的數量 / 實際為正的數量")
print(" F1 score: 2 * (Precision * Recall) / (Precision + Recall)")