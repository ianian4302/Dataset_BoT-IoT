import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. 載入資料
DATA_PATH = os.path.join(os.getcwd(), 'data', 'bot-iot.csv')
df = pd.read_csv(DATA_PATH)

# 2. 特徵工程：移除高基數特徵、標籤編碼、one-hot
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
high_card_cols = [col for col in cat_cols if df[col].nunique() > 100]
for col in high_card_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
low_card_cols = [col for col in cat_cols if col not in high_card_cols]
if low_card_cols:
    df = pd.get_dummies(df, columns=low_card_cols)

df = df.fillna(0)

# 3. 標籤選擇與資料不平衡處理
if 'attack' in df.columns:
    target_col = 'attack'
elif 'label' in df.columns:
    target_col = 'label'
elif 'category' in df.columns:
    target_col = 'category'
else:
    target_col = df.columns[-1]

# 4. 檢查不平衡，繪製原始分布
plt.figure(figsize=(8,4))
df[target_col].value_counts().plot(kind='bar')
plt.title('Original Label Distribution')
plt.ylabel('Count')
plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig('output/original_label_distribution.png')
plt.close()

# 5. SMOTE 平衡資料
X = df.drop(columns=[target_col])
y = df[target_col]
if y.nunique() > 2:
    le = LabelEncoder()
    y = le.fit_transform(y)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

plt.figure(figsize=(8,4))
pd.Series(y_bal).value_counts().plot(kind='bar')
plt.title('Balanced Label Distribution (SMOTE)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/balanced_label_distribution.png')
plt.close()

# 6. 訓練/測試分割
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. 機器學習模型訓練與評估
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': acc,
        'report': classification_report(y_test, y_pred, output_dict=True),
        'cm': confusion_matrix(y_test, y_pred)
    }

# 8. 準確率比較圖
plt.figure(figsize=(6,4))
plt.bar(results.keys(), [results[m]['accuracy'] for m in results], color=['#4C72B0', '#55A868'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.savefig('output/model_accuracy_comparison_app1.png')
plt.close()

# 9. 混淆矩陣圖
for name in results:
    plt.figure(figsize=(5,4))
    sns.heatmap(results[name]['cm'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'output/confusion_matrix_{name.replace(" ", "_").lower()}_app1.png')
    plt.close()

# 10. 特徵重要性（僅隨機森林）
if 'Random Forest' in models:
    importances = models['Random Forest'].feature_importances_
    indices = np.argsort(importances)[-15:][::-1]
    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('output/feature_importance_rf_app1.png')
    plt.close()

print('All analysis and comparison images saved in output/.')
