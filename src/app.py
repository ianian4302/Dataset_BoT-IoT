import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(file_path):
    """載入並探索 Bot-IoT 資料集"""
    print(f"載入資料: {file_path}")
    df = pd.read_csv(file_path)
    
    print("\n== 資料集基本信息 ==")
    print(f"資料形狀: {df.shape}")
    print("\n前5筆資料:")
    print(df.head())
    
    print("\n資料類型:")
    print(df.dtypes)
    
    print("\n描述性統計:")
    print(df.describe())
    
    print("\n檢查缺失值:")
    print(df.isnull().sum())
    
    # 檢查標籤分佈
    if 'attack' in df.columns:
        target_col = 'attack'
    elif 'label' in df.columns:
        target_col = 'label'
    elif 'category' in df.columns:
        target_col = 'category'
    else:
        target_col = df.columns[-1]
        print(f"假設最後一欄 '{target_col}' 為標籤欄位")
    
    print(f"\n標籤 '{target_col}' 分佈:")
    print(df[target_col].value_counts())
    
    return df, target_col

def preprocess_data(df, target_col):
    """資料預處理"""
    print("\n== 資料預處理 ==")
    # 分離特徵與標籤
    # 確保 target_col 及所有標籤相關欄位不在 X
    exclude_cols = set([target_col, 'category', 'subcategory', 'label', 'attack'])
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    X = df.drop(columns=exclude_cols)
    y = df[target_col]
    # 處理類別型特徵
    cat_cols = X.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        print(f"處理類別型特徵: {cat_cols.tolist()}")
        # 識別高基數特徵 (例如IP地址、連接埠)
        high_cardinality_cols = []
        for col in cat_cols:
            unique_count = X[col].nunique()
            print(f"  - {col}: {unique_count} 個唯一值")
            if unique_count > 100:  # 閾值可調整
                high_cardinality_cols.append(col)
        if high_cardinality_cols:
            print(f"\n高基數特徵 (跳過one-hot編碼): {high_cardinality_cols}")
            from sklearn.preprocessing import LabelEncoder
            for col in high_cardinality_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            low_cardinality_cols = [col for col in cat_cols if col not in high_cardinality_cols]
            if low_cardinality_cols:
                print(f"執行one-hot編碼: {low_cardinality_cols}")
                X = pd.get_dummies(X, columns=low_cardinality_cols)
        else:
            X = pd.get_dummies(X, columns=cat_cols)
    # one-hot後補齊所有NaN為0，避免後續模型報錯
    X = X.fillna(0)
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"訓練集: {X_train.shape[0]} 筆資料, 測試集: {X_test.shape[0]} 筆資料")
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """訓練模型並評估效能 (同時比較隨機森林與邏輯斯迴歸)"""
    from sklearn.linear_model import LogisticRegression
    results = {}
    # 隨機森林
    print("\n== 隨機森林訓練與評估 ==")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"隨機森林 準確率: {accuracy_rf:.4f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred_rf))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    feature_importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    print("\n前10個最重要的特徵:")
    print(feature_importances.sort_values(ascending=False).head(10))
    results['RandomForest'] = {
        'model': rf_model,
        'y_pred': y_pred_rf,
        'cm': cm_rf,
        'feature_importances': feature_importances,
        'accuracy': accuracy_rf
    }
    # 邏輯斯迴歸
    print("\n== 邏輯斯迴歸訓練與評估 ==")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"邏輯斯迴歸 準確率: {accuracy_lr:.4f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred_lr))
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    results['LogisticRegression'] = {
        'model': lr_model,
        'y_pred': y_pred_lr,
        'cm': cm_lr,
        'feature_importances': None,
        'accuracy': accuracy_lr
    }
    return results

def visualize_results(feature_importances, cm, y_test, y_pred):
    """視覺化結果"""
    print("\n== 視覺化結果 ==")
    
    # 建立輸出目錄
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 特徵重要性視覺化
    plt.figure(figsize=(12, 8))
    top_features = feature_importances.sort_values(ascending=False).head(15)
    top_features.plot(kind='barh')
    plt.title('Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    # 2. 混淆矩陣視覺化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. 儲存預測結果
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_path = os.path.join(output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to: {output_dir}")
    return output_dir

def main():
    """主程序"""
    print("======= Bot-IoT 資料集分析與模型訓練 (多資料量比較) =======")
    
    # 讓用戶指定資料集路徑
    default_path = os.path.join(os.getcwd(), 'data', 'bot-iot.csv')
    # file_path = input(f"請輸入資料集路徑 (預設: {default_path}): ")
    # 資料集路徑顯示
    print(f"預設資料集路徑: {default_path}")
    file_path = default_path  # 預設路徑
    if not file_path:
        file_path = default_path
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 '{file_path}'")
        file_path = input("請重新輸入正確的資料集路徑: ")
        if not os.path.exists(file_path):
            print("檔案路徑無效，程式結束。")
            return
    try:
        # 載入完整資料
        df, target_col = load_and_explore_data(file_path)
        
        # 設定不同資料量
        sample_sizes = [1000, 10000, 100000]
        results = []
        for size in sample_sizes:
            print(f"\n===== 使用 {size} 筆資料進行訓練與測試 =====")
            if len(df) < size:
                print(f"資料集不足 {size} 筆，僅使用全部資料 ({len(df)})")
                df_sample = df.copy()
            else:
                df_sample = df.sample(n=size, random_state=42)
            # 預處理
            X_train, X_test, y_train, y_test, feature_names = preprocess_data(df_sample, target_col)
            # 訓練與評估 (同時比較兩種模型)
            model_results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
            # 隨機森林視覺化
            output_dir = visualize_results(
                model_results['RandomForest']['feature_importances'],
                model_results['RandomForest']['cm'],
                y_test, model_results['RandomForest']['y_pred'])
            # 收集結果
            results.append({
                'size': len(df_sample),
                'rf_accuracy': model_results['RandomForest']['accuracy'],
                'lr_accuracy': model_results['LogisticRegression']['accuracy'],
                'output_dir': output_dir
            })
        # 彙整比較結果
        print("\n===== 不同資料量比較結果 =====")
        sizes = [res['size'] for res in results]
        rf_accuracies = [res['rf_accuracy'] for res in results]
        lr_accuracies = [res['lr_accuracy'] for res in results]
        for res in results:
            print(f"資料量: {res['size']}")
            print(f"  隨機森林 準確率: {res['rf_accuracy']:.4f}")
            print(f"  邏輯斯迴歸 準確率: {res['lr_accuracy']:.4f}")
            print(f"  輸出目錄: {res['output_dir']}")
            print("-----------------------------")
        # 新增：繪製比較圖表
        plt.figure(figsize=(8, 6))
        plt.plot(sizes, rf_accuracies, marker='o', label='Random Forest')
        plt.plot(sizes, lr_accuracies, marker='s', label='Logistic Regression')
        plt.xscale('log')
        plt.xlabel('Sample Size (log scale)')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison by Sample Size')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison_en.png'))
        plt.close()
        print("\nComparison chart saved as: output/model_accuracy_comparison_en.png")
        print("\nAll sample size comparisons completed!")
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()