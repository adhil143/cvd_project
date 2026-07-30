import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, RFE
from xgboost import XGBClassifier
import shap
from sklearn.metrics import confusion_matrix, matthews_corrcoef

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def train_model():
    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found in the current directory.")
        return

    print("Step 1: Loading dataset heart.csv...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # 1. Separate features and target
    X_raw = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # 2. One-hot encode categorical columns
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    print(f"Encoded dataset shape: {X_encoded.shape}")

    # 3. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Scale features
    print("Step 2: Preprocessing and scaling features...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 5. Ensemble Feature Selection
    print("Step 3: Calculating ensemble feature selection ranks...")
    feature_names = X_train_scaled.columns.tolist()
    ranks = pd.DataFrame(index=feature_names)

    # Method 1: F-test
    f_scores, _ = f_classif(X_train_scaled, y_train)
    ranks['F_test'] = pd.Series(f_scores, index=feature_names).rank(ascending=False)

    # Method 2: L1 Logistic Regression
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE)
    l1_model.fit(X_train_scaled, y_train)
    ranks['L1_Logistic'] = pd.Series(np.abs(l1_model.coef_[0]), index=feature_names).rank(ascending=False)

    # Method 3: XGBoost Importance
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)
    xgb_base.fit(X_train_scaled, y_train)
    ranks['XGBoost'] = pd.Series(xgb_base.feature_importances_, index=feature_names).rank(ascending=False)

    # Method 4: RFE (using Logistic Regression as base)
    rfe = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), n_features_to_select=1)
    rfe.fit(X_train_scaled, y_train)
    ranks['RFE'] = pd.Series(rfe.ranking_, index=feature_names)

    # Method 5: SHAP (using TreeExplainer on base XGBoost)
    explainer_base = shap.TreeExplainer(xgb_base)
    shap_values_base = explainer_base.shap_values(X_train_scaled)
    shap_importance = np.abs(shap_values_base).mean(axis=0)
    ranks['SHAP'] = pd.Series(shap_importance, index=feature_names).rank(ascending=False)

    # Calculate average rank and pick top 5
    ranks['Average_Rank'] = ranks.mean(axis=1)
    ranks_sorted = ranks.sort_values('Average_Rank')
    
    top_5_features = ranks_sorted.head(5).index.tolist()
    print("Top 5 selected features:")
    for f in top_5_features:
        print(f" - {f}")

    # 6. Train Final XGBoost Model on Top-5 features
    print("Step 4: Training final XGBoost model on selected features...")
    X_train_top5 = X_train_scaled[top_5_features]
    X_test_top5 = X_test_scaled[top_5_features]

    final_model = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)
    final_model.fit(X_train_top5, y_train)

    # Predict probas on test set
    y_pred_proba = final_model.predict_proba(X_test_top5)[:, 1]

    # 7. Threshold Tuning
    print("Step 5: Tuning risk decision threshold via MCC...")
    thresholds = np.linspace(0, 1, 100)
    mccs = []
    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        mccs.append(matthews_corrcoef(y_test, preds))

    best_threshold = thresholds[np.argmax(mccs)]
    print(f"Optimal threshold: {best_threshold:.4f}")

    # Print final evaluation metrics
    final_preds = (y_pred_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Final Model Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Final Model Specificity: {specificity:.4f}")

    # 8. Save Serialized Artifacts to Disk
    print("Step 6: Saving pipeline artifacts to disk...")
    joblib.dump(final_model, "cvd_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(top_5_features, "top5_features.pkl")
    joblib.dump(best_threshold, "threshold.pkl")
    print("Training process completed successfully! All assets saved.")

if __name__ == "__main__":
    train_model()
