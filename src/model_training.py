import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

import joblib


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

file_path = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\dataset\cleaned_loan_dataset.xlsx"

df = pd.read_excel(file_path)

print("===================================")
print("AI LOAN RISK ASSESSMENT SYSTEM")
print("===================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())


# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

df_model = df.drop(
    columns=[
        "Unnamed: 0",
        "Calculated_Avg_Bill",
        "Calculated_Avg_Payment"
    ],
    errors="ignore"
).copy()

print("\nColumns after removing unnecessary columns:")
print(df_model.columns.tolist())


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df_model.drop(columns=["Y"])
y = df_model["Y"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 4. DISPLAY TARGET DISTRIBUTION
# ==========================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===================================")
print("DATA SPLIT")
print("===================================")

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ==========================================
# 6. DECISION TREE MODEL
# ==========================================

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=8
)

decision_tree.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")


# ==========================================
# 7. DECISION TREE PREDICTION
# ==========================================

y_pred_dt = decision_tree.predict(X_test)

print("\nDecision Tree predictions completed!")

print("\nFirst 20 predictions:")
print(y_pred_dt[:20])


# ==========================================
# 8. DECISION TREE EVALUATION
# ==========================================

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("\n===================================")
print("DECISION TREE RESULTS")
print("===================================")

print("Accuracy :", round(accuracy_dt, 4))
print("Precision:", round(precision_dt, 4))
print("Recall   :", round(recall_dt, 4))
print("F1 Score :", round(f1_dt, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))


# ==========================================
# 9. DECISION TREE LOAN DECISIONS
# ==========================================

print("\n===================================")
print("DECISION TREE LOAN DECISIONS")
print("===================================")

for prediction in y_pred_dt[:20]:

    if prediction == 0:
        print("LOAN APPROVED")
    else:
        print("LOAN REJECTED")


# ==========================================
# 10. RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

random_forest.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")


# ==========================================
# 11. RANDOM FOREST PREDICTION
# ==========================================

y_pred_rf = random_forest.predict(X_test)

print("\nRandom Forest predictions completed!")


# ==========================================
# 12. RANDOM FOREST EVALUATION
# ==========================================

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

print("Accuracy :", round(accuracy_rf, 4))
print("Precision:", round(precision_rf, 4))
print("Recall   :", round(recall_rf, 4))
print("F1 Score :", round(f1_rf, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# ==========================================
# 13. RANDOM FOREST LOAN DECISIONS
# ==========================================

print("\n===================================")
print("RANDOM FOREST LOAN DECISIONS")
print("===================================")

for prediction in y_pred_rf[:20]:

    if prediction == 0:
        print("LOAN APPROVED")
    else:
        print("LOAN REJECTED")


# ==========================================
# 14. MODEL COMPARISON
# ==========================================

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print("\nDecision Tree:")
print("Accuracy :", round(accuracy_dt, 4))
print("Precision:", round(precision_dt, 4))
print("Recall   :", round(recall_dt, 4))
print("F1 Score :", round(f1_dt, 4))

print("\nRandom Forest:")
print("Accuracy :", round(accuracy_rf, 4))
print("Precision:", round(precision_rf, 4))
print("Recall   :", round(recall_rf, 4))
print("F1 Score :", round(f1_rf, 4))


# ==========================================
# 15. BEST MODEL
# ==========================================

if f1_rf > f1_dt:
    print("\nBest Model: RANDOM FOREST")
else:
    print("\nBest Model: DECISION TREE")


# ==========================================
# 16. FINANCIAL RISK SCORE
# ==========================================

risk_probability = random_forest.predict_proba(X_test)[:, 1]

risk_score = (risk_probability * 100).round(2)

financial_score = (100 - risk_score).round(2)


# ==========================================
# 17. RISK CATEGORY
# ==========================================

def get_risk_category(score):

    if score < 30:
        return "LOW RISK"

    elif score < 60:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"


risk_category = [
    get_risk_category(score)
    for score in risk_score
]


# ==========================================
# 18. DISPLAY RISK ASSESSMENT
# ==========================================

print("\n===================================")
print("FINANCIAL RISK ASSESSMENT")
print("===================================")

for i in range(10):

    print("\nApplicant", i + 1)

    print("Risk Score      :", risk_score[i])

    print("Financial Score :", financial_score[i])

    print("Risk Category   :", risk_category[i])

    if y_pred_rf[i] == 0:
        print("Loan Decision   : LOAN APPROVED")
    else:
        print("Loan Decision   : LOAN REJECTED")


# ==========================================
# 19. SAVE RANDOM FOREST MODEL
# ==========================================

model_path = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\models\random_forest_model.pkl"

joblib.dump(random_forest, model_path)

print("\nRandom Forest model saved successfully!")

print("Model location:", model_path)


# ==========================================
# 20. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "LOAN APPROVED",
        "LOAN REJECTED"
    ]
)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# 21. ROC CURVE
# ==========================================

y_probability = random_forest.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {auc_score:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Random Forest ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()


# ==========================================
# 22. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

feature_importance.head(10).plot(
    kind="bar"
)

plt.title(
    "Top 10 Important Features - Random Forest"
)

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()


# ==========================================
# 23. FINAL MODEL ANALYSIS
# ==========================================

print("\n==========================================")
print("MODEL ANALYSIS COMPLETED")
print("==========================================")

print(
    f"ROC-AUC Score: {auc_score:.4f}"
)

print("\nTop 10 Important Features:")

print(
    feature_importance.head(10)
)