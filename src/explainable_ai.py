import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\dataset\cleaned_loan_dataset.xlsx"

df = pd.read_excel(file_path)

# ==========================================
# 2. PREPARE DATA
# ==========================================

X = df.drop(
    columns=[
        "Y",
        "Unnamed: 0",
        "Calculated_Avg_Bill",
        "Calculated_Avg_Payment"
    ],
    errors="ignore"
)

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\models\random_forest_model.pkl"
)
print("Random Forest model loaded successfully!")

# ==========================================
# 4. CREATE SHAP EXPLAINER
# ==========================================

explainer = shap.TreeExplainer(model)

# Use a small sample for explanation
X_sample = X.iloc[:100]

shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated successfully!")

# ==========================================
# 5. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_values[:, :, 1],
    X_sample,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSHAP explanation completed!")
print("SHAP plot saved as shap_summary.png")