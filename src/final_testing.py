import pandas as pd
import joblib


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model_path = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\models\random_forest_model.pkl"

model = joblib.load(model_path)

print("==========================================")
print("FINAL MODEL TESTING")
print("==========================================")

print("\nRandom Forest model loaded successfully!")


# ==========================================
# 2. LOAD DATASET
# ==========================================

file_path = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\dataset\cleaned_loan_dataset.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")


# ==========================================
# 3. PREPARE FEATURES
# ==========================================

df_model = df.drop(
    columns=[
        "Unnamed: 0",
        "Calculated_Avg_Bill",
        "Calculated_Avg_Payment",
        "Y"
    ],
    errors="ignore"
).copy()


# ==========================================
# 4. SELECT TEST APPLICANTS
# ==========================================

X_test_samples = df_model.iloc[0:10].copy()

print("\nTesting 10 applicants...")


# ==========================================
# 5. PREDICTION
# ==========================================

predictions = model.predict(X_test_samples)

probabilities = model.predict_proba(X_test_samples)[:, 1]


# ==========================================
# 6. RISK SCORE
# ==========================================

risk_scores = (probabilities * 100).round(2)

financial_scores = (100 - risk_scores).round(2)


# ==========================================
# 7. RISK CATEGORY
# ==========================================

def get_risk_category(score):

    if score < 30:
        return "LOW RISK"

    elif score < 60:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("FINAL LOAN PREDICTION RESULTS")
print("==========================================")

for i in range(len(predictions)):

    if predictions[i] == 0:
        decision = "LOAN APPROVED"
    else:
        decision = "LOAN REJECTED"

    category = get_risk_category(risk_scores[i])

    print("\nApplicant", i + 1)

    print("Risk Score      :", risk_scores[i])

    print("Financial Score :", financial_scores[i])

    print("Risk Category   :", category)

    print("Loan Decision   :", decision)


# ==========================================
# 9. FINAL TESTING COMPLETED
# ==========================================

print("\n==========================================")
print("FINAL MODEL TESTING COMPLETED")
print("==========================================")

print("Model is ready for application development.")