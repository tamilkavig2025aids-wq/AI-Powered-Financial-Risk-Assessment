import pandas as pd

# Load cleaned dataset
file_path = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project\dataset\cleaned_loan_dataset.xlsx"

df = pd.read_excel(file_path)

# Check whether Applicant_ID already exists
if "Applicant_ID" in df.columns:
    print("Applicant_ID already exists.")
else:
    # Create unique Applicant IDs
    df.insert(
        0,
        "Applicant_ID",
        [f"APP{i:05d}" for i in range(1, len(df) + 1)]
    )

    # Save updated dataset
    df.to_excel(file_path, index=False)

    print("Applicant IDs added successfully!")

print("Total applicants:", len(df))

print("\nFirst 5 Applicant IDs:")
print(df[["Applicant_ID"]].head())

print("\nLast 5 Applicant IDs:")
print(df[["Applicant_ID"]].tail())