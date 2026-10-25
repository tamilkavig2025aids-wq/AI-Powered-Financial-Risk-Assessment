import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Loan Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\TamilKavi\Documents\AI_Loan_Risk_Project"

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_loan_dataset.xlsx"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #172033;
        font-weight: 700;
    }

    .app-header {
        background: linear-gradient(135deg, #173b73, #285ca8);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }

    .app-header h1 {
        color: white;
        margin-bottom: 5px;
        font-size: 32px;
    }

    .app-header p {
        color: #e8efff;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #173b73;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .info-box {
        background-color: #eef4ff;
        border-left: 5px solid #285ca8;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0 20px 0;
    }

    .success-box {
        background-color: #edf9f1;
        border-left: 5px solid #249653;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .danger-box {
        background-color: #fff1f1;
        border-left: 5px solid #d64545;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e7ef;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-title {
        color: #667085;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #173b73;
        font-size: 26px;
        font-weight: 700;
    }

    .receipt-box {
        background-color: white;
        border: 1px solid #dfe5ee;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
    }

    .small-text {
        color: #667085;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_excel(DATA_PATH)

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_type" not in st.session_state:
    st.session_state.user_type = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "customer_input" not in st.session_state:
    st.session_state.customer_input = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_category(risk_score):

    if risk_score < 30:
        return "LOW RISK"

    elif risk_score < 60:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"


def loan_status(prediction):

    if prediction == 0:
        return "LOAN APPROVED"

    return "LOAN REJECTED"


def predict_application(input_row):

    model = load_model()

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_row])

    # IMPORTANT:
    # Use exactly the features used during model training.
    feature_columns = list(model.feature_names_in_)

    input_df = input_df[feature_columns]

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    # Probability of class 1 = loan rejection risk
    risk_probability = float(probabilities[1])

    risk_score = round(risk_probability * 100, 2)

    financial_score = round(100 - risk_score, 2)

    category = risk_category(risk_score)

    status = loan_status(prediction)

    return {
        "prediction": prediction,
        "status": status,
        "risk_score": risk_score,
        "financial_score": financial_score,
        "risk_category": category,
        "approval_probability": round(float(probabilities[0]) * 100, 2),
        "rejection_probability": round(float(probabilities[1]) * 100, 2)
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(input_row, result):

    recommendations = []

    credit_limit = float(input_row.get("X1", 0))

    payment_ratio = float(
        input_row.get("Payment_Ratio", 0)
    )

    utilization = float(
        input_row.get("Credit_Utilization", 0)
    )

    avg_bill = float(
        input_row.get("Avg_Bill_Amount", 0)
    )

    avg_payment = float(
        input_row.get("Avg_Payment_Amount", 0)
    )

    # Credit utilization
    if utilization > 0.70:

        recommendations.append(
            "Reduce your credit utilization. Try to keep your outstanding balance lower compared with your available credit."
        )

    elif utilization > 0.40:

        recommendations.append(
            "Try to maintain a lower credit utilization ratio to strengthen your financial profile."
        )

    # Payment ratio
    if payment_ratio < 0.50:

        recommendations.append(
            "Increase the proportion of your monthly bill that you repay. Consistent higher repayments can improve your financial profile."
        )

    elif payment_ratio < 0.80:

        recommendations.append(
            "Try to repay a larger portion of your monthly outstanding balance on time."
        )

    # Payment history
    repayment_columns = [
        "X6",
        "X7",
        "X8",
        "X9",
        "X10",
        "X11"
    ]

    delayed_months = 0

    for col in repayment_columns:

        value = float(input_row.get(col, 0))

        if value > 0:
            delayed_months += 1

    if delayed_months > 0:

        recommendations.append(
            "Maintain timely payments and avoid delayed installments. Recent repayment behaviour has a strong influence on your risk assessment."
        )

    # Average payment
    if avg_payment < avg_bill:

        recommendations.append(
            "Try to make repayments closer to or above your monthly billed amount whenever financially possible."
        )

    # Credit limit
    if credit_limit < 30000:

        recommendations.append(
            "Build a stronger credit history through regular payments and responsible use of available credit."
        )

    # Default recommendation
    if not recommendations:

        recommendations.append(
            "Continue making payments on time and maintain responsible credit usage."
        )

    return recommendations


# ============================================================
# RECEIPT
# ============================================================

def create_receipt(result):

    timestamp = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    receipt = f"""
AI LOAN RISK ASSESSMENT SYSTEM
========================================

Loan Assessment Receipt

Assessment Date : {timestamp}

Applicant ID    : Customer Assessment

----------------------------------------
LOAN RESULT
----------------------------------------

Status          : {result['status']}
Risk Category   : {result['risk_category']}

Risk Score      : {result['risk_score']}%
Financial Score : {result['financial_score']}%

Approval Probability:
{result['approval_probability']}%

Rejection Probability:
{result['rejection_probability']}%

----------------------------------------

This assessment was generated using
an AI-based Random Forest model.

This result is an automated risk assessment
and should be considered along with other
financial verification procedures.

========================================
"""

    return receipt


# ============================================================
# LOGOUT
# ============================================================

def logout_user():

    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.page = "Dashboard"
    st.session_state.prediction_result = None
    st.session_state.customer_input = None

    st.rerun()


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():

    st.markdown(
        """
        <div class="app-header">
            <h1>🏦 AI Loan Risk Assessment</h1>
            <p>
                Intelligent Financial Risk Assessment and Loan Prediction System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.subheader("Secure Login")

        st.write(
            "Sign in to access the loan assessment system."
        )

        login_type = st.radio(
            "Login as",
            ["Customer", "Admin"],
            horizontal=True
        )

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        if st.button(
            "🔐 Login",
            use_container_width=True,
            type="primary"
        ):

            if (
                login_type == "Admin"
                and username == "admin"
                and password == "admin123"
            ):

                st.session_state.logged_in = True
                st.session_state.user_type = "Admin"
                st.session_state.page = "Dashboard"

                st.success("Admin login successful.")

                st.rerun()

            elif (
                login_type == "Customer"
                and username == "customer"
                and password == "customer123"
            ):

                st.session_state.logged_in = True
                st.session_state.user_type = "Customer"
                st.session_state.page = "Dashboard"

                st.success("Customer login successful.")

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

        st.divider()

        st.caption(
            "Demo Admin: admin / admin123"
        )

        st.caption(
            "Demo Customer: customer / customer123"
        )


# ============================================================
# ADMIN SIDEBAR
# ============================================================

def show_admin_sidebar():

    with st.sidebar:

        st.markdown("## 🏦 Loan AI")

        st.caption("Administrator Portal")

        st.divider()

        st.write("### Navigation")

        if st.button(
            "📊 Dashboard",
            use_container_width=True
        ):
            st.session_state.page = "Dashboard"

        if st.button(
            "👥 Applicants",
            use_container_width=True
        ):
            st.session_state.page = "Applicants"

        if st.button(
            "🤖 AI Model",
            use_container_width=True
        ):
            st.session_state.page = "AI Model"

        if st.button(
            "🔍 Explainable AI",
            use_container_width=True
        ):
            st.session_state.page = "Explainable AI"

        if st.button(
            "📈 Reports",
            use_container_width=True
        ):
            st.session_state.page = "Reports"

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):
            logout_user()


# ============================================================
# CUSTOMER SIDEBAR
# ============================================================

def show_customer_sidebar():

    with st.sidebar:

        st.markdown("## 🏦 Loan AI")

        st.caption("Customer Portal")

        st.divider()

        if st.button(
            "🏠 Dashboard",
            use_container_width=True
        ):
            st.session_state.page = "Dashboard"

        if st.button(
            "📝 Loan Assessment",
            use_container_width=True
        ):
            st.session_state.page = "Loan Assessment"

        if st.button(
            "📋 My Result",
            use_container_width=True
        ):
            st.session_state.page = "My Result"

        if st.button(
            "🧾 Assessment Receipt",
            use_container_width=True
        ):
            st.session_state.page = "Receipt"

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):
            logout_user()


# ============================================================
# ADMIN DASHBOARD
# ============================================================

def admin_dashboard(df):

    st.markdown(
        """
        <div class="app-header">
            <h1>📊 Admin Dashboard</h1>
            <p>Overview of applicants and AI loan risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_applicants = len(df)

    approved = int(
        (df["Y"] == 0).sum()
    )

    rejected = int(
        (df["Y"] == 1).sum()
    )

    approval_rate = (
        approved / total_applicants * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Applicants",
            f"{total_applicants:,}"
        )

    with col2:

        st.metric(
            "Approved",
            f"{approved:,}"
        )

    with col3:

        st.metric(
            "Rejected",
            f"{rejected:,}"
        )

    with col4:

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )

    st.divider()

    st.subheader("📌 Loan Distribution")

    chart_data = pd.DataFrame(
        {
            "Status": [
                "Loan Approved",
                "Loan Rejected"
            ],
            "Applicants": [
                approved,
                rejected
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Status")
    )

    st.info(
        "The dashboard uses the cleaned loan dataset used during model development."
    )


# ============================================================
# ADMIN APPLICANTS
# ============================================================

def admin_applicants(df):

    st.markdown(
        """
        <div class="app-header">
            <h1>👥 Applicant Management</h1>
            <p>Search and review applicant records.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    search = st.text_input(
        "🔎 Search Applicant ID",
        placeholder="Example: APP00001"
    )

    display_df = df.copy()

    if search:

        if "Applicant_ID" in display_df.columns:

            display_df = display_df[
                display_df["Applicant_ID"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

    if "Applicant_ID" in display_df.columns:

        columns_to_show = [
            "Applicant_ID",
            "X1",
            "X5",
            "Y"
        ]

        available_columns = [
            col
            for col in columns_to_show
            if col in display_df.columns
        ]

        table = display_df[
            available_columns
        ].copy()

        if "Y" in table.columns:

            table["Y"] = table["Y"].map(
                {
                    0: "LOAN APPROVED",
                    1: "LOAN REJECTED"
                }
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ADMIN AI MODEL
# ============================================================

def admin_ai_model():

    st.markdown(
        """
        <div class="app-header">
            <h1>🤖 AI Model Performance</h1>
            <p>Performance information for the trained Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Random Forest Classifier")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "81.60%"
        )

    with col2:
        st.metric(
            "Precision",
            "66.09%"
        )

    with col3:
        st.metric(
            "Recall",
            "34.51%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "45.35%"
        )

    st.divider()

    st.subheader("ROC-AUC Score")

    st.metric(
        "ROC-AUC",
        "0.7772"
    )

    st.info(
        "The Random Forest model was selected because it provided better overall accuracy and F1 performance than the Decision Tree model used during comparison."
    )

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    roc_path = os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    )

    feature_path = os.path.join(
        RESULTS_DIR,
        "feature_importance.png"
    )

    if os.path.exists(confusion_path):

        st.subheader("Confusion Matrix")

        st.image(
            confusion_path,
            use_container_width=True
        )

    if os.path.exists(roc_path):

        st.subheader("ROC Curve")

        st.image(
            roc_path,
            use_container_width=True
        )

    if os.path.exists(feature_path):

        st.subheader("Feature Importance")

        st.image(
            feature_path,
            use_container_width=True
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

def explainable_ai():

    st.markdown(
        """
        <div class="app-header">
            <h1>🔍 Explainable AI</h1>
            <p>Understand which financial factors influence the model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        Explainable AI helps administrators understand how the
        Random Forest model makes risk assessments.
        """
    )

    shap_path = os.path.join(
        RESULTS_DIR,
        "shap_summary.png"
    )

    if os.path.exists(shap_path):

        st.image(
            shap_path,
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP summary image was not found."
        )

    st.divider()

    st.subheader("Important Model Factors")

    factors = pd.DataFrame(
        {
            "Feature": [
                "Recent repayment status",
                "Previous repayment status",
                "Credit utilization",
                "Average payment amount",
                "Payment ratio",
                "Credit limit"
            ],
            "Description": [
                "Recent repayment behaviour",
                "Historical repayment behaviour",
                "Amount of available credit being used",
                "Average amount paid",
                "Relationship between payments and bills",
                "Available credit limit"
            ]
        }
    )

    st.dataframe(
        factors,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ADMIN REPORTS
# ============================================================

def admin_reports(df):

    st.markdown(
        """
        <div class="app-header">
            <h1>📈 Reports</h1>
            <p>Summary of the loan dataset and model results.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(df)

    approved = int(
        (df["Y"] == 0).sum()
    )

    rejected = int(
        (df["Y"] == 1).sum()
    )

    report = pd.DataFrame(
        {
            "Metric": [
                "Total Applicants",
                "Loan Approved",
                "Loan Rejected",
                "Approval Percentage",
                "Rejection Percentage"
            ],
            "Value": [
                total,
                approved,
                rejected,
                f"{approved / total * 100:.2f}%",
                f"{rejected / total * 100:.2f}%"
            ]
        }
    )

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Applicant Distribution")

    chart = pd.DataFrame(
        {
            "Loan Status": [
                "Approved",
                "Rejected"
            ],
            "Count": [
                approved,
                rejected
            ]
        }
    )

    st.bar_chart(
        chart.set_index("Loan Status")
    )


# ============================================================
# CUSTOMER DASHBOARD
# ============================================================

def customer_dashboard():

    st.markdown(
        """
        <div class="app-header">
            <h1>🏦 Welcome to AI Loan Assessment</h1>
            <p>Smart financial risk assessment powered by machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Welcome!")

    st.write(
        """
        Our AI-based system analyses your financial and repayment
        information to estimate your loan eligibility.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            **1. Enter Details**

            Provide your financial and repayment information.
            """
        )

    with col2:

        st.info(
            """
            **2. AI Assessment**

            Our Random Forest model evaluates your financial risk.
            """
        )

    with col3:

        st.info(
            """
            **3. Get Result**

            Receive your loan eligibility result and recommendations.
            """
        )

    st.divider()

    st.subheader("Ready to check your eligibility?")

    if st.button(
        "📝 Start Loan Assessment",
        type="primary",
        use_container_width=True
    ):

        st.session_state.page = "Loan Assessment"

        st.rerun()


# ============================================================
# CUSTOMER LOAN ASSESSMENT
# ============================================================

def customer_loan_assessment():

    st.markdown(
        """
        <div class="app-header">
            <h1>📝 Loan Eligibility Assessment</h1>
            <p>
                Tell us about your financial situation.
                Our AI model will assess your loan risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
    "🔒 Privacy Notice\n\n"
    "Your information is used only for the loan risk assessment "
    "performed within this application."
)

    with st.form("loan_application_form"):

        # ----------------------------------------------------
        # PERSONAL INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">👤 Personal Information</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1
            )

        with col2:

            gender_label = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female"
                ]
            )

        col3, col4 = st.columns(2)

        with col3:

            education_label = st.selectbox(
                "Education",
                [
                    "Graduate School",
                    "University",
                    "High School",
                    "Other"
                ]
            )

        with col4:

            marital_label = st.selectbox(
                "Marital Status",
                [
                    "Married",
                    "Single",
                    "Other"
                ]
            )

        # ----------------------------------------------------
        # CREDIT INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">💳 Credit Information</div>',
            unsafe_allow_html=True
        )

        credit_limit = st.number_input(
            "Available Credit Limit (₹)",
            min_value=1000.0,
            max_value=10000000.0,
            value=50000.0,
            step=1000.0
        )

        st.caption(
            "Enter the credit limit available to you."
        )

        # ----------------------------------------------------
        # REPAYMENT HISTORY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📅 Repayment History</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Select the repayment status for each of the last six months."
        )

        repayment_options = {
            "Paid on time": 0,
            "1 month late": 1,
            "2 months late": 2,
            "3 months late": 3,
            "4 months late": 4,
            "5 months late": 5,
            "6+ months late": 6
        }

        repayment_values = []

        month_names = [
            "Latest Month",
            "2 Months Ago",
            "3 Months Ago",
            "4 Months Ago",
            "5 Months Ago",
            "6 Months Ago"
        ]

        for i in range(6):

            col_a, col_b = st.columns([1, 2])

            with col_a:

                st.write(
                    f"**{month_names[i]}**"
                )

            with col_b:

                selected_status = st.selectbox(
                    f"Repayment status {i + 1}",
                    list(repayment_options.keys()),
                    key=f"repayment_{i}",
                    label_visibility="collapsed"
                )

                repayment_values.append(
                    repayment_options[selected_status]
                )

        # ----------------------------------------------------
        # SIX MONTH BILL AND PAYMENT INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">💰 Monthly Financial Information</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Enter your billed amount and payment amount for the last six months."
        )

        bill_values = []
        payment_values = []

        for i in range(6):

            st.markdown(
                f"**{month_names[i]}**"
            )

            col_a, col_b = st.columns(2)

            with col_a:

                bill = st.number_input(
                    "Monthly bill amount (₹)",
                    min_value=0.0,
                    max_value=10000000.0,
                    value=10000.0,
                    step=500.0,
                    key=f"bill_{i}"
                )

            with col_b:

                payment = st.number_input(
                    "Monthly payment amount (₹)",
                    min_value=0.0,
                    max_value=10000000.0,
                    value=5000.0,
                    step=500.0,
                    key=f"payment_{i}"
                )

            bill_values.append(bill)
            payment_values.append(payment)

        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Check My Loan Eligibility",
            use_container_width=True,
            type="primary"
        )

    # --------------------------------------------------------
    # PROCESS APPLICATION
    # --------------------------------------------------------

    if submitted:

        # Mapping customer-friendly values
        # to the original dataset encoding.

        gender = 1 if gender_label == "Male" else 2

        education_mapping = {
            "Graduate School": 1,
            "University": 2,
            "High School": 3,
            "Other": 4
        }

        marital_mapping = {
            "Married": 1,
            "Single": 2,
            "Other": 3
        }

        education = education_mapping[
            education_label
        ]

        marital = marital_mapping[
            marital_label
        ]

        # ----------------------------------------------------
        # MODEL FEATURES
        # ----------------------------------------------------

        input_row = {}

        input_row["X1"] = credit_limit
        input_row["X2"] = gender
        input_row["X3"] = education
        input_row["X4"] = marital
        input_row["X5"] = age

        # Repayment history
        input_row["X6"] = repayment_values[0]
        input_row["X7"] = repayment_values[1]
        input_row["X8"] = repayment_values[2]
        input_row["X9"] = repayment_values[3]
        input_row["X10"] = repayment_values[4]
        input_row["X11"] = repayment_values[5]

        # Bills
        input_row["X12"] = bill_values[0]
        input_row["X13"] = bill_values[1]
        input_row["X14"] = bill_values[2]
        input_row["X15"] = bill_values[3]
        input_row["X16"] = bill_values[4]
        input_row["X17"] = bill_values[5]

        # Payments
        input_row["X18"] = payment_values[0]
        input_row["X19"] = payment_values[1]
        input_row["X20"] = payment_values[2]
        input_row["X21"] = payment_values[3]
        input_row["X22"] = payment_values[4]
        input_row["X23"] = payment_values[5]

        # ----------------------------------------------------
        # ENGINEERED FEATURES
        # ----------------------------------------------------

        avg_bill = np.mean(
            bill_values
        )

        avg_payment = np.mean(
            payment_values
        )

        # Total bill / credit utilization
        if credit_limit > 0:

            total_bill = max(
                np.mean(bill_values),
                0
            )

            credit_utilization = (
                total_bill / credit_limit
            )

        else:

            credit_utilization = 0

        # Payment ratio
        if avg_bill > 0:

            payment_ratio = (
                avg_payment / avg_bill
            )

        else:

            payment_ratio = 0

        # Prevent extreme ratios
        credit_utilization = min(
            max(credit_utilization, 0),
            10
        )

        payment_ratio = min(
            max(payment_ratio, 0),
            10
        )

        input_row[
            "Credit_Utilization"
        ] = credit_utilization

        input_row[
            "Avg_Bill_Amount"
        ] = avg_bill

        input_row[
            "Avg_Payment_Amount"
        ] = avg_payment

        input_row[
            "Payment_Ratio"
        ] = payment_ratio

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            result = predict_application(
                input_row
            )

            st.session_state.prediction_result = result

            st.session_state.customer_input = input_row

            st.session_state.page = "My Result"

            st.rerun()

        except Exception as e:

            st.error(
                f"Unable to process the application: {e}"
            )


# ============================================================
# CUSTOMER RESULT
# ============================================================

def customer_result():

    st.markdown(
        """
        <div class="app-header">
            <h1>📋 Loan Assessment Result</h1>
            <p>Your AI-powered financial risk assessment is complete.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    result = st.session_state.prediction_result

    input_row = st.session_state.customer_input

    if result is None:

        st.info(
            "No assessment has been completed yet."
        )

        if st.button(
            "Start Assessment",
            type="primary"
        ):

            st.session_state.page = "Loan Assessment"

            st.rerun()

        return

    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    if result["prediction"] == 0:

        st.markdown(
            f"""
            <div class="success-box">
                <h2>✅ LOAN APPROVED</h2>
                <p>
                    Congratulations! Based on the AI risk assessment,
                    your financial profile has been classified as
                    <b>{result['risk_category']}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="danger-box">
                <h2>⚠️ LOAN REJECTED</h2>
                <p>
                    The AI assessment indicates a higher financial risk.
                    Review the improvement suggestions below and continue
                    strengthening your financial profile.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Risk Score",
            f"{result['risk_score']}%"
        )

    with col2:

        st.metric(
            "Financial Score",
            f"{result['financial_score']}%"
        )

    with col3:

        st.metric(
            "Risk Category",
            result["risk_category"]
        )

    st.divider()

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    st.subheader("📊 Assessment Details")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Loan Approval Probability**"
        )

        st.progress(
            int(
                result["approval_probability"]
            )
        )

        st.write(
            f"{result['approval_probability']}%"
        )

    with col2:

        st.write(
            "**Loan Rejection Risk**"
        )

        st.progress(
            int(
                result["rejection_probability"]
            )
        )

        st.write(
            f"{result['rejection_probability']}%"
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    if result["prediction"] == 1:

        st.divider()

        st.subheader(
            "💡 How You Can Improve Your Financial Profile"
        )

        recommendations = generate_recommendations(
            input_row,
            result
        )

        for recommendation in recommendations:

            st.warning(
                f"💡 {recommendation}"
            )

        st.info(
            "Improving repayment behaviour, reducing credit usage and maintaining consistent payments may strengthen your financial profile over time."
        )

    else:

        st.divider()

        st.subheader(
            "🎉 Keep Your Financial Profile Strong"
        )

        st.success(
            "Continue making payments on time and maintain responsible credit usage."
        )

        st.success(
            "Avoid unnecessary increases in outstanding balances."
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🧾 View Assessment Receipt",
            use_container_width=True
        ):

            st.session_state.page = "Receipt"

            st.rerun()

    with col2:

        if st.button(
            "🔄 New Assessment",
            use_container_width=True
        ):

            st.session_state.prediction_result = None
            st.session_state.customer_input = None

            st.session_state.page = "Loan Assessment"

            st.rerun()


# ============================================================
# CUSTOMER RECEIPT
# ============================================================

def customer_receipt():

    st.markdown(
        """
        <div class="app-header">
            <h1>🧾 Assessment Receipt</h1>
            <p>Summary of your AI loan risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    result = st.session_state.prediction_result

    if result is None:

        st.info(
            "Complete a loan assessment first."
        )

        return

    receipt = create_receipt(
        result
    )

    st.text_area(
        "Assessment Summary",
        receipt,
        height=420
    )

    st.download_button(
        "⬇️ Download Assessment Receipt",
        data=receipt,
        file_name="loan_assessment_receipt.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    try:

        df = load_dataset()

        load_model()

    except Exception as e:

        st.error(
            f"Unable to load project files: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if not st.session_state.logged_in:

        show_login_page()

        return

    # --------------------------------------------------------
    # ADMIN
    # --------------------------------------------------------

    if st.session_state.user_type == "Admin":

        show_admin_sidebar()

        if st.session_state.page == "Dashboard":

            admin_dashboard(df)

        elif st.session_state.page == "Applicants":

            admin_applicants(df)

        elif st.session_state.page == "AI Model":

            admin_ai_model()

        elif st.session_state.page == "Explainable AI":

            explainable_ai()

        elif st.session_state.page == "Reports":

            admin_reports(df)

        else:

            admin_dashboard(df)

    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    elif st.session_state.user_type == "Customer":

        show_customer_sidebar()

        if st.session_state.page == "Dashboard":

            customer_dashboard()

        elif st.session_state.page == "Loan Assessment":

            customer_loan_assessment()

        elif st.session_state.page == "My Result":

            customer_result()

        elif st.session_state.page == "Receipt":

            customer_receipt()

        else:

            customer_dashboard()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()