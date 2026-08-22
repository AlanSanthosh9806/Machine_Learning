import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Bank Loan Default Predictor")
st.markdown("""
This app handles missing customer data natively using **Median Imputation**, trains a **Random Forest Classifier** (Bagging), 
and visualizes the structural insights like **Out-of-Bag (OOB) Accuracy** and **Feature Importance** out-of-the-box.
""")

# 1. Generate Synthetic Data with Missing Values
@st.cache_data
def generate_bank_data():
    np.random.seed(42)
    n_samples = 1200
    
    # Intrinsic feature distributions
    age = np.random.randint(21, 65, size=n_samples).astype(float)
    income = np.random.normal(60000, 18000, size=n_samples)
    credit_score = np.random.randint(300, 850, size=n_samples).astype(float)
    loan_amount = np.random.normal(18000, 6000, size=n_samples)
    debt_to_income = (loan_amount / (income + 1)) * 100
    
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': income,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Debt_to_Income_Ratio': debt_to_income
    })
    
    # Ground truth mapping logic
    logit = (debt_to_income * 0.15) - (credit_score / 200) + (age / 80)
    prob = 1 / (1 + np.exp(-logit))
    df['Default'] = np.random.binomial(1, prob)
    
    # Introduce random missing values (NaNs) to simulate messy real-world bank data
    for col in ['Age', 'Credit_Score', 'Annual_Income']:
        df.loc[df.sample(frac=0.10).index, col] = np.nan
        
    return df

df_raw = generate_bank_data()

# Sidebar Hyperparameters
st.sidebar.header("⚙️ Model Configuration")
n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider("Max Tree Depth", min_value=3, max_value=20, value=10)

# 2. Programmatic Missing Value Imputation
# Scikit-learn Random Forests require numerical entries without NaNs. 
# We implement a median-fill matrix strategy to resolve data holes safely.
df_imputed = df_raw.copy()
imputation_values = {}

for col in df_imputed.columns.drop('Default'):
    median_val = df_imputed[col].median()
    imputation_values[col] = median_val
    df_imputed[col] = df_imputed[col].fillna(median_val)

# Separate Target and Covariates
X = df_imputed.drop(columns=['Default'])
y = df_imputed['Default']

# 3. Model Construction (Bagging & OOB score activated)
# Setting oob_score=True allows out-of-the-box validation without an explicit train/test split.
model = RandomForestClassifier(
    n_estimators=n_estimators, 
    max_depth=max_depth, 
    oob_score=True, 
    random_state=42
)
model.fit(X, y)

# Layout Architecture
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Raw Customer Registry Data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    st.subheader("🔍 Identified Missing Items")
    missing_counts = df_raw.isnull().sum().rename("NaN Count")
    st.write(missing_counts)

with col2:
    st.subheader("🔮 Real-time Risk Assessment")
    st.write("Input criteria manually below to generate a default classification:")
    
    # Interactive input fields
    in_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    in_income = st.number_input("Annual Income ($)", min_value=1000, value=55000)
    in_credit = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
    in_loan = st.number_input("Requested Loan Amount ($)", min_value=500, value=15000)
    
    # Calculate derived metric
    in_dti = (in_loan / in_income) * 100
    
    # Assembly payload
    input_payload = pd.DataFrame([{
        'Age': in_age,
        'Annual_Income': in_income,
        'Credit_Score': in_credit,
        'Loan_Amount': in_loan,
        'Debt_to_Income_Ratio': in_dti
    }])
    
    if st.button("Evaluate Default Risk", type="primary"):
        prediction = model.predict(input_payload)[0]
        probabilities = model.predict_proba(input_payload)[0]
        default_prob = probabilities[1]
        
        if prediction == 1:
            st.error(f"❌ **High Risk Flagged:** {default_prob:.2%} probability of defaulting.")
        else:
            st.success(f"✅ **Low Risk Verified:** {default_prob:.2%} default probability. Application safe.")

st.divider()

# 4. Out-of-the-box Performance Trackers
st.subheader("📊 Ensemble Metrics & Structural Analytics")
metrics_col, chart_col = st.columns([1, 2])

with metrics_col:
    st.metric(
        label="Out-of-Bag (OOB) Accuracy Score", 
        value=f"{model.oob_score_:.2%}"
    )
    st.markdown("""
    **What is OOB Accuracy?**  
    Because Random Forest relies on bootstrap sampling (bagging), approximately one-third of the data is left out during the training of each individual tree. 
    The model aggregates predictions on these unseen samples to generate a highly accurate internal validation score without needing a separate validation dataset.
    """)

with chart_col:
    # Extract built-in feature importances
    feature_importance_scores = model.feature_importances_
    sorted_indices = np.argsort(feature_importance_scores)[::-1]
    
    importance_df = pd.DataFrame({
        'Feature': X.columns[sorted_indices],
        'Importance Score': feature_importance_scores[sorted_indices]
    })
    
    # Create the Plotly/Seaborn representation
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=importance_df, 
        x='Importance Score', 
        y='Feature', 
        ax=ax, 
        palette="Blues_r",
        hue='Feature',
        legend=False
    )
    ax.set_title("Native Random Forest Feature Importance Matrix", fontsize=12)
    ax.set_xlabel("Relative Predictive Gini Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    
    st.pyplot(fig)
