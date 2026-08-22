import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# Page configuration
st.set_page_config(page_title="E-Commerce Churn & CLV Predictor", layout="wide")

st.title("🛍️ E-Commerce Customer Churn & CLV Predictor")
st.markdown("""
This production-grade dashboard utilizes a **multi-stage CatBoost ensemble pipeline**. 
It natively ingests categorical strings to predict **churn risk** and instantly projects downstream **Customer Lifetime Value (CLV)**.
""")

# 1. Pipeline Engine: Data Generation (Simulating a real E-commerce platform)
@st.cache_data
def generate_customer_ecosystem():
    np.random.seed(101)
    n_records = 1500
    
    # Categorical arrays (Natively ingested by CatBoost)
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
    devices = ['iOS App', 'Android App', 'Mobile Web', 'Desktop']
    payments = ['Credit Card', 'PayPal', 'Apple Pay', 'Bank Transfer']
    
    df = pd.DataFrame({
        'City': np.random.choice(cities, n_records),
        'Device_Type': np.random.choice(devices, n_records),
        'Payment_Method': np.random.choice(payments, n_records),
        'Tenure_Months': np.random.randint(1, 48, size=n_records).astype(float),
        'Days_Since_Last_Purchase': np.random.randint(1, 90, size=n_records).astype(float),
        'Coupon_Usage_Frequency': np.random.randint(0, 15, size=n_records).astype(float),
        'Support_Tickets_Opened': np.random.randint(0, 8, size=n_records).astype(float),
        'Monthly_Spend': np.random.normal(75, 25, size=n_records).clip(15, 250)
    })
    
    # Mathematical logic simulating Churn (Target 1)
    churn_logit = (df['Days_Since_Last_Purchase'] * 0.04) + (df['Support_Tickets_Opened'] * 0.5) - (df['Tenure_Months'] * 0.03) - 2.0
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    df['Churn_Next_Month'] = np.random.binomial(1, churn_prob)
    
    # Mathematical logic simulating Lifetime Value (Target 2 - Regressor)
    # Loyal customers generate highly compounding residual monetary value
    base_clv = df['Monthly_Spend'] * (df['Tenure_Months'] + 12) * 0.8
    noise = np.random.normal(0, 100, size=n_records)
    df['Projected_CLV'] = np.where(df['Churn_Next_Month'] == 0, base_clv + noise, df['Monthly_Spend'] * 1.2 + noise/2)
    df['Projected_CLV'] = df['Projected_CLV'].clip(lower=0)
    
    return df

df_master = generate_customer_ecosystem()

# Explicitly identify categorical feature columns for CatBoost
cat_features = ['City', 'Device_Type', 'Payment_Method']
feature_cols = ['City', 'Device_Type', 'Payment_Method', 'Tenure_Months', 
                'Days_Since_Last_Purchase', 'Coupon_Usage_Frequency', 'Support_Tickets_Opened', 'Monthly_Spend']

# 2. Multi-Stage Model Fitting
@st.cache_resource
def train_multi_stage_ensemble(data):
    X = data[feature_cols]
    y_cls = data['Churn_Next_Month']
    y_reg = data['Projected_CLV']
    
    # Stage 1: Binary Boosting Classifier
    classifier = CatBoostClassifier(iterations=120, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    classifier.fit(X, y_cls, cat_features=cat_features)
    
    # Stage 2: Boosting Regressor (Optimised explicitly on active users to forecast potential scale)
    regressor = CatBoostRegressor(iterations=120, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
    regressor.fit(X, y_reg, cat_features=cat_features)
    
    return classifier, regressor

clf_model, reg_model = train_multi_stage_ensemble(df_master)

# Application Layout splitting
tab1, tab2 = st.tabs(["🔮 Single Customer Analysis", "📊 Enterprise Model Diagnostic Insights"])

with tab1:
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("📋 Enter Customer Metrics")
        
        # Categorical Selection Components
        in_city = st.selectbox("Customer Demographics (City)", options=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'])
        in_device = st.selectbox("Primary Access Platform", options=['iOS App', 'Android App', 'Mobile Web', 'Desktop'])
        in_payment = st.selectbox("Default Settlement Method", options=['Credit Card', 'PayPal', 'Apple Pay', 'Bank Transfer'])
        
        # Continuous Temporal & Financial Metrics
        in_tenure = st.slider("Account Lifetime (Tenure Months)", 1, 60, 14)
        in_recency = st.slider("Days Elapsed Since Last Purchase", 1, 90, 18)
        in_coupons = st.number_input("Coupons Redeemed this Quarter", min_value=0, max_value=30, value=3)
        in_tickets = st.number_input("Customer Support Tickets Filed", min_value=0, max_value=20, value=1)
        in_spend = st.number_input("Average Monthly Spend Volume ($)", min_value=5, max_value=500, value=85)
        
        # Compile UI inputs directly into structural payload matching data framework
        input_payload = pd.DataFrame([{
            'City': in_city, 'Device_Type': in_device, 'Payment_Method': in_payment,
            'Tenure_Months': float(in_tenure), 'Days_Since_Last_Purchase': float(in_recency),
            'Coupon_Usage_Frequency': float(in_coupons), 'Support_Tickets_Opened': float(in_tickets),
            'Monthly_Spend': float(in_spend)
        }])

    with col_result:
        st.subheader("⚡ Predictive Engine Verdict")
        st.write("Click below to run this entry through the dual-stage evaluation matrix:")
        
        if st.button("Execute Pipeline Diagnostic", type="primary"):
            # Execute Phase 1: Classification inference
            churn_flag = clf_model.predict(input_payload)[0]
            churn_probabilities = clf_model.predict_proba(input_payload)[0]
            churn_risk_pct = churn_probabilities[1]
            
            # Execute Phase 2: Downstream Continuous Monetary Value Forecast
            predicted_clv = reg_model.predict(input_payload)[0]
            
            # Visual presentation of multi-stage system decisions
            st.markdown("---")
            if churn_flag == 1:
                st.error(f"🚨 **High Risk Alert:** Customer exhibits an estimated **{churn_risk_pct:.1%}** probability of churning next month.")
                st.metric(label="Projected Customer Lifetime Value (CLV Residual)", value=f"${predicted_clv:,.2f}")
                st.warning("⚠️ Action Required: Deploy automated localized retention or exclusive coupon packages immediately.")
            else:
                st.success(f"✅ **Account Stable:** Customer shows minimal attrition indicators (**{churn_risk_pct:.1%}** Churn Risk).")
                st.metric(label="Forecasted Long-Term Lifetime Value (CLV)", value=f"${predicted_clv:,.2f}")
                st.info("💡 High Value Target: Retain organic cross-selling pathways and premium loyalty rewards tiers.")

with tab2:
    st.subheader("🔬 Enterprise Analytics & Feature Importance Breakdown")
    st.write("Review the underlying predictive structures generated instantly by the dual boosting layers:")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 🔍 Attrition Classifier Drivers (Stage 1)")
        clf_importance = clf_model.get_feature_importance()
        clf_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': clf_importance}).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=clf_imp_df, x='Importance', y='Feature', ax=ax, palette="Oranges_r", hue='Feature', legend=False)
        ax.set_title("What drives a user to drop their subscription?")
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("#### 💰 Lifetime Value Regressor Drivers (Stage 2)")
        reg_importance = reg_model.get_feature_importance()
        reg_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': reg_importance}).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=reg_imp_df, x='Importance', y='Feature', ax=ax, palette="Purples_r", hue='Feature', legend=False)
        ax.set_title("What features lock in high revenue generations?")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 📋 Synthetic Registry Ledger Sample (First 5 Rows)")
    st.dataframe(df_master.head(5), use_container_width=True)
