import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

# ----------------------------------------------------
# 1. SETUP DUMMY DATA FOR TRAINING
# ----------------------------------------------------
# Let's create a tiny dataset so the models have something to learn from.
# Features: [Size in sq ft, Number of Bedrooms]
X_train = np.array([
    [600, 1], [800, 1], [1000, 2], [1200, 2], [1500, 3], 
    [1800, 3], [2200, 4], [2500, 4], [3000, 5], [3500, 5]
])

# Targets for Linear Regression (Exact Prices in Dollars)
y_linear = np.array([
    120000, 150000, 190000, 220000, 280000, 
    310000, 400000, 450000, 550000, 620000
])

# Targets for Logistic Regression (Is the house "Expensive"? 1 = Yes, 0 = No)
# Let's assume any house over $300,000 is classified as Expensive
y_logistic = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# ----------------------------------------------------
# 2. TRAIN THE MACHINE LEARNING MODELS
# ----------------------------------------------------
# Train the Continuous Predictor (How much?)
lin_model = LinearRegression()
lin_model.fit(X_train, y_linear)

# Train the Binary Classifier (Is it or is it not?)
log_model = LogisticRegression()
log_model.fit(X_train, y_logistic)

# ----------------------------------------------------
# 3. STREAMLIT WEB INTERFACE ARCHITECTURE
# ----------------------------------------------------
st.set_page_config(page_title="House Predictor Tool", page_icon="🏠")

st.title("🏠 House Prediction App")
st.write("Compare how **Linear** and **Logistic** Regression process the exact same house parameters.")

# Layout: Split inputs into side-by-side columns
st.header("1. Input House Details")
col1, col2 = st.columns(2)

with col1:
    size = st.number_input(
        "Total Square Footage:", 
        min_value=500, 
        max_value=5000, 
        value=1500, 
        step=50
    )

with col2:
    bedrooms = st.slider(
        "Number of Bedrooms:", 
        min_value=1, 
        max_value=5, 
        value=3
    )

# Format user input for scikit-learn format: [[size, bedrooms]]
user_input = np.array([[size, bedrooms]])

# ----------------------------------------------------
# 4. EXECUTE PREDICTIONS & DISPLAY OUTPUTS
# ----------------------------------------------------
st.header("2. Model Predictions")

# Left Column: Linear Regression Output
# Right Column: Logistic Regression Output
col_lin, col_log = st.columns(2)

with col_lin:
    st.subheader("📉 Linear Regression")
    st.caption("Predicts a continuous exact price.")
    
    # Run the linear formula prediction
    predicted_price = lin_model.predict(user_input)[0]
    
    # Custom display card
    st.metric(
        label="Predicted Exact Valuation", 
        value=f"${predicted_price:,.2f}"
    )
    st.info(f"The model calculates a raw mathematical price based on a straight line calculation.")

with col_log:
    st.subheader("📈 Logistic Regression")
    st.caption("Classifies if the house is 'Expensive' (>$300k).")
    
    # Get probability percentages for class 0 and class 1
    probabilities = log_model.predict_proba(user_input)[0]
    expensive_prob = probabilities[1] * 100
    
    # Make final categorical determination
    final_classification = log_model.predict(user_input)[0]
    status = "🔴 EXPENSIVE" if final_classification == 1 else "🟢 AFFORDABLE"
    
    # Custom display card
    st.metric(
        label="Classification Result", 
        value=status
    )
    st.info(f"Calculated Probability: **{expensive_prob:.1f}%** chance of being premium luxury tier.")

# Optional Data Viewer element
if st.checkbox("Show training baseline matrix data"):
    df = pd.DataFrame(X_train, columns=["Square Footage", "Bedrooms"])
    df["Exact Valuation ($)"] = y_linear
    df["Is Expensive Tier? (1=Yes)"] = y_logistic
    st.dataframe(df)
