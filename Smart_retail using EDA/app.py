import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------------------------------------
# 1. Page Configuration & Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Retail & E-Commerce Analytics Hub",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Advanced Retail Analytics & Strategy Hub")
st.markdown("Execute descriptive, diagnostic, predictive, and prescriptive analysis frameworks across core retail segments.")

# -------------------------------------------------------------
# 2. Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.header("Select Retail Domain")
app_mode = st.sidebar.selectbox(
    "Choose a Use Case:",
    ["1. Customer Segmentation", "2. Inventory & Supply Chain", "3. Market Basket Analysis"]
)

# -------------------------------------------------------------
# MODE 1: CUSTOMER SEGMENTATION
# -------------------------------------------------------------
if app_mode == "1. Customer Segmentation":
    st.header("🎯 Customer Segmentation Framework")
    
    # Generate Synthetic Shopper Clusters
    np.random.seed(42)
    n_shoppers = 100
    
    c1 = pd.DataFrame({
        'Age': np.random.normal(24, 3, n_shoppers),
        'Annual_Income_k': np.random.normal(35, 8, n_shoppers),
        'Spending_Score': np.random.normal(75, 10, n_shoppers),
        'Segment': 'Trendsetters'
    })
    c2 = pd.DataFrame({
        'Age': np.random.normal(45, 6, n_shoppers),
        'Annual_Income_k': np.random.normal(95, 12, n_shoppers),
        'Spending_Score': np.random.normal(25, 12, n_shoppers),
        'Segment': 'Frugal High-Earners'
    })
    c3 = pd.DataFrame({
        'Age': np.random.normal(38, 8, n_shoppers),
        'Annual_Income_k': np.random.normal(110, 15, n_shoppers),
        'Spending_Score': np.random.normal(85, 8, n_shoppers),
        'Segment': 'VIP Shoppers'
    })
    
    shoppers_data = pd.concat([c1, c2, c3]).reset_index(drop=True)
    shoppers_data[['Age', 'Annual_Income_k', 'Spending_Score']] = shoppers_data[['Age', 'Annual_Income_k', 'Spending_Score']].astype(int).clip(lower=18)
    
    # --- ANALYTICS LAYER TABS ---
    t1, t2, t3, t4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "📋 Prescriptive"])
    
    with t1:
        st.subheader("Descriptive Analysis: Current Cohort Profile Summary")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_3d = px.scatter_3d(
                shoppers_data, x='Age', y='Annual_Income_k', z='Spending_Score',
                color='Segment', opacity=0.8, height=500, color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        with col2:
            st.write("#### Average Cohort Metrics")
            summary_stats = shoppers_data.groupby('Segment')[['Age', 'Annual_Income_k', 'Spending_Score']].mean().round(1)
            st.dataframe(summary_stats, use_container_width=True)
            
    with t2:
        st.subheader("Diagnostic Analysis: Identifying the Drivers of Behavior")
        st.markdown("**Core Finding:** Variance in spending score is dictated heavily by lifestyle life-stage positioning, not purely net income asset value.")
        fig_box = px.box(shoppers_data, x="Segment", y="Annual_Income_k", color="Segment", title="Income Distributions Across Target Groups")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with t3:
        st.subheader("Predictive Analysis: Value Tier Conversion Modeling")
        st.markdown("Using a localized baseline regression projection, we calculate how age shifts impact long-term lifetime value (LTV).")
        # Live calculation of trendline parameters
        m, b = np.polyfit(shoppers_data['Age'], shoppers_data['Spending_Score'], 1)
        st.metric("Predicted Drop in Spending Score per Year of Age", f"{m:.2f} Points")
        fig_trend = px.scatter(shoppers_data, x="Age", y="Spending_Score", color="Segment", trendline="ols")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with t4:
        st.subheader("Prescriptive Analysis: Core Strategic Action Items")
        st.success("✔️ **Trendsetters:** Deploy high-frequency gamified flash notifications via mobile app channels.")
        st.warning("✔️ **Frugal High-Earners:** Focus ad copy on utility, bulk savings value, and premium quality durability metrics.")
        st.info("✔️ **VIP Shoppers:** Provide immediate white-glove direct customer service lines and early access collection drops.")

# -------------------------------------------------------------
# WORKSPACE 2: INVENTORY & SUPPLY CHAIN MANAGEMENT
# -------------------------------------------------------------
elif app_mode == "2. Inventory & Supply Chain":
    st.header("📈 Inventory & Supply Chain Optimizations")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    winter_gear = [950, 820, 450, 150, 50, 30, 40, 120, 310, 680, 1100, 1450]
    swimwear = [40, 90, 310, 680, 990, 1240, 1310, 950, 410, 120, 50, 30]
    electronics = [450, 380, 410, 430, 460, 510, 490, 530, 610, 580, 1200, 1950]
    
    sales_data = pd.DataFrame({
        'Month': months * 3,
        'Monthly_Units_Sold': winter_gear + swimwear + electronics,
        'Product_Category': ['Winter Outerwear']*12 + ['Premium Swimwear']*12 + ['Consumer Electronics']*12
    })
    
    t1, t2, t3, t4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "📋 Prescriptive"])
    
    with t1:
        st.subheader("Descriptive Analysis: Historic Category Volumes")
        col1, col2 = st.columns([3, 1])
        with col1:
            fig_line = px.line(
                sales_data, x='Month', y='Monthly_Units_Sold', color='Product_Category', markers=True, height=450,
                color_discrete_map={'Winter Outerwear': 'royalblue', 'Premium Swimwear': 'orange', 'Consumer Electronics': 'crimson'}
            )
            st.plotly_chart(fig_line, use_container_width=True)
        with col2:
            fig_pie = px.pie(sales_data, values='Monthly_Units_Sold', names='Product_Category', title="Annual Share")
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with t2:
        st.subheader("Diagnostic Analysis: Seasonal Anomaly Profiling")
        st.markdown("**Root Cause Identification:** Demand peaks are strictly tied to climate conditions and external holiday calendars:")
        st.markdown("- **Winter Outerwear:** Driven by Q4 dropping temperatures.  \n- **Premium Swimwear:** Driven by Q2/Q3 vacation schedules.  \n- **Consumer Electronics:** Spikes dramatically in November (+106% MoM) and December due to Black Friday/Holiday gift giving.")
        
    with t3:
        st.subheader("Predictive Analysis: Next-Season Demand Forecasting")
        st.markdown("Applying a baseline 8% annual market growth modifier to calculate the incoming target inventory volume metrics:")
        
        forecast_df = sales_data.copy()
        forecast_df['Forecasted_Units'] = (forecast_df['Monthly_Units_Sold'] * 1.08).astype(int)
        
        fig_fore = px.bar(forecast_df, x="Month", y=["Monthly_Units_Sold", "Forecasted_Units"], 
                          barmode="group", color_discrete_sequence=["silver", "teal"], title="Next-Season Operational Safety Target Overlap")
        st.plotly_chart(fig_fore, use_container_width=True)
        
    with t4:
        st.subheader("Prescriptive Analysis: Logistics Action Items")
        st.info("📦 **Electronics Buffer Requirements:** Lock in supply contracts for Consumer Electronics by **August** to safeguard inventory ahead of the November holiday surge.")
        st.success("❄️ **Winter Outerwear Markdown Window:** Initiate structured markdowns for winter stock starting **February 15** to reclaim warehouse capacity before swimwear intake begins.")

# -------------------------------------------------------------
# WORKSPACE 3: MARKET BASKET ANALYSIS
# -------------------------------------------------------------
elif app_mode == "3. Market Basket Analysis":
    st.header("🛒 Market Basket Affinity Analysis")
    
    products = ['Charcoal', 'Lighter Fluid', 'Barbecue Meat', 'Craft Beer', 'Paper Plates', 'Sunscreen']
    co_occurrence_matrix = np.array([
        [1.00, 0.88, 0.72, 0.51, 0.42, 0.21],  
        [0.88, 1.00, 0.65, 0.44, 0.38, 0.18],  
        [0.72, 0.65, 1.00, 0.68, 0.55, 0.33],  
        [0.51, 0.44, 0.68, 1.00, 0.39, 0.41],  
        [0.42, 0.38, 0.55, 0.39, 1.00, 0.28],  
        [0.21, 0.18, 0.33, 0.41, 0.28, 1.00]   
    ])
    
    df_matrix = pd.DataFrame(co_occurrence_matrix, columns=products, index=products)
    
    t1, t2, t3, t4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "📋 Prescriptive"])
    
    with t1:
        st.subheader("Descriptive Analysis: Co-occurrence Mapping")
        fig_heat = px.imshow(df_matrix, text_auto=".2f", color_continuous_scale="YlOrRd")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with t2:
        st.subheader("Diagnostic Analysis: Strongest Co-purchase Affinities")
        charcoal_affinity = df_matrix['Charcoal'].drop('Charcoal').reset_index()
        charcoal_affinity.columns = ['Related Product', 'Co-purchase Probability']
        charcoal_affinity = charcoal_affinity.sort_values(by='Co-purchase Probability', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(charcoal_affinity, x='Co-purchase Probability', y='Related Product', orientation='h', color_continuous_scale="Oranges")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.markdown("#### Primary Co-purchase Drivers")
