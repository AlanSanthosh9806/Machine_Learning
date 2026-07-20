import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------------------------------------------
# 1. PAGE SETUP & DATA CONFIGURATION
# ----------------------------------------------------
st.set_page_config(page_title="Text Clustering Workspace", page_icon="🔮", layout="wide")

st.title("🔮 Unstructured Text Clustering & Visualization Dashboard")
st.write("Automatically categorize and map raw unstructured sentences into visual keyword-driven clusters.")

# Baseline template containing 20 reviews of mixed contexts (Good, Bad, Delivery, Hardware Issues)
default_text_corpus = """Absolutely love this wireless mouse! The battery lasts for weeks and it connects instantly to my MacBook. Worth every rupee! 👍
DO NOT BUY!!! The charging cable arrived frayed and completely broken. Tried to contact seller but got no response. Horrible experience.
The product quality is okay, but delivery took almost 10 days to reach Mumbai. Packaging was torn when it arrived.
waste of money.. stopped working after 3 days of use. cheap plastic material.
Works fine I guess. Nothing special but handles basic daily tasks well enough. A bit overpriced compared to others.
Amazing sound quality for the price!! Bass is punchy and noise cancellation is decent enough for daily transit. ⭐⭐⭐⭐⭐
it's alright... not good not bad. it does the job but the app integration is very buggy and disconnects randomly.
Perfect fit, comfortable to wear during workouts, and sweat-resistant. Will definitely buy from this brand again!
OMG worst kitchen blender ever 😡 index finger got cut because of loose blade design!! unsafe for families do not buy.
delivery boy was very polite and helpful. product packing was excellent. 10/10 service.
Decent phone case for my iPhone 14. Fits snugly but the power button is a bit stiff to press. color matches description.
terrible sound quality!! sounds like music is coming from inside a tin can. returning it tomorrow morning.
Wow! Exceeded my expectations. The fabric is super soft, breathable, and doesn't shrink after washing. Highly recommend. 👌✨
Is this even original? The logo looks printed askew and the stitching is falling apart right out of the plastic cover. Disappointed.
Keyboard keys are clicky and mechanical feel is superb for gaming! RGB lighting profiles are fully customizable.
Received a completely different item altogether. Ordered black running shoes, got blue house slippers instead?? total mess.
The battery backup is insanely good. Used it for 4 full days on a single charge during my trekking trip in Himachal.
average performance. It works fine for small rooms but takes forever to cool down a larger living room area.
DO NOT TRUST THE REVIEWS!!! chemical smell is so strong it gave me a massive headache within 5 minutes of opening. 🤮
Keyboard is fine but delivery took ages... almost forgot I ordered it. bad logistics."""

# ----------------------------------------------------
# 2. CONTROL DASHBOARD SIDEBAR PANEL
# ----------------------------------------------------
st.sidebar.header("⚙️ Cluster Hyperparameters")
num_clusters = st.sidebar.slider("Select Number of Target Clusters (K):", min_value=2, max_value=5, value=3)

st.header("1. Input Document Text Stream")
raw_input_area = st.text_area("Input reviews or sentences (one unique entry per text line breaks):", value=default_text_corpus, height=220)

# Process only if text lines exist
corpus_lines = [line.strip() for line in raw_input_area.split("\n") if line.strip()]

if len(corpus_lines) >= num_clusters:
    # ----------------------------------------------------
    # 3. TEXT VECTORIZATION & K-MEANS MACHINE LEARNING
    # ----------------------------------------------------
    # Step 1: Convert raw text sentences into mathematical TF-IDF numerical vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    numerical_text_matrix = tfidf_vectorizer.fit_transform(corpus_lines)
    
    # Step 2: Fit K-Means clustering algorithm to partition text based on keyword similarities
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    predicted_cluster_labels = kmeans_model.fit_predict(numerical_text_matrix)
    
    # Step 3: Compress high-dimensional text vectors into a 2D plane (X, Y) using PCA dimensionality reduction
    pca_engine = PCA(n_components=2, random_state=42)
    two_dimensional_coordinates = pca_engine.fit_transform(numerical_text_matrix.toarray())
    
    # Assemble raw metrics into a cohesive Pandas visual dataframe matrix
    viz_dataframe = pd.DataFrame({
        "Document Text": corpus_lines,
        "Assigned Cluster ID": [f"Cluster Group {label}" for label in predicted_cluster_labels],
        "Principal Axis X": two_dimensional_coordinates[:, 0],
        "Principal Axis Y": two_dimensional_coordinates[:, 1],
    })
    
    # ----------------------------------------------------
    # 4. PLOTLY GRAPHIC ENGINE RENDERING
    # ----------------------------------------------------
    st.header("2. Interactive Cluster Geometry Visualization Space")
    st.write("Dots positioned closer together share high keyword semantic similarities inside your raw document strings.")
    
    # Build a Plotly interactive scatter chart structure 
    fig = px.scatter(
        viz_dataframe,
        x="Principal Axis X",
        y="Principal Axis Y",
        color="Assigned Cluster ID",
        hover_data={"Document Text": True, "Principal Axis X": False, "Principal Axis Y": False},
        title="2D Spatial Distribution Profile of Text Data Points",
        template="plotly_dark",
        size_max=12
    )
    
    # Tweak graphic marker sizing for visual clarity 
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='White')))
    st.plotly_chart(fig, use_container_width=True)
    
    # ----------------------------------------------------
    # 5. VIEW SEGMENTED DATABASE MATRIX TRACES
    # ----------------------------------------------------
    st.header("3. Grouped Content Registry Breakdown")
    selected_view_group = st.selectbox("Isolate Specific Cluster Category View Matrix:", sorted(viz_dataframe["Assigned Cluster ID"].unique()))
    
    filtered_group_df = viz_dataframe[viz_dataframe["Assigned Cluster ID"] == selected_view_group][["Document Text"]]
    st.dataframe(filtered_group_df, use_container_width=True)

else:
    st.error(f"You must supply at least {num_clusters} lines of unique text elements to build {num_clusters} visual clusters successfully.")
