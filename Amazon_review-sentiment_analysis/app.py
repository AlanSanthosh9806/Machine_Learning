import streamlit as st
import pandas as pd
import re

# ----------------------------------------------------
# 1. HELPER FUNCTIONS FOR STRUCTURING TEXT
# ----------------------------------------------------
def clean_text(text):
    """Removes emojis, punctuation, and extra spaces to normalize text."""
    # Remove punctuation and special characters
    text_only = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    return " ".join(text_only.split()).lower()

def count_emojis(text):
    """Rough count of common emojis or punctuation clusters used as icons."""
    emoji_pattern = re.compile(r'[\u2600-\u27BF]|[\u1F300-\u1F64F]|[\u1F680-\u1F6FF]|👍|⭐')
    return len(emoji_pattern.findall(text))

def analyze_basic_sentiment(text):
    """Rule-based tagger simulating a simple classification model output."""
    pos_words = {'love', 'amazing', 'perfect', 'good', 'fine', 'worth', 'decent', 'comfortable'}
    neg_words = {'not', 'buy', 'broken', 'waste', 'cheap', 'horrible', 'buggy', 'bad'}
    
    words = set(clean_text(text).split())
    pos_score = len(words.intersection(pos_words))
    neg_score = len(words.intersection(neg_words))
    
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

# ----------------------------------------------------
# 2. STREAMLIT CONFIGURATION & INTERFACE
# ----------------------------------------------------
st.set_page_config(page_title="Text Structuring Tool", page_icon="📊", layout="wide")

st.title("📊 Unstructured Text to Structured Data Converter")
st.write("Convert raw text blocks into a clean, feature-engineered matrix ready for Machine Learning pipelines.")

# Sidebar / Input Container
st.header("1. Input Raw Unstructured Dataset")
sample_raw_input = """Absolutely love this wireless mouse! The battery lasts for weeks and it connects instantly to my MacBook. Worth every rupee! 👍
DO NOT BUY!!! The charging cable arrived frayed and completely broken. Tried to contact seller but got no response. Horrible experience.
The product quality is okay, but delivery took almost 10 days to reach Mumbai. Packaging was torn when it arrived.
waste of money.. stopped working after 3 days of use. cheap plastic material.
Works fine I guess. Nothing special but handles basic daily tasks well enough. A bit overpriced compared to others.
Amazing sound quality for the price!! Bass is punchy and noise cancellation is decent enough for daily transit. ⭐⭐⭐⭐⭐
it's alright... not good not bad. it does the job but the app integration is very buggy and disconnects randomly.
Perfect fit, comfortable to wear during workouts, and sweat-resistant. Will definitely buy from this brand again!"""

# Multi-line text input field populated with our sample Amazon reviews
raw_text_data = st.text_area(
    "Paste your raw reviews here (one review per line):", 
    value=sample_raw_input, 
    height=250
)

# ----------------------------------------------------
# 3. TEXT PROCESSING ENGINE
# ----------------------------------------------------
if raw_text_data.strip():
    # Split text line by line to isolate unique data points
    raw_lines = [line.strip() for line in raw_text_data.split("\n") if line.strip()]
    
    structured_records = []
    for idx, raw_review in enumerate(raw_lines, start=1):
        cleaned = clean_text(raw_review)
        record = {
            "Review_ID": f"AMZN_{idx:03d}",
            "Raw_Review_Text": raw_review,
            "Cleaned_Text": cleaned,
            "Character_Length": len(raw_review),
            "Word_Count": len(raw_review.split()),
            "Emoji_Count": count_emojis(raw_review),
            "Extracted_Sentiment": analyze_basic_sentiment(raw_review)
        }
        structured_records.append(record)
        
    # Convert list of structured JSON-like dictionaries into a Pandas Dataframe Matrix
    structured_df = pd.DataFrame(structured_records)
    
    # Display the structured matrix
    st.header("2. Converted Structured Preview Matrix")
    st.dataframe(structured_df, use_container_width=True)
    
    # ----------------------------------------------------
    # 4. DOWNLOAD ENGINE (UTILITY COMPONENT)
    # ----------------------------------------------------
    st.header("3. Download Structured Target Artifact")
    
    # Convert DataFrame back to standard CSV string formatting
    csv_buffer = structured_df.to_csv(index=False).encode('utf-8')
    
    # Native Streamlit download component 
    st.download_button(
        label="📥 Download Structured CSV Dataset",
        data=csv_buffer,
        file_name="structured_amazon_reviews.csv",
        mime="text/csv"
    )
    st.success(f"Successfully processed {len(structured_df)} distinct reviews into structural features.")
else:
    st.warning("Please enter or paste unstructured text in the text area block above to continue.")
