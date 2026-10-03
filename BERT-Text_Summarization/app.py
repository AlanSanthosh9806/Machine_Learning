import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Page Configuration
st.set_page_config(
    page_title="Text Summarization App",
    page_icon="📝",
    layout="wide"
)

# 2. Load Model and Tokenizer with Caching
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading model and tokenizer... Please wait (this happens only once)."):
    tokenizer, model = load_model_and_tokenizer()

# 3. User Interface Layout
st.title("📝 AI Text Summarizer")
st.write("Condense long articles, essays, or notes instantly using transformer models.")

# Sidebar controls for configuration
st.sidebar.header("Configuration")
max_len = st.sidebar.slider("Max Summary Length", min_value=30, max_value=200, value=130)
min_len = st.sidebar.slider("Min Summary Length", min_value=10, max_value=100, value=30)

# Main text input area
article_text = st.text_area("Paste your text or article here:", height=250, placeholder="Type or paste content...")

# 4. Processing and Output
if st.button("Generate Summary", type="primary"):
    if not article_text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Analyzing text and generating summary..."):
            try:
                # Format input prompt explicitly for Flan-T5
                input_text = f"summarize: {article_text}"
                
                # Tokenize and generate output
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    max_length=max_len, 
                    min_length=min_len, 
                    do_sample=False
                )
                
                generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Display Results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Original Stats")
                    st.info(f"Word Count: {len(article_text.split())}")
                
                with col2:
                    st.subheader("📈 Summary Stats")
                    st.success(f"Word Count: {len(generated_summary.split())}")
                
                st.markdown("### 📌 Final Summary")
                st.write(generated_summary)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")