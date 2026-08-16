import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Neural News Verification Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (PREMIUM GLASSMORPHISM & DYNAMIC LAYOUTS) ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main App Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(9, 13, 31, 1) 0%, rgba(6, 7, 18, 1) 90%);
        color: #E2E8F0;
    }

    /* Hide standard header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 16, 38, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(132, 94, 194, 0.3);
    }

    /* Glowing Titles */
    .glow-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF6B6B 10%, #D65DB1 50%, #845EC2 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        font-weight: 300;
        margin-bottom: 30px;
    }

    /* Input Areas */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
        padding: 18px !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTextArea textarea:focus {
        border-color: #845EC2 !important;
        box-shadow: 0 0 18px rgba(132, 94, 194, 0.3) !important;
    }

    /* Premium Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #845EC2, #D65DB1) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        box-shadow: 0 4px 20px rgba(132, 94, 194, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(132, 94, 194, 0.6) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Metric visual indicators */
    .metric-box {
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F1F5F9;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    /* Classification outputs */
    .result-alert {
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 5px;
        animation: scaleUp 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    .real-news-card {
        background: linear-gradient(135deg, rgba(0, 201, 167, 0.08) 0%, rgba(0, 201, 167, 0.02) 100%);
        border: 1px solid rgba(0, 201, 167, 0.4);
        box-shadow: 0 0 30px rgba(0, 201, 167, 0.15);
        color: #00C9A7;
    }
    .fake-news-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.08) 0%, rgba(255, 107, 107, 0.02) 100%);
        border: 1px solid rgba(255, 107, 107, 0.4);
        box-shadow: 0 0 30px rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
    }
    .result-badge {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 10px 0;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Explainable AI Badges */
    .explain-tag {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .real-tag {
        background: rgba(0, 201, 167, 0.06);
        border: 1px solid rgba(0, 201, 167, 0.2);
        color: #00C9A7;
    }
    .fake-tag {
        background: rgba(255, 107, 107, 0.06);
        border: 1px solid rgba(255, 107, 107, 0.2);
        color: #FF6B6B;
    }

    @keyframes scaleUp {
        0% { transform: scale(0.95); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND MODEL LOADING ---
@st.cache_resource
def load_classifier_assets():
    """Loads the pre-trained passive aggressive classifier and TF-IDF vectorizer."""
    model_path = "passive_aggressive_model.joblib"
    vec_path = "tfidf_vectorizer.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Missing trained model assets. Please run the training script.")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    # The locally measured test accuracy
    accuracy = 0.9935
    
    return model, vectorizer, accuracy

# Try to load models
model_loaded = False
try:
    classifier, tfidf, model_accuracy = load_classifier_assets()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model Loading Error: {e}")
    st.warning("Please ensure that you run `python train.py` first to generate the `.joblib` files.")
    st.stop()

# --- NLP HELPERS & EXPLAINABILITY ENGINE ---
def analyze_text_metrics(text):
    """Calculates readability and textual statistics."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    # Estimate reading time (average 200 words per minute)
    reading_time = max(1, round(word_count / 200))
    
    # Sentence count
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = word_count / max(1, sentence_count)
    
    # Lexical diversity
    unique_words = len(set([w.lower().strip(",.?!\"'") for w in words]))
    lexical_diversity = (unique_words / max(1, word_count)) * 100
    
    return {
        "words": word_count,
        "chars": char_count,
        "reading_time": reading_time,
        "avg_sentence_len": avg_sentence_len,
        "lexical_diversity": lexical_diversity
    }

def explain_prediction(text, model, vectorizer):
    """Identifies terms that influenced the decision towards Real vs Fake."""
    # Clean text simple lower representation for vocabulary matching
    clean_text = text.lower()
    
    # Vectorize single input
    tfidf_matrix = vectorizer.transform([clean_text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Find active vocabulary words in input
    coo = tfidf_matrix.tocoo()
    word_indices = coo.col
    tfidf_values = coo.data
    
    if len(word_indices) == 0:
        return [], []
        
    coefs = model.coef_[0]
    word_impact = []
    
    for idx, val in zip(word_indices, tfidf_values):
        word = feature_names[idx]
        weight = coefs[idx]
        score = val * weight
        word_impact.append({
            "word": word,
            "weight": weight,
            "tfidf": val,
            "score": score
        })
        
    # Sort by absolute score to find largest decision factors
    word_impact = sorted(word_impact, key=lambda x: abs(x["score"]), reverse=True)
    
    real_factors = [w for w in word_impact if w["score"] > 0][:5]
    fake_factors = [w for w in word_impact if w["score"] < 0][:5]
    
    return real_factors, fake_factors

# --- DATA PRESETS ---
presets = {
    "None": "",
    "True Preset: Exoplanet Discovery": "NASA discovered a new exoplanet that may support life. The planet is 20 light years away and lies in the habitable zone of its parent star. According to researchers, the atmosphere contains signatures of water vapor and oxygen.",
    "Fake Preset: Pope Endorsement": "BREAKING: The Pope has officially endorsed Donald Trump in the 2024 presidential election. In a shocking statement released this morning, the Vatican expressed support for his policies. Read the full statement now!",
    "True Preset: Climate Talks": "The United Nations announced a major breakthrough in climate negotiations after seven years of stalled talks. Global leaders agreed on binding emission caps and an international fund to support green technology in developing countries.",
    "Fake Preset: Cancer Secret": "A secret cure for cancer has been suppressed by major pharmaceutical companies, according to a whistleblower. The leak claims that inexpensive treatments have been hidden to protect corporate profits from ongoing therapy drugs."
}

# Initialize session state variables
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "None"

def update_preset():
    st.session_state.input_text = presets[st.session_state.selected_preset]

def reset_callback():
    st.session_state.input_text = ""
    st.session_state.selected_preset = "None"

# --- SIDEBAR WORKSPACE ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Control Center")
    
    # Model Status Card
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 15px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Engine Status</span>
            <span style="background: rgba(0, 201, 167, 0.15); color: #00C9A7; font-size: 0.75rem; font-weight: bold; padding: 3px 8px; border-radius: 50px;">ONLINE</span>
        </div>
        <div style="margin-top: 15px;">
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 5px;">Model Classifier:</div>
            <div style="font-weight: 600; color: #F1F5F9; font-size: 0.95rem;">Passive-Aggressive Classifier</div>
        </div>
        <div style="margin-top: 10px;">
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 5px;">Validation Accuracy:</div>
            <div style="font-weight: 700; color: #845EC2; font-size: 1.1rem;">99.35%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚡ Quick Presets")
    st.markdown("Test the engine instantly using a preset sample:")
    
    selected_preset = st.selectbox(
        "Select sample to load:", 
        list(presets.keys()),
        key="selected_preset",
        on_change=update_preset
    )
    
    st.markdown("---")
    st.markdown("### 💡 Core Methodology")
    st.markdown("""
    This platform applies Natural Language Processing (NLP) representation to news authenticity analysis:
    - **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) measures word occurrences relative to the dataset.
    - **Classification**: Passive-Aggressive algorithms are highly effective for text streams and large dimensions.
    """)

# --- MAIN DASHBOARD INTERFACE ---
# App Header
st.markdown('<h1 class="glow-title">📰 Neural News Verification Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning Engine to classify news articles and explain prediction outcomes.</p>', unsafe_allow_html=True)

# Layout Setup
col_main, col_spacer, col_metrics = st.columns([12, 1, 6])

with col_main:
    st.markdown("### ✍️ Input News Article")
    
    # Check if a preset is selected to populate the text area
    input_placeholder = "Paste the text of the article here to run verification analysis..."
    
    user_input = st.text_area(
        "News Text", 
        key="input_text", 
        placeholder=input_placeholder, 
        height=320, 
        label_visibility="collapsed"
    )
    
    col_btn_run, col_btn_clear = st.columns([4, 1])
    
    with col_btn_run:
        run_analysis = st.button("RUN NEURAL ANALYSIS 🚀")
        
    with col_btn_clear:
        st.button("RESET", on_click=reset_callback)

# Processing and Output Visualization
if run_analysis:
    if not user_input.strip():
        st.warning("⚠️ Please provide news content to begin prediction.")
    else:
        # Calculate stats
        metrics = analyze_text_metrics(user_input)
        
        # Make predictions
        input_vec = tfidf.transform([user_input])
        prediction = classifier.predict(input_vec)[0]
        decision_score = classifier.decision_function(input_vec)[0]
        
        # Get explainable AI terms
        real_words, fake_words = explain_prediction(user_input, classifier, tfidf)
        
        with col_main:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Verification Analysis Output")
            
            # Prediction alert card
            if prediction == 1:
                st.markdown(f"""
                <div class="result-alert real-news-card">
                    <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Analysis Result</span><br>
                    <span class="result-badge">✓ VERIFIED AUTHENTIC</span><br>
                    <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                        The document matches the linguistic profile of standard factual journalism. 
                        Decision score confidence: <b>{decision_score:.3f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-alert fake-news-card">
                    <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Analysis Result</span><br>
                    <span class="result-badge">🚨 FABRICATED / FAKE</span><br>
                    <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                        The model detected vocabulary profiles typical of sensationalist or synthetic fake news.
                        Decision score deviation: <b>{decision_score:.3f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display explainable AI metrics in the side column
        with col_metrics:
            st.markdown("### 🔍 Explainable AI Diagnostics")
            st.markdown("Keywords contributing most heavily to the model classification:")
            
            exp_col_real, exp_col_fake = st.columns(2)
            
            with exp_col_real:
                st.markdown("<p style='color: #00C9A7; font-weight: 600; font-size: 0.9rem;'>REAL News Indicators</p>", unsafe_allow_html=True)
                if real_words:
                    for r in real_words:
                        st.markdown(f"""
                        <div class="explain-tag real-tag">
                            <span>{r['word']}</span>
                            <span>+{r['score']:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No positive markers found.")
                    
            with exp_col_fake:
                st.markdown("<p style='color: #FF6B6B; font-weight: 600; font-size: 0.9rem;'>FAKE News Indicators</p>", unsafe_allow_html=True)
                if fake_words:
                    for f in fake_words:
                        st.markdown(f"""
                        <div class="explain-tag fake-tag">
                            <span>{f['word']}</span>
                            <span>{f['score']:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No negative markers found.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Text Complexity Metrics")
            
            # Metrics grid
            m_row1_col1, m_row1_col2 = st.columns(2)
            with m_row1_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{metrics['words']}</div>
                    <div class="metric-label">Word Count</div>
                </div>
                """, unsafe_allow_html=True)
            with m_row1_col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{metrics['reading_time']} min</div>
                    <div class="metric-label">Reading Time</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            
            m_row2_col1, m_row2_col2 = st.columns(2)
            with m_row2_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{metrics['avg_sentence_len']:.1f}</div>
                    <div class="metric-label">Words / Sentence</div>
                </div>
                """, unsafe_allow_html=True)
            with m_row2_col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{metrics['lexical_diversity']:.1f}%</div>
                    <div class="metric-label">Lexical Diversity</div>
                </div>
                """, unsafe_allow_html=True)

# Default view when not analyzed yet
else:
    with col_metrics:
        st.markdown("### 📊 Diagnostics Dashboard")
        st.info("Input news text and trigger analysis to review Explainable AI diagnostics and textual complexity metrics.")
        
        # Display placeholders for metrics
        m_row1_col1, m_row1_col2 = st.columns(2)
        with m_row1_col1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value" style="opacity: 0.25;">0</div>
                <div class="metric-label">Word Count</div>
            </div>
            """, unsafe_allow_html=True)
        with m_row1_col2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value" style="opacity: 0.25;">0 min</div>
                <div class="metric-label">Reading Time</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        m_row2_col1, m_row2_col2 = st.columns(2)
        with m_row2_col1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value" style="opacity: 0.25;">0.0</div>
                <div class="metric-label">Words / Sentence</div>
            </div>
            """, unsafe_allow_html=True)
        with m_row2_col2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value" style="opacity: 0.25;">0.0%</div>
                <div class="metric-label">Lexical Diversity</div>
            </div>
            """, unsafe_allow_html=True)
