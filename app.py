import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import google.generativeai as genai
import json

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
    .mixed-news-card {
        background: linear-gradient(135deg, rgba(255, 179, 64, 0.08) 0%, rgba(255, 179, 64, 0.02) 100%);
        border: 1px solid rgba(255, 179, 64, 0.4);
        box-shadow: 0 0 30px rgba(255, 179, 64, 0.15);
        color: #FFB340;
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

# --- CLEANING PIPELINE ---
def clean_text(text):
    """Leak-free text preprocessing."""
    if not isinstance(text, str):
        return ""
    
    # 1. Remove Reuters publisher prefixes (e.g. "WASHINGTON (Reuters) - ")
    text = re.sub(r'^[A-Z\s\.\,]+ \((Reuters|REUTERS)\) - ', '', text)
    text = re.sub(r'^[A-Z\s\.\,]+ \((Reuters|REUTERS)\) -', '', text)
    # Remove standard city/agency prefixes
    text = re.sub(r'^[A-Z\s\.\,]+ - ', '', text)
    text = re.sub(r'^[A-Z\s\.\,]+ -', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # 5. Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # 6. Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 7. Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
    accuracy = 0.9898 # The leak-free validation accuracy
    
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
    # Preprocess text
    cleaned_input = clean_text(text)
    
    # Vectorize single input
    tfidf_matrix = vectorizer.transform([cleaned_input])
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

# --- GEMINI FACT-CHECKING LOGIC ---
def gemini_fact_check(text, api_key):
    """Uses Gemini 1.5 Flash API to analyze factual claims and logical consistency."""
    if not api_key:
        return {"error": "API Key is missing. Please enter your Gemini API Key in the sidebar."}
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.5-flash')
        
        prompt = f"""
You are an expert fact-checker and media intelligence analyst.
Analyze the following news text and determine its factuality, bias, and credibility.

Text to analyze:
\"\"\"
{text}
\"\"\"

Provide your response strictly in raw JSON format (do not wrap in markdown backticks or code blocks, output only the valid JSON string). The JSON must contain exactly these keys:
- "verdict": "REAL", "FAKE", or "MIXED"
- "confidence": an integer between 0 and 100
- "accuracy_score": an integer between 0 and 100
- "bias_score": an integer between 0 and 100 (where 0 is completely neutral, 100 is highly biased/partisan)
- "summary": a 2-3 sentence overview of why this text is classified this way
- "red_flags": a list of up to 4 warning flags found (e.g., "Sensationalist language", "Lack of checkable citations", "Out of context claims")
- "verifiable_claims": a list of claims extracted from the text, with instructions on how to verify them or whether they are verified
- "credibility_analysis": a brief evaluation of the text's overall logical structure and plausibility.
"""
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean potential markdown wrapping
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(response_text)
        return data
    except Exception as e:
        return {"error": f"API Request Failed: {str(e)}"}

# --- DATA PRESETS ---
presets = {
    "None": "",
    "True Preset: Exoplanet Discovery": "NASA discovered a new exoplanet that may support life. The planet is 20 light years away and lies in the habitable zone of its parent star. According to researchers, the atmosphere contains signatures of water vapor and oxygen.",
    "Fake Preset: Pope Endorsement": "BREAKING: The Pope has officially endorsed Donald Trump in the 2024 presidential election. In a shocking statement released this morning, the Vatican expressed support for his policies. Read the full statement now!",
    "True Preset: Climate Talks": "The United Nations announced a major breakthrough in climate negotiations after seven years of stalled talks. Global leaders agreed on binding emission caps and an international fund to support green technology in developing countries.",
    "Fake Preset: Cancer Secret": "A secret cure for cancer has been suppressed by major pharmaceutical companies, according to a whistleblower. The leak claims that inexpensive treatments have been hidden to protect corporate profits from ongoing therapy drugs."
}

# --- STATE INITIALIZATION ---
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "None"
if "gemini_api_key" not in st.session_state:
    # Auto-load Gemini API key from various potential sources
    api_key = ""
    # 1. Streamlit secrets
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    # 2. Environment variables
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
    # 3. Local .env file
    elif os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str and not line_str.startswith("#") and "=" in line_str:
                        k, v = line_str.split("=", 1)
                        if k.strip() == "GEMINI_API_KEY":
                            api_key = v.strip().strip('"').strip("'")
                            break
        except Exception:
            pass
    st.session_state.gemini_api_key = api_key

def update_preset():
    st.session_state.input_text = presets[st.session_state.selected_preset]

def reset_callback():
    st.session_state.input_text = ""
    st.session_state.selected_preset = "None"

# --- SIDEBAR WORKSPACE ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Control Center")
    
    # Engine Selection
    selected_engine = st.radio(
        "Choose Analysis Engine:",
        ["Local ML Classifier", "Gemini AI Fact-Checker (Recommended)"],
        index=1
    )
    
    # Gemini Key Input
    if selected_engine == "Gemini AI Fact-Checker (Recommended)":
        st.markdown("### 🔑 Gemini AI Credentials")
        api_key = st.text_input(
            "Google AI Studio API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            key="gemini_api_key"
        )
        st.markdown("[Get free Gemini API Key](https://aistudio.google.com/)")
    else:
        api_key = ""

    st.markdown("### ⚡ Quick Presets")
    selected_preset = st.selectbox(
        "Select sample to load:", 
        list(presets.keys()),
        key="selected_preset",
        on_change=update_preset
    )
    
    st.markdown("---")
    st.markdown("### 💡 Core Methodology")
    if selected_engine == "Local ML Classifier":
        st.markdown("""
        **Local ML Classifier**:
        - Preprocesses input to strip publisher metadata, urls, and numbers.
        - Applies TF-IDF representations.
        - Leverages a Passive-Aggressive Classifier to catch stylistic markers of fake vs real news.
        """)
    else:
        st.markdown("""
        **Gemini AI Fact-Checker**:
        - Cross-references claims against known world history and current news timelines.
        - Evaluates logical flow, bias, fallacies, and sensationalist rhetoric.
        - Highlights verifiable claims and source credibility.
        """)

# --- MAIN DASHBOARD INTERFACE ---
# App Header
st.markdown('<h1 class="glow-title">📰 Neural News Verification Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning & Generative AI Fact-checking Engines to classify news authenticity.</p>', unsafe_allow_html=True)

# Layout Setup
col_main, col_spacer, col_metrics = st.columns([12, 1, 6])

with col_main:
    st.markdown("### ✍️ Input News Article")
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
        # Calculate base metrics
        metrics = analyze_text_metrics(user_input)
        
        # 1. Local ML Classifier Path
        if selected_engine == "Local ML Classifier":
            # Clean and predict
            cleaned_input = clean_text(user_input)
            input_vec = tfidf.transform([cleaned_input])
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
                        <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Local ML Result</span><br>
                        <span class="result-badge">✓ VERIFIED AUTHENTIC</span><br>
                        <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                            The document matches the linguistic profile of standard factual journalism. 
                            Linguistic style score: <b>{decision_score:.3f}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-alert fake-news-card">
                        <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Local ML Result</span><br>
                        <span class="result-badge">🚨 FABRICATED / FAKE</span><br>
                        <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                            The model detected vocabulary profiles typical of sensationalist or synthetic fake news.
                            Linguistic style deviation: <b>{decision_score:.3f}</b>
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

        # 2. Gemini Fact-Checker Path
        else:
            with st.spinner("Initiating Gemini AI Fact-Checking Engine..."):
                result = gemini_fact_check(user_input, api_key)
                
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                verdict = result.get("verdict", "MIXED")
                confidence = result.get("confidence", 0)
                accuracy = result.get("accuracy_score", 0)
                bias = result.get("bias_score", 0)
                summary = result.get("summary", "")
                red_flags = result.get("red_flags", [])
                claims = result.get("verifiable_claims", [])
                analysis = result.get("credibility_analysis", "")
                
                with col_main:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📊 Verification Analysis Output")
                    
                    if verdict == "REAL":
                        st.markdown(f"""
                        <div class="result-alert real-news-card">
                            <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Gemini AI Verdict</span><br>
                            <span class="result-badge">✓ FACTUAL / REAL</span><br>
                            <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                                {analysis}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif verdict == "FAKE":
                        st.markdown(f"""
                        <div class="result-alert fake-news-card">
                            <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Gemini AI Verdict</span><br>
                            <span class="result-badge">🚨 FABRICATED / FAKE</span><br>
                            <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                                {analysis}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-alert mixed-news-card">
                            <span style="font-size: 1.1rem; text-transform: uppercase; font-weight: bold; opacity: 0.85;">Gemini AI Verdict</span><br>
                            <span class="result-badge">⚠ MIXED / UNVERIFIED</span><br>
                            <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.9;">
                                {analysis}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📝 Fact-Check Overview & Verdict Summary")
                    st.write(summary)
                    
                    if claims:
                        st.markdown("### 🔍 Verifiable Claims Checklist")
                        for claim in claims:
                            st.markdown(f"- **Claim**: {claim}")

                with col_metrics:
                    st.markdown("### 📈 Fact-Checking Diagnostics")
                    
                    # Factual Accuracy progress bar
                    st.markdown(f"""
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 5px;">
                            <span style="color: #94A3B8;">Factual Accuracy</span>
                            <span style="color: #F1F5F9; font-weight: 600;">{accuracy}%</span>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #00C9A7, #845EC2); width: {accuracy}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Bias Score progress bar
                    st.markdown(f"""
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 5px;">
                            <span style="color: #94A3B8;">Partisan Bias / Sensationalism</span>
                            <span style="color: #F1F5F9; font-weight: 600;">{bias}%</span>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #845EC2, #FF6B6B); width: {bias}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Score progress bar
                    st.markdown(f"""
                    <div style="margin-bottom: 25px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 5px;">
                            <span style="color: #94A3B8;">Model Evaluation Confidence</span>
                            <span style="color: #F1F5F9; font-weight: 600;">{confidence}%</span>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #845EC2, #D65DB1); width: {confidence}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 🚨 Credibility Warnings")
                    if red_flags:
                        for flag in red_flags:
                            st.markdown(f"""
                            <div class="explain-tag fake-tag" style="justify-content: flex-start; gap: 8px;">
                                <span>⚠️</span>
                                <span>{flag}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("No logical fallacies or credibility warning flags detected.")

        # Bottom section: common text metrics
        with col_metrics:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Text Complexity Metrics")
            
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
