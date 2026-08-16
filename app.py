import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# Setup Page
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- CUSTOM CSS (PREMIUM UI OVERHAUL) ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Font Override */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main App Background - Dark Glassmorphism */
    .stApp {
        background: radial-gradient(circle at top right, #110e20 0%, #08070d 100%);
        color: #e0e0e0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Premium Header Style text */
    .glow-header {
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #845EC2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-header {
        text-align: center;
        font-weight: 300;
        color: #A0A5AC;
        margin-top: -10px;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    /* Style the text area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 12px;
        padding: 15px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #845EC2 !important;
        box-shadow: 0 0 15px rgba(132, 94, 194, 0.4) !important;
    }

    /* Premium Button Style */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #845EC2, #D65DB1);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 15px rgba(214, 93, 177, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(214, 93, 177, 0.6);
        color: white;
        border: none;
    }

    /* Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        animation: fadeInScale 0.5s ease-out forwards;
    }
    .real-news {
        background: rgba(0, 201, 167, 0.1);
        border: 1px solid #00C9A7;
        box-shadow: 0 0 20px rgba(0, 201, 167, 0.3);
        color: #00C9A7;
    }
    .fake-news {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid #FF6B6B;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
        color: #FF6B6B;
    }

    .result-icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    /* Keyframes for animations */
    @keyframes fadeInScale {
        0% { opacity: 0; transform: scale(0.95) translateY(10px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- APP UI ---
st.markdown('<p class="glow-header">📰 Neural News Verification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced NLP analysis to separate reality from fiction.</p>', unsafe_allow_html=True)

# Main centered container
col1, col2, col3 = st.columns([1, 10, 1])

with col2:
    # Minimalistic loading / model stat message
    placeholder = st.empty()
    placeholder.info("⏳ Initializing ML Engine. This takes a moment...")

    # ML Backend Logic (Untouched conceptually, just moved into column flow)
    @st.cache_resource
    def load_data():
        fake = pd.read_csv("Fake.csv")
        true = pd.read_csv("True.csv")
        fake["label"] = 0
        true["label"] = 1
        df = pd.concat([fake, true])
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    @st.cache_resource
    def train_model(data):
        x_train, x_test, y_train, y_test = train_test_split(
            data["text"], data["label"], test_size=0.2, random_state=42
        )
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        model = PassiveAggressiveClassifier(max_iter=1000)
        model.fit(x_train_vec, y_train)

        acc = accuracy_score(y_test, model.predict(x_test_vec))
        return model, vectorizer, acc

    # Try-except block to handle cases where CSVs are missing locally
    try:
        df = load_data()
        model, vectorizer, accuracy = train_model(df)
        placeholder.success(f"✅ Neural Engine Active (Accuracy: {accuracy*100:.1f}%)")
    except Exception as e:
        placeholder.error(f"⚠️ Could not load datasets: {e}")
        st.stop()
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Input zone
    st.markdown("### 📝 Input Article Text")
    user_input = st.text_area("", placeholder="Paste the news story here to reveal its authenticity...", height=200, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Check Button
    if st.button("RUN ANALYSIS 🚀"):
        if user_input.strip() == "":
            st.warning("⚠️ Please provide some text to analyze.")
        else:
            with st.spinner("Analyzing semantics..."):
                input_vec = vectorizer.transform([user_input])
                pred = model.predict(input_vec)
                
                # Output custom HTML card based on result
                if pred[0] == 1:
                    result_html = '''
                    <div class="result-card real-news">
                        <div class="result-icon">✅</div>
                        <p class="result-title">REAL NEWS</p>
                        <p style="margin-top: 10px;">Our model classified this content as authentic.</p>
                    </div>
                    '''
                else:
                    result_html = '''
                    <div class="result-card fake-news">
                        <div class="result-icon">🚨</div>
                        <p class="result-title">FAKE NEWS</p>
                        <p style="margin-top: 10px;">Our model detected high probabilities of fabricated content.</p>
                    </div>
                    '''
                
                st.markdown(result_html, unsafe_allow_html=True)
