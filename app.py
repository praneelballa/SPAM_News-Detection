import streamlit as st
import joblib
import re
import matplotlib.pyplot as plt
from fact_checking.fact_checker import FactChecker
from fact_checking.decision_engine import final_decision

# --- 1. SETUP & LOAD MODELS ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    fact_checker = FactChecker("fact_checking/facts.json")
    return model, vectorizer, fact_checker

model, vectorizer, fact_checker = load_assets()

# --- 2. HELPER FUNCTIONS ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def plot_explanation(vec, model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    vector = vec.toarray()[0]
    
    contributions = []
    for i, v in enumerate(vector):
        if v != 0:
            contributions.append((feature_names[i], coefficients[i] * v))
    
    contributions = sorted(contributions, key=lambda x: x[1])
    top = contributions[:5] + contributions[-5:]
    
    words = [w for w, _ in top]
    scores = [s for _, s in top]
    colors = ['red' if s < 0 else 'green' for s in scores]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(words, scores, color=colors)
    ax.set_title("Word Contributions (Red: Fake-leaning | Green: Real-leaning)")
    plt.tight_layout()
    return fig

# --- 3. USER INTERFACE ---
st.set_page_config(page_title="AI Fact Checker", page_icon="🧠")
st.title("🧠 AI Fake News + Fact Verifier")

user_input = st.text_area("Enter a claim or news article:", height=200)

if st.button("Analyze & Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        
        result = final_decision(user_input, model, vectorizer, fact_checker)
        
        st.divider()
        st.header(f"Verdict: {result['verdict']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 ML Analysis")
            ml_pred = result["ml_prediction"]
            
            # Convert numeric to string first
            if isinstance(ml_pred, int):
                ml_pred = "real" if ml_pred == 1 else "fake"
            
            # Use your specific string check
            if ml_pred == "real":
                st.success(f"ML Classification: {ml_pred.upper()}")
            else:
                st.error(f"ML Classification: {ml_pred.upper()}")
            
            fig = plot_explanation(vec, model, vectorizer)
            st.pyplot(fig)

        with col2:
            st.subheader("🔍 Fact-Check Match")
            if result["similarity"] > 0.7:
                st.info(f"**Closest Match:** {result['fact_match']}")
                st.write(f"**Label:** {result['fact_label']}")
                st.write(f"**Source:** {result['source']}")
                st.progress(result["similarity"], text=f"Similarity: {result['similarity']:.2f}")
            else:
                st.warning("No high-confidence match found in the facts database.")
    else:
        st.warning("Please enter some text to analyze.")