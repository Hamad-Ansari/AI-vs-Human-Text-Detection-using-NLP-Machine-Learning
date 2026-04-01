"""
🤖 AI vs Human Text Detector — Streamlit App
=============================================
Models used:
  • best_ml_model.pkl  — Best classical ML model (TF-IDF based)
  • tfidf_vectorizer.pkl — TF-IDF vectorizer
  • lstm_model.h5      — Bidirectional LSTM deep-learning model
  • lstm_tokenizer.pkl — Keras tokenizer for LSTM input

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import string
import time
import os
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0f1117; color: #e0e0e0; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid #00d4ff33;
        box-shadow: 0 4px 32px #00d4ff22;
        text-align: center;
    }
    .hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0;
               background: linear-gradient(90deg, #00d4ff, #a855f7);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p  { color: #94a3b8; margin: .5rem 0 0; font-size: 1.05rem; }

    /* Result cards */
    .result-ai {
        background: linear-gradient(135deg, #1f0a0a, #2d1b1b);
        border: 1.5px solid #ef4444aa;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        box-shadow: 0 0 24px #ef444433;
    }
    .result-human {
        background: linear-gradient(135deg, #0a1f0a, #1b2d1b);
        border: 1.5px solid #22c55eaa;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        box-shadow: 0 0 24px #22c55e33;
    }
    .result-title { font-size: 2rem; font-weight: 800; margin: 0; }
    .result-sub   { color: #94a3b8; font-size: .95rem; margin-top: .3rem; }
    .confidence-bar {
        height: 10px; border-radius: 99px; margin-top: .8rem;
        background: #1e293b;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%; border-radius: 99px;
        transition: width .6s ease;
    }

    /* Metric chip */
    .chip {
        display: inline-block;
        background: #1e293b;
        border-radius: 99px;
        padding: .25rem .9rem;
        font-size: .82rem;
        margin: .2rem;
        color: #94a3b8;
    }

    /* Info box */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #a855f7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: .8rem 0;
        font-size: .92rem;
        color: #cbd5e1;
    }

    /* Upload zone */
    .uploadedFile { background: #1e293b !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
MAX_LENGTH = 200   # must match training

@st.cache_resource(show_spinner="⚙️ Loading models…")
def load_models():
    import joblib
    errors = []
    models = {}

    # ML model + vectorizer
    for key, path in [("ml_model", "best_ml_model.pkl"),
                      ("tfidf",    "tfidf_vectorizer.pkl")]:
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception as e:
                errors.append(f"{path}: {e}")
        else:
            errors.append(f"File not found: {path}")

    # LSTM tokenizer
    if os.path.exists("lstm_tokenizer.pkl"):
        try:
            with open("lstm_tokenizer.pkl", "rb") as f:
                models["tokenizer"] = pickle.load(f)
        except Exception as e:
            errors.append(f"lstm_tokenizer.pkl: {e}")
    else:
        errors.append("File not found: lstm_tokenizer.pkl")

    # LSTM model (TF)
    if os.path.exists("lstm_model.h5"):
        try:
            import tensorflow as tf
            models["lstm"] = tf.keras.models.load_model("lstm_model.h5")
        except Exception as e:
            errors.append(f"lstm_model.h5: {e}")
    else:
        errors.append("File not found: lstm_model.h5")

    return models, errors


models, load_errors = load_models()

# ── Text preprocessing (mirrors training notebook) ────────────────────────────
import nltk
nltk.download("stopwords",   quiet=True)
nltk.download("punkt",       quiet=True)
nltk.download("punkt_tab",   quiet=True)
nltk.download("wordnet",     quiet=True)

from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def advanced_clean(text: str) -> str:
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens if t not in _STOP_WORDS]
    return " ".join(tokens)


# ── Prediction functions ──────────────────────────────────────────────────────
def predict_ml(text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa
    processed = advanced_clean(text)
    vec       = models["tfidf"].transform([processed])
    pred      = models["ml_model"].predict(vec)[0]
    prob      = models["ml_model"].predict_proba(vec)[0, 1]
    return int(pred), float(prob)

def predict_lstm(text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = models["tokenizer"].texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH)
    prob   = float(models["lstm"].predict(padded, verbose=0)[0, 0])
    pred   = int(prob > 0.5)
    return pred, prob

def predict_ensemble(text: str):
    """Average ML + LSTM probabilities."""
    _, p_ml   = predict_ml(text)
    _, p_lstm = predict_lstm(text)
    prob      = (p_ml + p_lstm) / 2
    pred      = int(prob > 0.5)
    return pred, prob, p_ml, p_lstm


# ── Stylometric quick stats ───────────────────────────────────────────────────
def quick_stats(text: str) -> dict:
    words     = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    chars     = len(text)
    unique    = len(set(w.lower() for w in words))
    return {
        "Characters":         chars,
        "Words":              len(words),
        "Sentences":          len(sentences),
        "Avg word length":    round(np.mean([len(w) for w in words]), 2) if words else 0,
        "Vocab richness":     round(unique / len(words), 3) if words else 0,
        "Avg sentence length": round(len(words) / len(sentences), 1) if sentences else 0,
    }


# ── Render result card ────────────────────────────────────────────────────────
def render_result(label: int, prob: float, model_name: str, p_ml=None, p_lstm=None):
    is_ai     = label == 1
    css_class = "result-ai" if is_ai else "result-human"
    emoji     = "🤖" if is_ai else "👤"
    title     = "AI Generated" if is_ai else "Human Written"
    color     = "#ef4444" if is_ai else "#22c55e"
    conf_pct  = int(prob * 100) if is_ai else int((1 - prob) * 100)

    bar_html = (
        f'<div class="confidence-bar">'
        f'<div class="confidence-fill" style="width:{conf_pct}%;'
        f'background:linear-gradient(90deg,{color},{color}88);"></div>'
        f'</div>'
    )

    extra = ""
    if p_ml is not None and p_lstm is not None:
        extra = (
            f'<br><span class="chip">ML confidence: {p_ml*100:.1f}%</span>'
            f'<span class="chip">LSTM confidence: {p_lstm*100:.1f}%</span>'
        )

    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title" style="color:{color}">{emoji} {title}</div>
        <div class="result-sub">
            Confidence: <strong style="color:{color}">{conf_pct}%</strong>
            &nbsp;|&nbsp; Model: <strong>{model_name}</strong>
            {extra}
        </div>
        {bar_html}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_choice = st.selectbox(
        "Prediction model",
        ["🤝 Ensemble (ML + LSTM)", "🔤 ML Model (TF-IDF)", "🧠 LSTM Deep Learning"],
    )

    st.markdown("---")
    st.markdown("### 📋 Model Status")
    for name, key in [("ML Model", "ml_model"), ("TF-IDF", "tfidf"),
                      ("LSTM Model", "lstm"), ("LSTM Tokenizer", "tokenizer")]:
        ok = key in models
        st.markdown(f"{'✅' if ok else '❌'} **{name}**")

    if load_errors:
        with st.expander("⚠️ Load warnings"):
            for e in load_errors:
                st.warning(e)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Detects whether text was written by a **human** or generated by an **AI** using:
    - Stylometric analysis
    - TF-IDF + best classical ML
    - Bidirectional LSTM
    """)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🔍 AI vs Human Text Detector</h1>
    <p>Detect whether text is human-written or AI-generated using NLP & Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_bulk, tab_upload = st.tabs([
    "✏️ Single Text", "📋 Bulk Paste", "📁 Upload CSV"
])


# ─── Tab 1 · Single Text ───────────────────────────────────────────────────
with tab_single:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("### 📝 Enter text to analyze")
        user_text = st.text_area(
            label="",
            placeholder="Paste your text here…",
            height=280,
            key="single_input",
            label_visibility="collapsed",
        )
        st.caption(f"Characters: {len(user_text)}")

        analyze_btn = st.button("🔍 Analyze Text", use_container_width=True,
                                 type="primary")

    with col_result:
        st.markdown("### 🎯 Prediction Result")

        if analyze_btn:
            if not user_text.strip():
                st.warning("Please enter some text first.")
            elif len(user_text.split()) < 10:
                st.warning("Please enter at least 10 words for a reliable prediction.")
            else:
                # Check required models
                required = {"ml_model", "tfidf"} if "ML" in model_choice else set()
                if "LSTM" in model_choice:
                    required |= {"lstm", "tokenizer"}
                if "Ensemble" in model_choice:
                    required = {"ml_model", "tfidf", "lstm", "tokenizer"}

                missing = required - set(models)
                if missing:
                    st.error(f"Missing models: {', '.join(missing)}")
                else:
                    with st.spinner("Analyzing…"):
                        time.sleep(0.3)
                        try:
                            if "Ensemble" in model_choice:
                                label, prob, p_ml, p_lstm = predict_ensemble(user_text)
                                render_result(label, prob, "Ensemble", p_ml, p_lstm)
                            elif "ML" in model_choice:
                                label, prob = predict_ml(user_text)
                                render_result(label, prob, "ML (TF-IDF)")
                            else:
                                label, prob = predict_lstm(user_text)
                                render_result(label, prob, "LSTM")
                        except Exception as ex:
                            st.error(f"Prediction error: {ex}")

                    # Quick stats
                    st.markdown("---")
                    st.markdown("#### 📊 Text Statistics")
                    stats = quick_stats(user_text)
                    cols = st.columns(3)
                    for i, (k, v) in enumerate(stats.items()):
                        cols[i % 3].metric(k, v)
        else:
            st.markdown("""
            <div class="info-box">
                Enter text in the left panel and click
                <strong>Analyze Text</strong> to see the prediction.
            </div>
            """, unsafe_allow_html=True)


# ─── Tab 2 · Bulk Paste ────────────────────────────────────────────────────
with tab_bulk:
    st.markdown("### 📋 Analyze multiple texts at once")
    st.markdown(
        '<div class="info-box">Paste one text per line. Each line is analyzed independently.</div>',
        unsafe_allow_html=True,
    )

    bulk_text = st.text_area("Paste texts (one per line):", height=200, key="bulk_input")
    bulk_btn  = st.button("🔍 Analyze All", use_container_width=True,
                           type="primary", key="bulk_btn")

    if bulk_btn and bulk_text.strip():
        lines = [l.strip() for l in bulk_text.strip().splitlines() if l.strip()]
        required = {"ml_model", "tfidf"} if "ML" in model_choice else set()
        if "LSTM" in model_choice:
            required |= {"lstm", "tokenizer"}
        if "Ensemble" in model_choice:
            required = {"ml_model", "tfidf", "lstm", "tokenizer"}
        missing = required - set(models)

        if missing:
            st.error(f"Missing models: {', '.join(missing)}")
        else:
            results_rows = []
            prog = st.progress(0, text="Analyzing…")
            for i, line in enumerate(lines):
                try:
                    if len(line.split()) < 3:
                        pred_label, confidence = "Too short", 0.0
                    elif "Ensemble" in model_choice:
                        lbl, prob, _, _ = predict_ensemble(line)
                        pred_label = "AI Generated" if lbl == 1 else "Human Written"
                        confidence = prob if lbl == 1 else 1 - prob
                    elif "ML" in model_choice:
                        lbl, prob = predict_ml(line)
                        pred_label = "AI Generated" if lbl == 1 else "Human Written"
                        confidence = prob if lbl == 1 else 1 - prob
                    else:
                        lbl, prob = predict_lstm(line)
                        pred_label = "AI Generated" if lbl == 1 else "Human Written"
                        confidence = prob if lbl == 1 else 1 - prob
                except Exception as e:
                    pred_label, confidence = f"Error: {e}", 0.0

                results_rows.append({
                    "#":          i + 1,
                    "Text":       line[:120] + ("…" if len(line) > 120 else ""),
                    "Prediction": pred_label,
                    "Confidence": f"{confidence*100:.1f}%",
                })
                prog.progress((i + 1) / len(lines), text=f"Analyzed {i+1}/{len(lines)}")

            prog.empty()
            df_bulk = pd.DataFrame(results_rows)
            st.dataframe(df_bulk, use_container_width=True, hide_index=True)

            ai_count = sum(1 for r in results_rows if r["Prediction"] == "AI Generated")
            st.info(f"🤖 AI Generated: **{ai_count}** | 👤 Human Written: **{len(lines) - ai_count}**")

            csv_bytes = df_bulk.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "bulk_results.csv", "text/csv")


# ─── Tab 3 · Upload CSV ────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### 📁 Upload a CSV file for batch prediction")
    st.markdown(
        '<div class="info-box">Your CSV must have a column containing the text to analyze. '
        'Select the column below after upload.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df_up = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_up):,} rows × {len(df_up.columns)} columns")
            st.dataframe(df_up.head(5), use_container_width=True)

            text_col = st.selectbox("Select the text column:", df_up.columns.tolist())
            max_rows = st.slider("Max rows to process:", 10, min(5000, len(df_up)),
                                  min(500, len(df_up)), step=50)

            run_btn = st.button("🚀 Run Prediction", use_container_width=True,
                                 type="primary", key="csv_btn")

            if run_btn:
                required = {"ml_model", "tfidf"} if "ML" in model_choice else set()
                if "LSTM" in model_choice:
                    required |= {"lstm", "tokenizer"}
                if "Ensemble" in model_choice:
                    required = {"ml_model", "tfidf", "lstm", "tokenizer"}
                missing = required - set(models)

                if missing:
                    st.error(f"Missing models: {', '.join(missing)}")
                else:
                    df_proc = df_up.head(max_rows).copy()
                    preds, confs = [], []

                    prog = st.progress(0, text="Running predictions…")
                    for i, text in enumerate(df_proc[text_col].fillna("").astype(str)):
                        try:
                            if len(text.split()) < 3:
                                preds.append("Too short"); confs.append(0.0)
                            elif "Ensemble" in model_choice:
                                lbl, prob, _, _ = predict_ensemble(text)
                                preds.append("AI Generated" if lbl else "Human Written")
                                confs.append(prob if lbl else 1 - prob)
                            elif "ML" in model_choice:
                                lbl, prob = predict_ml(text)
                                preds.append("AI Generated" if lbl else "Human Written")
                                confs.append(prob if lbl else 1 - prob)
                            else:
                                lbl, prob = predict_lstm(text)
                                preds.append("AI Generated" if lbl else "Human Written")
                                confs.append(prob if lbl else 1 - prob)
                        except Exception as e:
                            preds.append(f"Error"); confs.append(0.0)

                        if (i + 1) % 20 == 0:
                            prog.progress((i + 1) / len(df_proc),
                                          text=f"Processed {i+1}/{len(df_proc)}")

                    prog.empty()
                    df_proc["Prediction"]   = preds
                    df_proc["AI Confidence"] = [f"{c*100:.1f}%" for c in confs]

                    st.dataframe(df_proc[[text_col, "Prediction", "AI Confidence"]],
                                  use_container_width=True)

                    ai_c = preds.count("AI Generated")
                    hu_c = preds.count("Human Written")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Processed", len(preds))
                    c2.metric("🤖 AI Generated",  ai_c)
                    c3.metric("👤 Human Written",  hu_c)

                    out_buf = io.StringIO()
                    df_proc.to_csv(out_buf, index=False)
                    st.download_button("⬇️ Download Results CSV",
                                       out_buf.getvalue().encode(),
                                       "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error reading file: {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:.85rem;'>"
    "AI vs Human Text Detector · Built with Streamlit · "
    "TF-IDF + Classical ML & Bidirectional LSTM"
    "</p>",
    unsafe_allow_html=True,
)
