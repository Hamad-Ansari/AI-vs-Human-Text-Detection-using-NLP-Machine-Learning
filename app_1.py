# ============================================
# 🚀 AI vs HUMAN TEXT DETECTOR - STREAMLIT APP
# ============================================
# 📌 Features:
#    - Single text prediction
#    - Batch CSV upload
#    - Sample texts to try
#    - Stylometric analysis dashboard
#    - Prediction history
# ============================================

import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import re
import string
import nltk
import time
from textblob import TextBlob
from collections import Counter
from datetime import datetime

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# 🎨 COLOR PALETTE
# ============================================
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c', 
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'dark': '#34495e',
    'light': '#ecf0f1'
}

# ============================================
# 📥 NLTK DATA DOWNLOAD
# ============================================
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)
    return True

download_nltk_data()

# ============================================
# ⚙️ HELPER CLASSES
# ============================================

class TextPreprocessor:
    """🔧 Text Preprocessing Pipeline"""
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def clean_text(self, text):
        """Basic cleaning"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def advanced_clean(self, text):
        """Advanced preprocessing with lemmatization"""
        text = self.clean_text(text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words]
        return ' '.join(tokens)


class StylometricAnalyzer:
    """🔬 Stylometric Feature Extraction"""
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def _count_syllables(self, word):
        word = word.lower()
        vowels = "aeiou"
        syllables = 0
        previous_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        if word.endswith('e'):
            syllables -= 1
        return max(syllables, 1)

    def extract_features(self, text):
        """Extract comprehensive stylometric features"""
        features = {}
        text = str(text)
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        words_no_punct = [w for w in words if w not in string.punctuation]

        # Length Features
        features['📏 Character Count'] = len(text)
        features['📝 Word Count'] = len(words_no_punct)
        features['📄 Sentence Count'] = len(sentences)
        features['🔤 Avg Word Length'] = round(np.mean([len(w) for w in words_no_punct]), 2) if words_no_punct else 0
        features['📊 Avg Sentence Length'] = round(features['📝 Word Count'] / features['📄 Sentence Count'], 2) if features['📄 Sentence Count'] > 0 else 0

        # Vocabulary Richness
        unique_words = set(words_no_punct)
        features['📚 Unique Words'] = len(unique_words)
        features['🎯 Vocabulary Richness'] = round(len(unique_words) / features['📝 Word Count'], 3) if features['📝 Word Count'] > 0 else 0

        # Lexical Features
        features['🛑 Stopword Ratio'] = round(sum(1 for w in words_no_punct if w in self.stop_words) / features['📝 Word Count'], 3) if features['📝 Word Count'] > 0 else 0
        features['❗ Punctuation Count'] = sum(1 for c in text if c in string.punctuation)

        # POS Tagging
        pos_tags = nltk.pos_tag(words_no_punct)
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        noun_count = sum(pos_counts[tag] for tag in ['NN', 'NNS', 'NNP', 'NNPS'])
        verb_count = sum(pos_counts[tag] for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        adj_count = sum(pos_counts[tag] for tag in ['JJ', 'JJR', 'JJS'])
        
        features['🏷️ Noun Ratio'] = round(noun_count / features['📝 Word Count'], 3) if features['📝 Word Count'] > 0 else 0
        features['🏃 Verb Ratio'] = round(verb_count / features['📝 Word Count'], 3) if features['📝 Word Count'] > 0 else 0
        features['✨ Adjective Ratio'] = round(adj_count / features['📝 Word Count'], 3) if features['📝 Word Count'] > 0 else 0

        # Sentiment
        blob = TextBlob(text)
        features['😊 Sentiment Polarity'] = round(blob.sentiment.polarity, 3)
        features['💭 Subjectivity'] = round(blob.sentiment.subjectivity, 3)

        # Readability (Flesch)
        syllable_count = sum(self._count_syllables(w) for w in words_no_punct)
        if features['📄 Sentence Count'] > 0 and features['📝 Word Count'] > 0:
            flesch = 206.835 - 1.015 * (features['📝 Word Count'] / features['📄 Sentence Count']) - 84.6 * (syllable_count / features['📝 Word Count'])
            features['📖 Readability Score'] = round(flesch, 2)
        else:
            features['📖 Readability Score'] = 0

        return features


# ============================================
# 💾 MODEL LOADING
# ============================================

@st.cache_resource
def load_models():
    """Load all models and assets"""
    try:
        assets = {
            "ml_model": joblib.load('models/best_ml_model.pkl'),
            "tfidf_vectorizer": joblib.load('models/tfidf_vectorizer.pkl'),
            "lstm_model": tf.keras.models.load_model('models/lstm_model.h5'),
            "lstm_tokenizer": pickle.load(open('models/lstm_tokenizer.pkl', 'rb')),
            "preprocessor": TextPreprocessor(),
            "stylometric_analyzer": StylometricAnalyzer()
        }
        return assets
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("📁 Please ensure the 'models' folder contains all required files.")
        return None


# ============================================
# 🔮 PREDICTION FUNCTIONS
# ============================================

def predict_single_text(text, model_type, assets):
    """Predict a single text"""
    if model_type == "LightGBM (ML)":
        processed = assets["preprocessor"].advanced_clean(text)
        vectorized = assets["tfidf_vectorizer"].transform([processed])
        prediction = assets["ml_model"].predict(vectorized)[0]
        probability = assets["ml_model"].predict_proba(vectorized)[0, 1]
    else:  # LSTM
        max_length = 200
        seq = assets["lstm_tokenizer"].texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_length)
        probability = float(assets["lstm_model"].predict(padded, verbose=0)[0, 0])
        prediction = 1 if probability > 0.5 else 0

    return {
        'prediction': prediction,
        'label': '🤖 AI Generated' if prediction == 1 else '👤 Human Written',
        'confidence': probability if prediction == 1 else 1 - probability,
        'ai_probability': probability
    }


def predict_batch(texts, model_type, assets, progress_bar=None):
    """Predict multiple texts"""
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == "":
            results.append({
                'text': text,
                'prediction': 'N/A',
                'label': '⚠️ Empty Text',
                'confidence': 0,
                'ai_probability': 0
            })
        else:
            result = predict_single_text(str(text), model_type, assets)
            result['text'] = text[:100] + "..." if len(str(text)) > 100 else text
            results.append(result)
        
        if progress_bar:
            progress_bar.progress((i + 1) / total)
    
    return results


# ============================================
# 📊 VISUALIZATION FUNCTIONS
# ============================================

def create_confidence_gauge(confidence, prediction):
    """Create a gauge chart for confidence"""
    color = COLORS['secondary'] if prediction == 1 else COLORS['success']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': 'white',
            'steps': [
                {'range': [0, 50], 'color': COLORS['light']},
                {'range': [50, 75], 'color': '#ffeaa7'},
                {'range': [75, 100], 'color': '#fab1a0'}
            ],
            'threshold': {
                'line': {'color': COLORS['dark'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['dark']}
    )
    
    return fig


def create_stylometric_radar(features):
    """Create radar chart for stylometric features"""
    # Select numeric features for radar
    radar_features = ['🎯 Vocabulary Richness', '🛑 Stopword Ratio', 
                      '🏷️ Noun Ratio', '🏃 Verb Ratio', '💭 Subjectivity']
    
    values = [float(features.get(f, 0)) for f in radar_features]
    labels = [f.split(' ', 1)[1] for f in radar_features]  # Remove emojis for chart
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color=COLORS['primary'], width=2),
        name='Text Features'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_batch_results_chart(results_df):
    """Create visualization for batch results"""
    counts = results_df['label'].value_counts()
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color_discrete_sequence=[COLORS['success'], COLORS['secondary'], COLORS['warning']],
        hole=0.4
    )
    
    fig.update_layout(
        title="📊 Batch Classification Results",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


# ============================================
# 📝 SAMPLE TEXTS
# ============================================

SAMPLE_TEXTS = {
    "🤖 AI Generated Sample 1": """
    Furthermore, the integration of artificial intelligence into modern healthcare systems 
    presents unprecedented opportunities for improving patient outcomes. The systematic 
    implementation of machine learning algorithms enables healthcare providers to analyze 
    vast quantities of medical data with remarkable precision. Consequently, this technological 
    advancement facilitates more accurate diagnoses and personalized treatment plans.
    """,
    
    "🤖 AI Generated Sample 2": """
    In conclusion, the impact of climate change on global ecosystems cannot be overstated. 
    Rising temperatures and shifting precipitation patterns are fundamentally altering 
    habitats worldwide. Moreover, these environmental changes pose significant challenges 
    to biodiversity conservation efforts. Therefore, immediate and coordinated action is 
    essential to mitigate these effects.
    """,
    
    "👤 Human Written Sample 1": """
    I remember the first time I saw the ocean. I was maybe 7 or 8, and my dad drove us 
    all the way from Ohio to Florida. The drive took forever, and my sister kept kicking 
    my seat. But when we finally got there and I saw all that water stretching out to 
    nowhere... man, I just stood there with my mouth open like an idiot. Mom has a photo 
    somewhere. I look ridiculous but happy.
    """,
    
    "👤 Human Written Sample 2": """
    So here's the thing about cooking - nobody tells you how much cleanup is involved! 
    I spent 2 hours making this fancy pasta dish I found online, and yeah it was pretty 
    good, but then I looked at my kitchen and just... sighed. Pots everywhere, sauce 
    splattered on the stove, somehow flour on the ceiling?? Still don't know how that 
    happened. Next time I'm ordering takeout lol.
    """,
    
    "🔄 Mixed/Ambiguous Sample": """
    The research indicates that sleep quality significantly affects cognitive performance. 
    Studies have shown that individuals who maintain consistent sleep schedules tend to 
    demonstrate improved memory retention. However, I've noticed that even when I sleep 
    8 hours, sometimes I still feel tired. Maybe it's the coffee? Anyway, the data suggests 
    that blue light exposure before bed might be a factor worth considering.
    """
}


# ============================================
# 🎨 STREAMLIT UI
# ============================================

# Page Config
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #3498db 0%, #9b59b6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .human-badge {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .ai-badge {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load models
assets = load_models()

# ============================================
# 📱 SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    model_choice = st.selectbox(
        "🤖 Select Model",
        ["LightGBM (ML)", "LSTM (Deep Learning)"],
        help="LightGBM: Fast & accurate | LSTM: Context-aware"
    )
    
    st.markdown("---")
    
    st.markdown("## 📊 Model Info")
    
    if model_choice == "LightGBM (ML)":
        st.info("""
        **LightGBM Model**
        - ⚡ Fast inference
        - 📈 ~96% accuracy
        - 🔤 Uses TF-IDF features
        - 📝 Best for general text
        """)
    else:
        st.info("""
        **LSTM Model**
        - 🧠 Deep learning based
        - 📈 ~95% accuracy
        - 🔗 Captures context
        - 📝 Better for long texts
        """)
    
    st.markdown("---")
    
    st.markdown("## 📜 History")
    if st.session_state.prediction_history:
        for item in st.session_state.prediction_history[-5:]:
            st.markdown(f"""
            <div class="feature-card">
                <small>{item['time']}</small><br>
                <strong>{item['label']}</strong><br>
                <small>Confidence: {item['confidence']:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.caption("No predictions yet")
    
    st.markdown("---")
    
    st.markdown("## 👨‍💻 About")
    st.markdown("""
    Built with ❤️ by **[Your Name]**
    
    📧 [Email](mailto:your@email.com)  
    🔗 [LinkedIn](https://linkedin.com/in/yourname)  
    🐙 [GitHub](https://github.com/yourname)
    """)


# ============================================
# 🏠 MAIN CONTENT
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI vs Human Text Detector</h1>
    <p>Advanced NLP-powered tool to distinguish AI-generated content from human writing</p>
</div>
""", unsafe_allow_html=True)

if assets is None:
    st.error("⚠️ Models not loaded. Please check your model files.")
    st.stop()

# Tabs for different testing modes
tab1, tab2, tab3 = st.tabs(["✍️ Single Text", "📁 Batch Upload", "🎯 Sample Texts"])

# ============================================
# TAB 1: SINGLE TEXT INPUT
# ============================================

with tab1:
    st.markdown("### 📝 Enter Text to Analyze")
    
    input_text = st.text_area(
        "Paste or type your text below:",
        height=200,
        placeholder="Enter any text here - an essay, article, email, or even a tweet. The model will analyze the writing style to determine if it was written by a human or generated by AI...",
        key="single_text_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
        if clear_btn:
            st.rerun()
    
    if analyze_btn and input_text.strip():
        with st.spinner("🔬 Analyzing text patterns..."):
            time.sleep(0.5)  # Small delay for UX
            
            # Get prediction
            result = predict_single_text(input_text, model_choice, assets)
            
            # Get stylometric features
            style_features = assets["stylometric_analyzer"].extract_features(input_text)
            
            # Save to history
            st.session_state.prediction_history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'text': input_text[:50] + "...",
                'label': result['label'],
                'confidence': result['confidence']
            })
        
        st.markdown("---")
        
        # Results Section
        st.markdown("### 🎯 Analysis Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if result['prediction'] == 0:
                st.markdown(f'<div class="human-badge">👤 Human Written</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-badge">🤖 AI Generated</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(create_confidence_gauge(result['confidence'], result['prediction']), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Stylometric Profile")
            st.plotly_chart(create_stylometric_radar(style_features), use_container_width=True)
        
        # Detailed Features
        st.markdown("### 🔬 Detailed Stylometric Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        feature_list = list(style_features.items())
        third = len(feature_list) // 3
        
        with col1:
            for key, value in feature_list[:third]:
                st.metric(label=key, value=value)
        
        with col2:
            for key, value in feature_list[third:2*third]:
                st.metric(label=key, value=value)
        
        with col3:
            for key, value in feature_list[2*third:]:
                st.metric(label=key, value=value)
        
        # Interpretation
        st.markdown("### 💡 Interpretation")
        
        if result['prediction'] == 1:
            st.warning(f"""
            **The text appears to be AI-generated** with {result['confidence']:.1%} confidence.
            
            🔍 **Potential indicators:**
            - Consistent sentence structure
            - Formal transition words
            - High vocabulary consistency
            - Neutral or balanced sentiment
            """)
        else:
            st.success(f"""
            **The text appears to be human-written** with {result['confidence']:.1%} confidence.
            
            🔍 **Potential indicators:**
            - Varied sentence patterns
            - Personal expressions or opinions
            - Natural irregularities
            - Emotional or subjective content
            """)
    
    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze.")


# ============================================
# TAB 2: BATCH UPLOAD
# ============================================

with tab2:
    st.markdown("### 📁 Batch Text Classification")
    st.markdown("Upload a CSV file with a column containing texts to classify multiple samples at once.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should have a column with text data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Show preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Select text column
            text_column = st.selectbox(
                "Select the column containing text:",
                options=df.columns.tolist()
            )
            
            # Limit rows option
            max_rows = st.slider(
                "Number of rows to process:",
                min_value=1,
                max_value=min(len(df), 500),
                value=min(len(df), 100)
            )
            
            if st.button("🚀 Run Batch Classification", type="primary"):
                st.markdown("---")
                st.markdown("#### 📊 Processing...")
                
                progress_bar = st.progress(0)
                texts = df[text_column].head(max_rows).tolist()
                
                results = predict_batch(texts, model_choice, assets, progress_bar)
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                st.success(f"✅ Classified {len(results)} texts!")
                
                # Visualization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(create_batch_results_chart(results_df), use_container_width=True)
                
                with col2:
                    # Summary stats
                    human_count = len(results_df[results_df['prediction'] == 0])
                    ai_count = len(results_df[results_df['prediction'] == 1])
                    avg_confidence = results_df['confidence'].mean()
                    
                    st.markdown("#### 📈 Summary Statistics")
                    st.metric("👤 Human Written", human_count)
                    st.metric("🤖 AI Generated", ai_count)
                    st.metric("🎯 Average Confidence", f"{avg_confidence:.1%}")
                
                # Results table
                st.markdown("#### 📋 Detailed Results")
                
                display_df = results_df[['text', 'label', 'confidence']].copy()
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                display_df.columns = ['Text (Preview)', 'Classification', 'Confidence']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    else:
        # Show sample format
        st.markdown("#### 📝 Expected CSV Format")
        sample_df = pd.DataFrame({
            'text': [
                "This is the first sample text to classify...",
                "Here's another example of text content...",
                "A third sample for demonstration purposes..."
            ],
            'source': ['sample1', 'sample2', 'sample3']
        })
        st.dataframe(sample_df, use_container_width=True)
        
        # Download sample
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="sample_input.csv",
            mime="text/csv"
        )


# ============================================
# TAB 3: SAMPLE TEXTS
# ============================================

with tab3:
    st.markdown("### 🎯 Test with Sample Texts")
    st.markdown("Try these pre-loaded examples to see how the model performs on known AI and human texts.")
    
    # Sample selection
    selected_sample = st.selectbox(
        "Choose a sample text:",
        options=list(SAMPLE_TEXTS.keys())
    )
    
    # Display selected sample
    sample_text = SAMPLE_TEXTS[selected_sample]
    
    st.markdown("#### 📝 Selected Text")
    st.text_area(
        "Sample content:",
        value=sample_text.strip(),
        height=150,
        disabled=True,
        key="sample_display"
    )
    
    # Ground truth
    if "AI Generated" in selected_sample:
        st.info("🏷️ **Ground Truth:** This text is AI-generated")
    elif "Human Written" in selected_sample:
        st.info("🏷️ **Ground Truth:** This text is human-written")
    else:
        st.info("🏷️ **Ground Truth:** This is a mixed/ambiguous sample")
    
    if st.button("🔍 Analyze Sample", type="primary", key="analyze_sample"):
        with st.spinner("🔬 Analyzing..."):
            time.sleep(0.5)
            
            result = predict_single_text(sample_text, model_choice, assets)
            style_features = assets["stylometric_analyzer"].extract_features(sample_text)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Prediction")
            
            if result['prediction'] == 0:
                st.markdown(f'<div class="human-badge">👤 Human Written</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-badge">🤖 AI Generated</div>', unsafe_allow_html=True)
            
            st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Check accuracy
            if "AI Generated" in selected_sample and result['prediction'] == 1:
                st.success("✅ Correct classification!")
            elif "Human Written" in selected_sample and result['prediction'] == 0:
                st.success("✅ Correct classification!")
            elif "Mixed" in selected_sample:
                st.info("ℹ️ Mixed sample - classification may vary")
            else:
                st.error("❌ Incorrect classification")
        
        with col2:
            st.markdown("#### 📊 Stylometric Profile")
            st.plotly_chart(create_stylometric_radar(style_features), use_container_width=True)
    
    # Test all samples
    st.markdown("---")
    st.markdown("#### 🧪 Test All Samples")
    
    if st.button("🚀 Run All Samples", key="run_all"):
        results_summary = []
        
        progress = st.progress(0)
        
        for i, (name, text) in enumerate(SAMPLE_TEXTS.items()):
            result = predict_single_text(text, model_choice, assets)
            
            # Determine ground truth
            if "AI Generated" in name:
                ground_truth = 1
            elif "Human Written" in name:
                ground_truth = 0
            else:
                ground_truth = -1  # Unknown
            
            correct = "✅" if result['prediction'] == ground_truth else ("➖" if ground_truth == -1 else "❌")
            
            results_summary.append({
                'Sample': name,
                'Prediction': result['label'],
                'Confidence': f"{result['confidence']:.1%}",
                'Result': correct
            })
            
            progress.progress((i + 1) / len(SAMPLE_TEXTS))
        
        st.dataframe(pd.DataFrame(results_summary), use_container_width=True)


# ============================================
# 📝 FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 AI vs Human Text Detector | Built with Streamlit & TensorFlow</p>
    <p>⚠️ Note: This tool is for educational purposes. No detection method is 100% accurate.</p>
</div>
""", unsafe_allow_html=True)