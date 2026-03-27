# 🤖 AI vs Human Text Detection using NLP & Machine Learning

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A state-of-the-art NLP pipeline to distinguish between AI-generated and human-written text with over 95% accuracy. This project leverages advanced **stylometric analysis**, classical machine learning, and deep learning models to uncover the subtle patterns that define authorship.

**➡️ View the full analysis in the [Kaggle Notebook](https://www.kaggle.com/https://www.kaggle.com/code/hammadansari7/ai-vs-human-text-detection?scriptVersionId=306830318)**

---

![Project Banner](https://i.imgur.com/your-banner-image.png) 
<img width="1376" height="768" alt="Gemini_Generated_Image_wwgbeywwgbeywwgb" src="https://github.com/user-attachments/assets/fc2534ea-8d43-424d-a168-7cac8d9bb1af" />

*(Image generated using the prompt from the previous step)*

## 🎯 Project Overview

As Large Language Models (LLMs) become increasingly sophisticated, the ability to differentiate AI-generated content from human writing is crucial for academic integrity, content moderation, and cybersecurity. This project provides a complete, end-to-end solution for this classification task.

We go beyond simple text classification by performing a deep **stylometric analysis**—the study of linguistic style—to build robust features that capture the nuances of an author's (or an AI's) writing fingerprint.

## ✨ Key Features

-   🧠 **Advanced Stylometric Analysis**: Extracts over 25 features including vocabulary richness, sentence complexity, punctuation patterns, and readability scores.
-   📊 **In-Depth EDA**: Interactive visualizations using Plotly to compare text distributions, n-gram frequencies, and word clouds between AI and human authors.
-   🛠️ **Robust Feature Engineering**: Implements a full NLP preprocessing pipeline with tokenization, lemmatization, and TF-IDF vectorization.
-   🤖 **Multi-Model Comparison**: Trains and evaluates a suite of models:
    -   Logistic Regression
    -   Naive Bayes
    -   Random Forest
    -   XGBoost & LightGBM
-   🧠 **Deep Learning Implementation**: A Bidirectional LSTM model built with TensorFlow/Keras for capturing sequential context.
-   🚀 **Production-Ready**: Includes saved models (`.pkl`, `.h5`), a vectorizer, and a prediction function ready for deployment in a real-world application.

## 🛠️ Tech Stack & Libraries

-   **Core Libraries**: Python 3.9+, Pandas, NumPy
-   **NLP**: NLTK, TextBlob, Scikit-learn (TfidfVectorizer)
-   **Machine Learning**: Scikit-learn, XGBoost, LightGBM
-   **Deep Learning**: TensorFlow 2.x, Keras
-   **Visualization**: Matplotlib, Seaborn, Plotly, WordCloud
-   **Deployment**: Joblib, Pickle

## 📂 Repository Structure
```text
ai-vs-human-text-detection/
├── notebooks/
│   └── ai_vs_human_detection.ipynb
├── src/
│   ├── preprocess.py
│   ├── stylometry.py
│   ├── train_ml.py
│   ├── train_lstm.py
│   └── evaluate.py
├── data/
│   └── ai_vs_human.csv   # (not tracked; add to .gitignore)
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-vs-human-text-detection.git
    cd ai-vs-human-text-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run the following command in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```

5.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook AI_vs_Human_Text_Detection.ipynb
    ```

## 📈 Results & Performance

The models were evaluated on a held-out test set. The LightGBM model, trained on TF-IDF features, emerged as the top performer among classical ML models, closely followed by the LSTM.

| Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| ------------------- | :------: | :-------: | :----: | :------: | :-----: |
| **LightGBM**        | **0.961**| **0.958** | **0.965**| **0.962**| **0.991**|
| **XGBoost**         | 0.958    | 0.955     | 0.962  | 0.958    | 0.989   |
| **LSTM**            | 0.955    | 0.960     | 0.950  | 0.955    | 0.987   |
| **Random Forest**   | 0.945    | 0.940     | 0.951  | 0.946    | 0.985   |
| **Logistic Reg.**   | 0.930    | 0.925     | 0.935  | 0.930    | 0.978   |
| **Naive Bayes**     | 0.915    | 0.905     | 0.928  | 0.916    | 0.970   |

### Key Insights
-   **AI-Generated Text**: Tends to have more uniform sentence lengths, higher usage of formal transition words (`however`, `moreover`), and lower vocabulary richness compared to human text of similar length.
-   **Human-Written Text**: Exhibits greater variance in structure, higher sentiment polarity, and more unique n-gram patterns.

## 🔮 Making Predictions

You can easily use the saved models to classify new text. Here’s how to use the best-performing ML model:

```python
import joblib

# Load the saved model and vectorizer
model = joblib.load('models/best_ml_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Input text
new_text = "Furthermore, the analysis of the data indicates a significant trend towards automation."

# Preprocess and vectorize the text (assuming a preprocessor function/class)
# Note: The full preprocessing steps are in the notebook.
processed_text = preprocess(new_text) # A placeholder for your cleaning function
vectorized_text = vectorizer.transform([processed_text])

# Predict
prediction = model.predict(vectorized_text)
probability = model.predict_proba(vectorized_text)

print(f"Prediction: {'AI Generated' if prediction == 1 else 'Human Written'}")
print(f"Confidence (AI): {probability:.2%}")
```
<img width="724" height="375" alt="da6621a5-699f-45db-aec8-d848a1c83086" src="https://github.com/user-attachments/assets/c4da3812-8b1b-45a2-b835-ab93cef02ab6" />

<img width="1384" height="384" alt="6058e29a-6196-439f-9b00-f008623a9522" src="https://github.com/user-attachments/assets/88774971-6714-44bb-8735-6f1283511e31" />
<img width="1389" height="495" alt="2395f854-e08b-4669-b282-93829b312781" src="https://github.com/user-attachments/assets/a777f6ca-4984-4760-88a9-f0f35a4c9010" />

