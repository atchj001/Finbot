# 📊 Finbot: A Literacy-Aware Financial Assistant with Explainable Investment Insights

**Finbot** is an intelligent, retrieval-based financial assistant that provides personalized guidance tailored to users’ financial literacy levels. It incorporates explainable AI for stock forecasting using CNN and LSTM models, enhanced with SHAP visualizations, and integrates user feedback for continuous improvement.

---

## 🚀 Features

- **🧠 Literacy-Tiered Q&A** – Beginner and Advanced response modes based on user selection.
- **📚 Retrieval-Based Question Answering** – Uses TF-IDF + cosine similarity with optimized thresholds.
- **📈 Stock Price Prediction** – CNN & LSTM models trained on historical data.
- **🧮 Explainable AI with SHAP** – Visualizes which data points influenced stock predictions.
- **📰 Sentiment-Based Investment Suggestions** – Analyzes real-time news headlines.
- **🖥️ PySimpleGUI Interface** – Clean, responsive GUI with tabs, chat bubbles, and charts.
- **📊 Feedback Collection System** – Logs helpfulness ratings and exports poor responses for analysis.

---

## 📁 Project Structure

Finbot/
├── data/
│ └── question_bank.json # Curated Q&A with tiered answers
├── models/
│ ├── cnn_model.h5 # Trained CNN model
│ ├── lstm_model.h5 # Trained LSTM model
│ └── scaler.pkl # Scaler for preprocessing
├── shap_visuals/
│ └── sample_shap_output.png # Example SHAP explanation
├── feedback/
│ ├── feedback_log.json # Logs user ratings
│ └── poor_ratings_export.xlsx # Exported poor responses
├── src/
│ ├── main.py # Main app launcher
│ ├── qa_module.py # Question Answering logic
│ ├── prediction_module.py # CNN/LSTM + SHAP prediction
│ ├── sentiment_analysis.py # News sentiment scoring
│ └── feedback_module.py # Logging + export handling
├── requirements.txt # Python dependencies
├── Finbot report.pdf # Full dissertation
└── README.md # Project readme


---

## 🧠 Methodology

- **TF-IDF + Cosine Similarity**: QA engine with a threshold sensitivity analysis (optimal: `0.1–0.2`)
- **Prediction Models**:
  - `CNN` – For volatile stocks (TSLA, NVDA)
  - `LSTM` – For stable stocks (PLUG)
- **SHAP**: KernelExplainer used for local explainability
- **Sentiment Analysis**: Uses `TextBlob` on news headlines to provide Buy/Hold/Sell labels

---

## 📊 Evaluation Summary

- **UEQ Improvements** *(Beta → Final)*:
  - Attractiveness: `+1.81`
  - Dependability: `+1.54`
  - Stimulation: `+0.92`

- **Model Comparison**:
  - CNN outperformed LSTM on high-volatility stocks
  - LSTM outperformed CNN on more stable stocks
- **Threshold Optimization**:
  - Best QA match threshold: `0.1` (high recall without sacrificing relevance)

---


## Main capabilities:

Ask finance questions by literacy level

View stock predictions + SHAP explanations

Get investment sentiment from news

Submit feedback to help improve Finbot
