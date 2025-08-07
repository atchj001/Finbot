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

- `data/`
  - `question_bank.json` – Curated Q&A dataset with tiered literacy responses
- `models/`
  - `cnn_model.h5` – Trained CNN model for stock forecasting
  - `lstm_model.h5` – Trained LSTM model for comparative analysis
  - `scaler.pkl` – Preprocessing scaler for normalizing input data
- `shap_visuals/`
  - `sample_shap_output.png` – Example SHAP explanation for transparency
- `feedback/`
  - `feedback_log.json` – Stores structured user feedback
  - `poor_ratings_export.xlsx` – Export of negative responses for review
- `src/`
  - `main.py` – Main GUI application launcher
  - `qa_module.py` – Retrieval-based question answering logic
  - `prediction_module.py` – CNN & LSTM implementation + SHAP
  - `sentiment_analysis.py` – Investment sentiment via news headlines
  - `feedback_module.py` – Rating capture and export logic
- `requirements.txt` – List of Python dependencies
- `Finbot report.pdf` – Full academic dissertation
- `README.md` – Project documentation (this file)



---

## Key libraries

PySimpleGUI

tensorflow, keras

scikit-learn

pandas, numpy

shap

textblob

yfinance

matplotlib


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
