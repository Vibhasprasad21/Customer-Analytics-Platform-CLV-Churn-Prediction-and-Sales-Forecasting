#🚀 Customer-Analytics-Platform-CLV-Churn-Prediction-and-Sales-Forecasting


An AI-powered system that combines **Customer Lifetime Value (CLV) prediction**, **Churn Detection**, and **Sales Forecasting** into a single, integrated business intelligence solution.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)

---

## 🧠 Overview

This project addresses a major gap in customer analytics by unifying three critical business intelligence tools:
- **CLV Estimation** using Gamma-Gamma model
- **Churn Prediction** using XGBoost
- **Sales Forecasting** using Bidirectional LSTM

It empowers organizations with predictive insights to improve **retention**, **marketing strategies**, and **revenue planning**.

---

## ✨ Features

- 📈 **CLV Prediction**: Identify high-value customers for targeted marketing.
- ⚠️ **Churn Detection**: Spot at-risk customers using behavioral patterns.
- 🔮 **Sales Forecasting**: Time-series models for sales trend prediction.
- 🧠 **AI-Powered Dashboard**: Real-time interactive visualizations via Streamlit.
- 📊 **Customer Segmentation**: Visual insights based on CLV and churn risk.
- 📎 **Report Generation**: Export Excel reports for business decision-making.

---

## 🧰 Tech Stack

| Layer          | Technology |
|----------------|------------|
| Programming    | Python 3.8 |
| Frontend       | Streamlit |
| ML Libraries   | XGBoost, TensorFlow/Keras, Lifetimes |
| Visualization  | Plotly, Seaborn, Matplotlib |
| Data Handling  | Pandas, NumPy |
| Deployment     | Streamlit Community Cloud |

---

## 🏗️ System Architecture

1. **Data Upload**: CSV/Excel input
2. **Preprocessing**: Missing value handling, encoding, scaling
3. **ML Models**:
   - Gamma-Gamma → CLV Prediction
   - XGBoost → Churn Detection
   - BiLSTM → Sales Forecasting
4. **Visualization**: Interactive dashboard with metrics, trends, insights
5. **Export**: Downloadable Excel reports

---


🧪 Installation
bash
Copy
Edit
# Clone the repo
git clone https://github.com/yourusername/customer-analytics-ai.git
cd customer-analytics-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
▶️ Usage
bash
Copy
Edit
streamlit run app.py
Navigate to http://localhost:8501

Upload your dataset

Select CLV, Churn, or Sales Forecast analysis

Visualize insights and download reports

🧾 Data Format
The uploaded CSV should contain:


Column	Description
Customer ID	Unique identifier
Purchase Date	Date of transaction
Transaction Amount	Purchase value
Region/Segment	Optional - enriches analytics
📊 Results
🔍 CLV: Accurate revenue estimation with low Mean Absolute Error (MAE)

🧪 Churn: High precision/recall using XGBoost (ROC-AUC validated)

📉 Sales Forecast: LSTM-based low MAPE/RMSE prediction accuracy

🚀 Future Scope
Real-time streaming using Kafka

Transformer-based forecasting models

Automated model retraining pipelines

CRM/e-commerce platform integrations

Mobile-friendly UI and chatbot support


