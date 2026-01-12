# Bitcoin AI Analyst

A real-time Machine Learning dashboard that predicts Bitcoin price movements. This project fetches live market data, analyzes technical indicators using three different AI algorithms (Random Forest, SVM, Logistic Regression), and visualizes the forecast on an interactive dashboard.

## Features

* **Multi-Model AI:** Compare predictions from Random Forest, Logistic Regression, and Support Vector Machines (SVM).
* **Real-Time Data:** Fetches live Bitcoin prices via the CoinGecko API.
* **Interactive Charts:** Professional financial charts using Plotly with dynamic forecast lines.
* **Personalization:** Upload your own custom logo or university crest directly via the sidebar.
* **Robust Error Handling:** Includes a simulation mode that activates automatically if API rate limits are hit.
* **Technical Indicators:** Uses 7-Day & 30-Day Moving Averages and Volatility features.

## Tech Stack

* **Frontend:** Streamlit
* **Data Visualization:** Plotly
* **Machine Learning:** Scikit-Learn
* **Data Handling:** Pandas, NumPy
* **Image Processing:** Pillow
* **API:** CoinGecko (Free Tier)

## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/crypto-ai-dashboard.git](https://github.com/YOUR_USERNAME/crypto-ai-dashboard.git)
cd crypto-ai-dashboard
