# Tradewise-Stock-Trading-Platform

A data-driven stock trading platform designed to support smarter investment decisions by integrating predictive modeling, sentiment analysis, and interactive dashboards. TradeWise empowers users with time series forecasts and real-time market sentiment, all visualized in a clean, intuitive interface.

##  Features

-  **Stock Price Forecasting** using ARIMA models (Auto ARIMA + cross-validation)
-  **Sentiment Analysis** of financial news using TextBlob and FinBERT
-  **Correlation Analysis** between stock prices, trading volume, sentiment, and S&P 500
-  **Interactive Dashboard** built with Streamlit and Plotly
-  Visualizations: Line plots, candlestick charts, area charts, histograms, scatter plots

##  Technologies Used

- **Python** (pandas, numpy, statsmodels, pmdarima)
- **NLP**: TextBlob, FinBERT (via Hugging Face)
- **Data Sources**: `yfinance` for stock prices, `NewsAPI` for headlines
- **Visualization**: Plotly, Matplotlib
- **Web Framework**: Streamlit
- **Version Control**: Git


##  Predictive Modelling (ARIMA)

- Time series forecasting for stocks including AAPL, AMZN, AMD, APO, GOOGL, and AMRK
- Log transformation, differencing, and ADF tests used to ensure stationarity
- Auto ARIMA used to find best (p,d,q) parameters
- Evaluation metrics: MAE, RMSE, Confidence Interval Width

### Example Forecasting Results

| Ticker | MAE  | RMSE | CI Width |
|--------|------|------|----------|
| AAPL   | 3.85 | 5.38 | 37.74    |
| AMD    | 4.05 | 5.26 | 42.50    |
| AMRK   | 1.90 | 2.64 | 13.07    |
| AMZN   | 3.85 | 4.98 | 44.28    |
| APO    | 7.23 | 8.14 | 20.46    |
| GOOGL  | 4.57 | 5.51 | 26.44    |

##  Sentiment Analysis

- News headlines retrieved from NewsAPI
- Sentiment classification using TextBlob and FinBERT
- Sentiment scores correlated with stock price changes
- Visual trend analysis of sentiment vs. price movement

##  Correlation Insights

- AAPL vs. S&P 500 → Strong positive correlation (r = 0.97)
- AAPL vs. News Sentiment → Weak correlation (r ≈ 0.01)
- Highlights the need to combine both technical and sentiment indicators

##  Dashboard Preview

- Built with Streamlit + Plotly for interactivity
- Features include:
  - Ticker filtering
  - SMA overlays
  - Candlestick charts
  - News sentiment tracking
  - Real-time API data caching

## Future Work

- Expand coverage to Forex and Crypto markets
- Integrate social media sentiment (Twitter, Reddit)
- Add predictive alerts and user customization features
- Explore LSTM or Transformer-based forecasting
