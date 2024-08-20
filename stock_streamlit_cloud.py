import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
from textblob import TextBlob
import requests
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Ticker symbols
ticker_list = ['AAPL', 'GOOGL', 'AMZN', 'AMD', 'AMRK', 'APO']

# Set the date range
earlier_years = 5
end_date = datetime.now().date()
start_date = end_date - relativedelta(years=earlier_years)

# Function to retrieve data
def retrieve_data(ticker_list, start_date, end_date):
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = yf.download(ticker, start=start_date, end=end_date)
            if ticker_df.empty:
                print(f"No data retrieved for {ticker}.")
                continue
            ticker_df['ticker'] = ticker
            ticker_df = ticker_df.reset_index()
            ticker_df = ticker_df.loc[:, ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            ticker_df['Date'] = ticker_df['Date'].dt.strftime('%Y-%m-%d')
            ticker_df_list.append(ticker_df)
        except Exception as e:
            print(f"An error occurred while retrieving data for {ticker}: {e}")

    if ticker_df_list:
        combined_df = pd.concat(ticker_df_list, ignore_index=True)
        return combined_df
    else:
        print("No data to return.")
        return pd.DataFrame()
    
news_api_key = 'a4bb177a2e774dc4abdafd5dbf9c7eeb'

# Fetch news articles using News API
def fetch_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    
    # Debugging: Print the status code and response content
    print(f"News API Status Code: {response.status_code}")
    print(f"News API Response: {news_data}")
    
    if response.status_code == 200:
        articles = news_data.get('articles', [])
        return articles
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Perform sentiment analysis on fetched articles
def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        if 'title' in article:
            analysis = TextBlob(article['title'])
            sentiment_score = analysis.sentiment.polarity
            sentiments.append(sentiment_score)
            
            # Debugging: Print the sentiment score for each article
            print(f"Article: {article['title']}\nSentiment Score: {sentiment_score}\n")
        else:
            print("No title found in article.")
        
    return sentiments

# Function to create candlestick chart
def create_candlestick_chart(df, ticker):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'))
    fig.update_layout(title=f'Candlestick chart for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    return fig

# Function to create a moving average chart
def create_moving_average_chart(df, ticker):
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='20-day MA'))
    fig.update_layout(title=f'Moving Averages for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price')
    return fig

# Retrieve the data
ticker_historial_data = retrieve_data(ticker_list, start_date, end_date)

# Streamlit app layout
st.set_page_config(page_title="Trading Monitoring Platform", layout="wide")

# Sidebar options for filtering
st.sidebar.title("Stock Info")
selected_ticker = st.sidebar.selectbox("Select a ticker", ticker_list)

# Date range selector
selected_start_date = st.sidebar.date_input("Start date", value=start_date)
selected_end_date = st.sidebar.date_input("End date", value=end_date)

# Chart type selector
chart_type = st.sidebar.selectbox(
    "Select chart type",
    ("Line Chart - Close Price", "Bar Chart - Volume", "Candlestick Chart", "Moving Averages")
)

# Filter data based on user selection
filtered_data = ticker_historial_data[
    (ticker_historial_data['ticker'] == selected_ticker) &
    (ticker_historial_data['Date'] >= selected_start_date.strftime('%Y-%m-%d')) &
    (ticker_historial_data['Date'] <= selected_end_date.strftime('%Y-%m-%d'))
]

# Main content
st.title(f"{selected_ticker} Stock Data")

# Display the filtered data
st.dataframe(filtered_data)

# Plot based on selected chart type
if chart_type == "Line Chart - Close Price":
    st.subheader(f"Closing Price of {selected_ticker}")
    st.line_chart(filtered_data.set_index('Date')['Close'])
elif chart_type == "Bar Chart - Volume":
    st.subheader(f"Volume of {selected_ticker}")
    st.bar_chart(filtered_data.set_index('Date')['Volume'])
elif chart_type == "Candlestick Chart":
    st.plotly_chart(create_candlestick_chart(filtered_data, selected_ticker))
elif chart_type == "Moving Averages":
    st.plotly_chart(create_moving_average_chart(filtered_data, selected_ticker))

# Sentiment Analysis
st.sidebar.subheader("Market Sentiment")
articles = fetch_news(news_api_key, selected_ticker)
sentiments = analyze_sentiment(articles)

# Display sentiment analysis results
if sentiments:
    st.write(f"Sentiment analysis for {selected_ticker}:")
    st.bar_chart(pd.DataFrame(sentiments, columns=["Sentiment"]))
else:
    st.write("No sentiment data available.")

# Footer
st.markdown("""
    <style>
        .footer {
            font-size: 12px;
            text-align: center;
            padding: 10px;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f9f9f9;
            color: #555;
        }
    </style>
    <div class="footer">
        <p>Trading Monitoring Platform &copy; 2024</p>
    </div>
""", unsafe_allow_html=True)
