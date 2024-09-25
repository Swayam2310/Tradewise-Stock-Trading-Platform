from datetime import datetime

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def retrieve_data(ticker, start_date, end_date):
    # Initialize an empty list to store DataFrames
    ticker_df_list = []

    try:
        # Download data for the ticker
        ticker_df = yf.download(ticker,
                                start=start_date,
                                end=end_date)

        # Check if the DataFrame is empty
        if ticker_df.empty:
            print(f"No data retrieved for {ticker}.")

        # Add a 'ticker' column
        ticker_df['ticker'] = ticker

        # Reset index and select relevant columns
        ticker_df = ticker_df.reset_index()

        ticker_df = ticker_df.loc[:, ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Convert 'Date' column to 'YYYY-MM-DD' format
        ticker_df['Date'] = ticker_df['Date'].dt.strftime('%Y-%m-%d')

        # Append the DataFrame to the list
        ticker_df_list.append(ticker_df)

    except Exception as e:
        print(f"An error occurred while retrieving data for {ticker}: {e}")

    # Concatenate all DataFrames into one
    if ticker_df_list:
        combined_df = pd.concat(ticker_df_list, ignore_index=True)
        return combined_df
    else:
        print("No data to return.")

        # Return an empty DataFrame if no data was retrieved
        return pd.DataFrame()


def visualize_data(stock_data, chart_type):
    # Display the selected chart
    if chart_type == "Line Chart":
        fig = px.line(stock_data,
                      x='Date',
                      y='Close',
                      title= f'Line Chart of Closing Prices of Stock {stock_data["ticker"].unique()[0]}',
                      labels={'Close': 'Closing Price'})
        st.plotly_chart(fig)

    elif chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(x=stock_data['Date'],
                                             open=stock_data['Open'],
                                             high=stock_data['High'],
                                             low=stock_data['Low'],
                                             close=stock_data['Close'])])
        fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

    elif chart_type == "Area Chart":
        fig = px.area(stock_data,
                      x=stock_data['Date'],
                      y='Volume',
                      title='Area Chart of Volume',
                      labels={'Volume': 'The amount of trading(Volume)'})
        st.plotly_chart(fig)

    elif chart_type == "Histogram":
        st.subheader("Histogram of Daily Returns")
        stock_data['Return'] = stock_data['Close'].pct_change().dropna()
        fig = px.histogram(stock_data,
                           x='Return',
                           title='Histogram of Daily Returns')
        fig.update_layout(xaxis_title='Daily Returns', yaxis_title='Frequency')
        st.plotly_chart(fig)


# Define an input that allows to select date
start_date = st.sidebar.date_input("Starting Date",
                                   min_value=datetime(1980, 1, 1),
                                   max_value=datetime(2015, 1, 1),
                                   value=datetime(2015, 1, 1)
                                   )

end_date = st.sidebar.date_input("Ending Date",
                                 min_value=datetime(2015, 1, 1),
                                 max_value=datetime.now()
                                 )

chart_type = ["Line Chart", "Candlestick", "Area Chart", "Histogram"]
selected_chart = st.sidebar.selectbox("Select Chart Type", chart_type)

st.markdown(f"### Stock Price Info: {st.session_state['ticker']}")
st.session_state['retrieved_data'] = retrieve_data(st.session_state['ticker'], start_date, end_date)

# Check if retrieved_data is not None and has data
if 'retrieved_data' in st.session_state:
    # 1. Visualize the selected stock price info in a DataFrame
    with st.container():
        st.dataframe(st.session_state['retrieved_data'],
                     height=318,
                     use_container_width=True,
                     hide_index=True,
                     column_order=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # 2. Visualize the selected stock price info with a chart
    with st.container():
        visualize_data(st.session_state['retrieved_data'], selected_chart)
else:
    st.warning("No data available for the selected ticker. Please try a different range or ticker.")