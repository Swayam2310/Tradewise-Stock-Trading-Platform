import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from datetime import datetime

@st.cache_data
def fetch_sp500_data(stock_data):
    """
        retrieve S&P 500 data based on the date range from the stock data
    :param stock_data:
    :return:
    """
    # Get the min and max date from the stock_data
    start_date = stock_data['Date'].min()
    end_date = stock_data['Date'].max()

    # Fetch the S&P 500 data from yfinance
    sp500_ticker = "^GSPC"  # S&P 500 ticker symbol
    sp500_data = yf.download(sp500_ticker,
                             start=start_date,
                             end=end_date)

    # Reset index to make 'Date' a column and ensure dates align
    sp500_data.reset_index(inplace=True)

    # Return only the necessary columns ('Date' and 'Close')
    return sp500_data[['Date', 'Close']]

def scatter_plot(stock_data):
    """
    Analyze the correlation between the Closing Price and the volume and visualize with Scatter
    :param stock_data:
    :return:
    """
    fig = px.scatter(stock_data,
                     x='Close',
                     y='Volume',
                     title="Scatter Plot of Close Price vs Volume")
    # renders the Plotly chart inside the Streamlit app
    st.plotly_chart(fig)


def corre_stock_sp500(stock_data, sp500_data):
    # Ensure date columns are datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])

    # Merge data on matching dates
    merged_data = pd.merge(stock_data,
                           sp500_data,
                           on='Date',
                           how='inner',
                           suffixes=('_stock', '_sp500'))

    # Calculate the correlation between stock's Close price and SP500's Close price
    correlation = merged_data['Close_stock'].corr(merged_data['Close_sp500'])

    # Determine the description based on the correlation value
    if correlation >= 0.8:
        description = "Very strong positive association"
    elif correlation >= 0.6:
        description = "Strong positive association"
    elif correlation >= 0.4:
        description = "Moderate positive association"
    elif correlation >= 0.2:
        description = "Weak positive association"
    elif correlation > 0:
        description = "Very weak positive association"
    elif correlation > -0.2:
        description = "Very weak negative association"
    elif correlation > -0.4:
        description = "Weak negative association"
    elif correlation > -0.6:
        description = "Moderate negative association"
    elif correlation > -0.8:
        description = "Strong negative association"
    elif correlation >= -1.0:
        description = "Very strong negative association"
    else:
        description = "Perfect negative association"

    # Display the correlation result and description
    st.markdown(f"### Correlation between {st.session_state['ticker']} and S&P 500 Close Prices")
    st.write(f"The value of Correlation is: **{correlation:.4f}**")
    st.write(f"Which mean they have: {description}")

    # Create a scatter plot between stock 'Close' price and S&P 500 'Close' price
    fig = px.scatter(merged_data,
                     x='Close_stock',
                     y='Close_sp500',
                     labels={'Close_stock': 'Stock Close Price', 'Close_sp500': 'S&P 500 Close Price'},
                     title=f"Scatter Plot of {st.session_state['ticker']} vs S&P 500",
                     trendline="ols")

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Invoke the function scatter_plot
if 'retrieved_data' in st.session_state:
    # This function will only be called if retrieved_data is in the st.session_state
    scatter_plot(st.session_state['retrieved_data'])

    # Fetch S&P 500 data based on stock date range
    sp500_data = fetch_sp500_data(st.session_state['retrieved_data'])

    # Store the S&P 500 data in session state
    st.session_state['sp500_data'] = sp500_data

    # Plot the scatter plot
    corre_stock_sp500(st.session_state['retrieved_data'], sp500_data)
else:
    # st.warning() is used to display a warning message to the user if the 'retrieved_data'
    #   is not available in the session state.
    st.warning("Please go to page 'Company Info' and 'Stock Price' to select the specific stock.")