import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

ticker_list = ['AAPL', 'GOOGL', 'AMZN', 'AMD', 'AMRK', 'APO']

earlier_years = 5
end_date = datetime.now().date()
start_date = end_date - relativedelta(years=earlier_years)


def retrieve_data(ticker_list, start_date, end_date):
    # Initialize an empty list to store DataFrames
    ticker_df_list = []

    for ticker in ticker_list:
        try:
            # Download data for the ticker
            ticker_df = yf.download(ticker,
                                    start=start_date,
                                    end=end_date)

            # Check if the DataFrame is empty
            if ticker_df.empty:
                print(f"No data retrieved for {ticker}.")
                continue

            # Add a 'ticker' column
            ticker_df['ticker'] = ticker

            # Reset index and select relevant columns
            ticker_df = ticker_df.reset_index()

            ticker_df = ticker_df.loc[:, ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Convert 'Date' column to desired format
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


ticker_historial_data = retrieve_data(ticker_list, start_date, end_date)
st.write(ticker_historial_data)