import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
import numpy as np

def resample_data(stock_data, freq_type):
    """
    Resample time series data to a specific frequency.

    Args:
    - stock_data (pd.DataFrame): DataFrame containing the time series data.
    - freq_type (str): The frequency to resample to (e.g., 'ME' for monthly, 'W' for weekly).

    Returns:
    - pd.Series: Resampled time series data.
    """
    return stock_data['Close'].resample(freq_type).mean()


def decompose_data(stock_data, model_type):
    """
    Decompose the time series data into trend, seasonal, and residual components.

    Args:
    - stock_data (pd.Series): Time series data to decompose.
    - model_type (str): Type of decomposition model ('additive' or 'multiplicative').

    Returns:
    - DecompositionResults: A result object with trend, seasonal, and residual components.
    """
    return seasonal_decompose(stock_data, model=model_type)


def plot_decomposition(decomposition):
    """
    Plot the decomposed components of the time series.

    Args:
    - decomposition (DecompositionResults): A result object with trend, seasonal, and residual components.
    """
    # Plot Trend Component
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=decomposition.trend.index,
                                   y=decomposition.trend,
                                   mode='lines',
                                   name='Trend'))
    fig_trend.update_layout(title='Trend Component',
                            xaxis_title='Date',
                            yaxis_title='Trend')
    st.plotly_chart(fig_trend)

    # Plot Seasonal Component
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=decomposition.seasonal.index,
                                      y=decomposition.seasonal,
                                      mode='lines',
                                      name='Seasonality'))
    fig_seasonal.update_layout(title='Seasonal Component',
                               xaxis_title='Date',
                               yaxis_title='Seasonality')
    st.plotly_chart(fig_seasonal)

    # Plot Residual Component
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=decomposition.resid.index,
                                      y=decomposition.resid,
                                      mode='lines',
                                      name='Residual'))
    fig_residual.update_layout(title='Residual Component',
                               xaxis_title='Date',
                               yaxis_title='Residual')
    st.plotly_chart(fig_residual)


def calculate_autocorrelation(stock_data_df, max_lag):
    """
    Calculate autocorrelation for different lags.

    Args:
    - stock_data_df (pd.DataFrame): DataFrame containing the time series data.
    - max_lag (int): Maximum number of lags to calculate autocorrelation for.

    Returns:
    - list: Autocorrelation values for lags from 1 to max_lag.
    """
    # Ensure 'Close' column is a Series with DatetimeIndex
    stock_data_monthly = stock_data_df['Close'].resample('ME').mean().dropna()

    # Calculate autocorrelation for each lag
    autocorr = [stock_data_monthly.autocorr(lag) for lag in range(1, max_lag + 1)]
    return autocorr


def visualize_autocorrelation(autocorr_values):
    """
    Visualize the autocorrelation values using Plotly.

    Args:
    - autocorr_values (list): List of autocorrelation values.
    """
    # Create Plotly bar chart for autocorrelation
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=np.arange(1, len(autocorr_values) + 1),
                             y=autocorr_values,
                             name='Autocorrelation'))

    fig_acf.update_layout(title='Autocorrelation Function (ACF)',
                          xaxis_title='Lags',
                          yaxis_title='Autocorrelation')

    # Display the ACF chart in Streamlit
    st.plotly_chart(fig_acf)

st.markdown(f'**Analyze the Seasonality for {st.session_state['ticker']}**')
stock_data = st.session_state['retrieved_data']
model_type = 'additive'
# Convert 'Date' column to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Set 'Date' column as the index
stock_data.set_index('Date', inplace=True)

resampled_data = resample_data(stock_data, 'M')

# Decompose the data
decomposed_data = decompose_data(resampled_data, model_type)

plot_decomposition(decomposed_data)

# Calculate the autocorrelations for lags up to 50
max_lag = 100
autocorr_values = calculate_autocorrelation(stock_data, max_lag)

# Visualize the autocorrelation
visualize_autocorrelation(autocorr_values)