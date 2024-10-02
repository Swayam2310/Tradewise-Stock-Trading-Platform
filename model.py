import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')

file = 'arima.csv'
stock_data = pd.read_csv(file)

stock_data.head()

missing_values = stock_data.isnull().sum()
missing_values


stock_data= stock_data.drop(columns=['Unnamed: 0'])
stock_data


data_types = stock_data.dtypes
data_types


stock_data['ticker'] = stock_data['ticker'].astype(str)

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

stock_data.set_index('Date', inplace=True)

stock_data.head()


unique_tickers = stock_data['ticker'].unique()
unique_tickers


# Creating a dictionary to name each ticker
ticker_names = {
    'AAPL': 'Apple',
    'GOOGL': 'Google',
    'AMZN': 'Amazon',
    'AMD': 'Advanced Micro Devices',
    'AMRK': 'A-Mark Precious Metals',
    'APO': 'Apollo'
}

stock_data['company_name'] = stock_data['ticker'].map(ticker_names)
stock_data.head()
stock_data.tail()


# Basic Statistics for Each Ticker
grouped_data = stock_data.groupby('ticker')

# Loop through each group and display the statistics for each ticker
for ticker, data in grouped_data:
    print(f"Statistics for {ticker}:\n")
    print(data.describe())
    print("\n" + "-"*75 + "\n")


# Closing Price Over Time
grouped_data = stock_data.groupby('ticker')

for ticker, data in grouped_data:
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['Close'], label=f'{ticker} Close Price', color='blue')
    
    plt.title(f'{ticker} Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    
    plt.grid(True)
    plt.legend()
    
    plt.show()


#Stock Price Trend with Moving Average (SMA)
grouped_data = stock_data.groupby('ticker')

for ticker, data in grouped_data:
    # Calculating 50-day and 200-day Simple Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    plt.figure(figsize=(10,6))
    
    plt.plot(data.index, data['Close'], label=f'{ticker} Close Price', color='blue')
    
    plt.plot(data.index, data['SMA_50'], label=f'{ticker} 50-day SMA', color='red', linestyle='--')
    
    plt.plot(data.index, data['SMA_200'], label=f'{ticker} 200-day SMA', color='yellow', linestyle='--')
    
    plt.title(f'{ticker} Close Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    plt.legend()
    
    plt.grid(True)
    
    plt.show()


#Relation between Closing price and Volume
for ticker, data in grouped_data:
    correlation = data['Close'].corr(data['Volume'])
    print(f'Correlation between Volume and Close price for {ticker}: {correlation}')



#Rolling Average of Volume 
for ticker, data in grouped_data:
    data['Volume_Rolling_Avg'] = data['Volume'].rolling(window=30).mean()

    plt.figure(figsize=(10,6))
    plt.fill_between(data.index, data['Volume'], color='lightblue', alpha=0.3, label='Daily Volume Area')
    plt.plot(data.index, data['Volume'], label='Daily Volume', color='royalblue', alpha=0.7)
    plt.plot(data.index, data['Volume_Rolling_Avg'], label='30-day Rolling Avg of Volume', color='red', linewidth=3, linestyle='--')

    plt.title(f'{ticker} Daily Volume and 30-day Rolling Average', fontsize=16, fontweight='bold', fontfamily='serif', pad=20)
    
    plt.xlabel('Date', fontsize=14, fontweight='medium')
    plt.ylabel('Volume', fontsize=14, fontweight='medium')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    max_volume_date = data['Volume'].idxmax()
    max_volume = data['Volume'].max()
    plt.annotate(f'Max Volume\n{max_volume:,}', xy=(max_volume_date, max_volume), 
                xytext=(max_volume_date, max_volume * 0.8),  p
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=10, backgroundcolor='white')

    plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    plt.show()


# ADF Test
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, color='dodgerblue', label='Original Data', linewidth=2, alpha=0.7)
    plt.plot(rolmean, color='red', label='Rolling Mean (12 periods)', linestyle='--', linewidth=2)
    plt.plot(rolstd, color='black', label='Rolling Std (12 periods)', linestyle=':', linewidth=2)
    
    plt.title('Rolling Mean & Standard Deviation', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    print("\033[1m\033[94mResults of Dickey-Fuller Test:\033[0m")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in adft[4].items():
        output[f'Critical Value ({key})'] = value
    
    print("\033[96m", output, "\033[0m")
    
for ticker, data in grouped_data:
    print(f'\n\033[1m\033[92mADF Test for {ticker}:\033[0m')
    test_stationarity(data['Close'])

def check_missing_values(data):
    if data['Close'].isnull().sum() > 0:
        print(f"Original data contains NaN values.")
    if np.isinf(data['Close']).sum() > 0:
        print(f"Original data contains infinite values.")

for ticker, data in grouped_data:
    print(f"Checking missing values for {ticker}:")
    check_missing_values(data)  s


# Log Transformation and Differencing for Stationary Series 
for ticker, data in grouped_data:
    print(f'\n\033[1m\033[92mProcessing {ticker}...\033[0m')
    
    print("\033[1mApplying Log Transformation...\033[0m")
    data['Log_Close'] = np.log(data['Close'])
    
    data['Log_Close'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    data.dropna(subset=['Log_Close'], inplace=True)
    
    if data['Log_Close'].empty:
        print(f"\033[93mWarning: The series for {ticker} is empty after log transformation. Skipping.\033[0m")
        continue
    
    print("\033[1mApplying First Differencing...\033[0m")
    data['Log_Close_diff'] = data['Log_Close'].diff()
    
    data['Log_Close_diff'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    data.dropna(subset=['Log_Close_diff'], inplace=True)
    
    if data['Log_Close_diff'].empty:
        print(f"\033[93mWarning: The series for {ticker} is empty after differencing. Skipping ADF test.\033[0m")
        continue

    if data['Log_Close_diff'].isnull().sum() > 0 or np.isinf(data['Log_Close_diff']).sum() > 0:
        print(f"\033[91mError: The series for {ticker} still contains NaN or infinite values. Skipping ADF test.\033[0m")
        continue
    
    print(f'\n\033[1m\033[94mADF Test for {ticker} after Log Differencing:\033[0m')
    test_stationarity(data['Log_Close_diff'])


#Finding best values for ARIMA Model
fitted_models = {}
best_order = {}

for ticker, data in grouped_data:
    print(f"\nFinding the best ARIMA order for {ticker} and fitting the model...\n")
    
    try:
        model_auto = pm.auto_arima(data['Close'],
                                   start_p=1, max_p=10,
                                   start_q=1, max_q=10,
                                   seasonal=False,
                                   stepwise=True,
                                   trace=True, 
                                   error_action='ignore', 
                                   suppress_warnings=True) 

        best_order[ticker] = model_auto.order
        print(f"Best ARIMA order for {ticker}: {model_auto.order}\n")
        
        model = ARIMA(data['Close'], order=model_auto.order)
        fitted_model = model.fit()
        
        fitted_models[ticker] = fitted_model

        print(f"Model fitted for {ticker}.\n")
    
    except Exception as e:
        print(f"Error fitting model for {ticker}: {e}")

print("\nBest ARIMA Orders and Models for All Tickers:"'\n')
print(f"{'-'*25}")
print(f"{'Ticker':<10} | {'ARIMA Order':<15}")
print(f"{'-'*25}")

for ticker, order in best_order.items():
    print(f"{ticker:<10} | {str(order):<15}")

print(f"{'-'*25}")


#ARIMA Model Summary for each Ticker
for ticker, model in fitted_models.items():
    print(f"\033[1mARIMA Model Summary for {ticker}\033[0m")  
    print(f"{'='*78}")
    
    print(model.summary())
    
    print(f"{'-'*85}\n")


# Arima Model Fit
forecast_steps = 60
arima_orders = {
    'AAPL': (0, 1, 0),
    'AMD': (0, 1, 0),
    'AMZN': (0, 1, 0),
    'GOOGL': (1, 1, 1),
    'AMRK': (1, 1, 1),
    'APO': (0, 1, 0)
}

def fit_arima_model(train_data, order, forecast_steps):
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()

        forecast = fitted_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()

        return forecast_values, confidence_intervals

    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None, None

def plot_forecast(ticker, actual_data, forecast_values, forecast_steps):
    plt.figure(figsize=(12, 7))
    
    plt.plot(actual_data.index, actual_data, label='Actual Data', color='forestgreen', linewidth=2.5)
    
    forecast_index = pd.date_range(start=actual_data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    
    forecast_values_with_last = pd.concat([pd.Series([actual_data.iloc[-1]], index=[actual_data.index[-1]]), forecast_values])

    plt.plot(forecast_index.insert(0, actual_data.index[-1]), forecast_values_with_last, 
             label='Forecast', color='darkblue', linewidth=2.5, linestyle='--')

    max_date = actual_data.idxmax()
    max_value = actual_data.max()
    
    plt.title(f'{ticker} Forecasted Close Prices', fontsize=18, fontweight='bold', color='darkslategray', pad=20)
    plt.xlabel('Date', fontsize=14, fontweight='medium', color='gray')
    plt.ylabel('Close Price', fontsize=14, fontweight='medium', color='gray')

    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, fancybox=True, framealpha=1, borderpad=1)

    plt.grid(True, which='major', linestyle='--', linewidth=0.6, color='gray', alpha=0.7)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.5)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    
  
    plt.show()

for ticker, data in grouped_data:
    close_prices = data['Close'].dropna()

    # Train-Test Split (80% training, 20% testing)
    train_size = int(len(close_prices) * 0.8)
    train_data = close_prices[:train_size]

    best_order = arima_orders.get(ticker)

    if best_order is None:
        print(f"Skipping {ticker} as no ARIMA order is available.")
        continue  

    forecast_values, confidence_intervals = fit_arima_model(train_data, best_order, forecast_steps)

    if forecast_values is not None:
        plot_forecast(ticker, close_prices, forecast_values, forecast_steps)

# Calculating Model accuracy
def calculate_errors_and_ci(actual_values, forecast_values, confidence_intervals):
    try:
        # Calculate MAE
        mae = mean_absolute_error(actual_values, forecast_values)
        
        # Calculate MSE
        mse = mean_squared_error(actual_values, forecast_values)
        
        # Calculate RMSE for better insight
        rmse = np.sqrt(mse)
        
        # Calculate Confidence Intervals 
        lower_ci = confidence_intervals.iloc[:, 0]
        upper_ci = confidence_intervals.iloc[:, 1]
        
        # Confidence interval range at 95% confidence level
        confidence_interval_range = upper_ci - lower_ci

        return mae, mse, rmse, confidence_interval_range
    except Exception as e:
        print(f"Error calculating errors or confidence intervals: {e}")
        return None, None, None, None

# Function to fit ARIMA model and generating forecast with calculation of errors
def fit_arima_model_and_evaluate(train_data, test_data, order, forecast_steps):
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()

        forecast = fitted_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()

        test_data = test_data[:len(forecast_values)]
        
        # Calculate MAE, MSE, and Confidence Interval
        mae, mse, rmse, ci_range = calculate_errors_and_ci(test_data, forecast_values, confidence_intervals)
        
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        print(f"Average Confidence Interval Range: {ci_range.mean():.4f}")
        
        return forecast_values, confidence_intervals
    except Exception as e:
        print(f"Error fitting ARIMA model or evaluating performance: {e}")
        return None, None

def run_evaluation(grouped_data, arima_orders, forecast_steps):
    for ticker, data in grouped_data:
        close_prices = data['Close'].dropna()

        # Train-Test Split (80% training, 20% testing)
        train_size = int(len(close_prices) * 0.8)
        train_data = close_prices[:train_size]
        test_data = close_prices[train_size:train_size + forecast_steps]

        best_order = arima_orders.get(ticker)

        if best_order is None:
            print(f"Skipping {ticker} as no ARIMA order is available.")
            continue  

        print(f"\nEvaluating for {ticker}...\n")
        forecast_values, confidence_intervals = fit_arima_model_and_evaluate(train_data, test_data, best_order, forecast_steps)

        if forecast_values is not None and confidence_intervals is not None:
            print(f"Evaluation for {ticker} complete.\n")

arima_orders = {
    'AAPL': (0, 1, 0),
    'AMD': (0, 1, 0),
    'AMZN': (0, 1, 0),
    'GOOGL': (1, 1, 1),
    'AMRK': (1, 1, 1),
    'APO': (0, 1, 0)
}
forecast_steps = 30

run_evaluation(grouped_data, arima_orders, forecast_steps)






