import pandas as pd
import numpy as np
import get_ticker
from datetime import date, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, skew
import math
from pylab import rcParams
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

# set plot size
rcParams['figure.figsize'] = 10, 5

def combine_data():
     # call get_ticker to get bitcoin price data and crypto greed-fear index
     bitcoin_data = get_ticker.get_bitcoin_prices()
     index = get_ticker.get_greed_index()

     # process the bitcoin data
     index['timestamp'] = pd.to_datetime(index['timestamp'], unit='s')
     #change the crypto greed-fear index to an integer
     index['value'] = index['value'].astype(int)
     bitcoin_data.index = pd.to_datetime(bitcoin_data.index)
     btc_selected = pd.DataFrame({'timestamp': bitcoin_data.index, 'Close': bitcoin_data['Close']})

     # merging dataframes
     merged_df = pd.merge(btc_selected, index, on=['timestamp'], how='inner')

     # prepare the merged dataset
     merged_df.set_index('timestamp', inplace=True)

     return merged_df
def data_explore(merged_df):
    merged_df.info()
    merged_df.describe()
    print(merged_df)
    print(merged_df.shape)

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot Bitcoin price in the first subplot
    ax1.plot(merged_df.index, merged_df['Close'], color='orange', label='Bitcoin Price')
    ax1.set_ylabel('Bitcoin Price')
    ax1.legend()

    # Plot index value in the second subplot
    ax2.plot(merged_df.index, merged_df['value'], color='green', label='Index Value')
    ax2.set_ylabel('Index Value')
    ax2.legend()

    # Set common x-axis label
    ax2.set_xlabel('Date')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # look at some features of btc-USD
    btc_skewness = merged_df['Close'].skew()
    print("Skewness of BTC-USD: ", btc_skewness)
    btc_std = merged_df['Close'].std()
    print("Standard deviation of BTC-USD: ", btc_std, " USD")

    btc_rolling_var = merged_df['Close'].rolling(30).var()
    # plot the variance chart over time
    plt.figure(figsize=(12,6))
    plt.plot(merged_df.index, btc_rolling_var, color='gray')
    plt.title('30 day rolling variance of bitcoin prices')
    plt.xlabel('Date')
    plt.ylabel('variance')
    plt.grid(True)
    plt.show()

    #scaler = MinMaxScaler()
    #variance_table = pd.DataFrame({'Date': merged_df.index, 'Rolling variance': btc_rolling_var})
    #variance_table['Scaled Variance'] = scaler.fit_transform(variance_table[['Rolling variance']])

def data_preprocess(merged_df):
    # tranforming the prices using log to reduce skewness
    merged_df['log_Close'] = np.log(merged_df['Close'])

    # try to remove trend using 1st order differencing
    #calculate first-order diff of log prices
    merged_df['Log Close Diff'] = merged_df['log_Close'].diff()

    # Plot ACF for logged prices
    plt.figure(figsize=(12, 6))
    plot_acf(merged_df['log_Close'].dropna(), lags=40, alpha=0.05, ax=plt.gca())
    plt.title('Autocorrelation Function (ACF) for Logged Prices')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid(True)
    plt.show()

    # Plot PACF for logged prices
    plt.figure(figsize=(12, 6))
    plot_pacf(merged_df['log_Close'].dropna(), lags=40, alpha=0.05, ax=plt.gca())
    plt.title('Partial Autocorrelation Function (PACF) for Logged Prices')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.show()

    #plot the differenced log prices
    plt.plot(merged_df.index, merged_df['Log Close Diff'])
    plt.title('First order diff. log prices')
    plt.xlabel('Date')
    plt.ylabel('differenced log prices')
    plt.grid(True)
    plt.show()

    #plotting ACF and PACF for logged prices to check autocorrelation
    plt.figure(figsize=(12, 6))
    plot_acf(merged_df['Log Close Diff'].dropna(), lags=40, alpha=0.05, ax=plt.gca())
    plt.title('Autocorrelation Function (ACF) for First Differences of Logged Prices')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid(True)
    plt.show()

    # Plot PACF for first differences of logged prices
    plt.figure(figsize=(12, 6))
    plot_pacf(merged_df['Log Close Diff'].dropna(), lags=40, alpha=0.05, ax=plt.gca())
    plt.title('Partial Autocorrelation Function (PACF) for First Differences of Logged Prices')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.show()

    return merged_df

def arimatimeseries_prep(merged_df):
    #Fit Auto-ARIMA model to find p and q
    model = auto_arima(merged_df['log_Close'].dropna(), seasonal=False, trace=True)
    #view model summary
    print(model.summary())
    #extract best model params
    print("Best (p,d,q) values: ", model.order)
    best_order = model.order

    # train test split(20-80)
    n = math.ceil(len(merged_df) * 0.8) # calculate the no. of rows for cutoff and round up the number to avoid reducing the n when there is remainder
    train_data = merged_df.iloc[:n]
    test_data = merged_df.iloc[n:]

    # getting the first forecasted date for test data
    traingdate_start = date(2018, 2, 1) + timedelta(days=n)
    print('forecast start date: ', traingdate_start)

    #get the standard deviation of the btc daily price in test data period
    test_std_deviation = test_data['Close'].std().round(2)
    print("Standard Deviation during the test period of BTC-USD: ", test_std_deviation, "USD")

    return train_data, test_data, n, best_order

def timeseries_arima(train_data, test_data, n, best_order):
    #Fit ARIMA model into a initial training data
    model = ARIMA(train_data.iloc[:n]['log_Close'], order=best_order)
    fitted_model = model.fit()

    #perform rolling forecast for all test-data available
    rolling_forecast=[]
    rolling_forecast_USD = []
    mse_list =[]
    for i in range(len(test_data)):
        try:
            # fit the model with udpated training data
            fitted_model = ARIMA(
                pd.concat([train_data.iloc[i - n:]['log_Close'], test_data.iloc[:i]['log_Close']]),
                order=best_order).fit()

            # one step ahead forecast
            forecast= fitted_model.forecast(steps=1)
            rolling_forecast.append(forecast.values[0])

            #convert the logged forecast price back to USD
            forecast_USD = np.exp(forecast)
            rolling_forecast_USD.append(forecast_USD.values[0])

            #evaluate model performance
            true_values = test_data['Close'].iloc[i]
            mse = mean_squared_error([true_values], [forecast_USD])
            mse_list.append(mse)

        #error handling
        except Exception as e:
            print("Error occured while updating the model for test data at index", i)
            print(e)
            break

    # create a new dataframe using the results
    result_df = pd.DataFrame({
        'Timestamp': test_data.index,
        'Real_BTC_Price': test_data['Close'],
        'Rolling_Forecast_USD': rolling_forecast_USD,
        'MSE': mse_list
    })
    return result_df

def greed_fear(merged_df):
    #check the histogram to see if it is normal distribution
    bins = [0, 25, 50, 75, 100]
    labels = ['Extreme fear', 'Fear', 'Greed', 'Extreme greed']
    plt.hist(merged_df['value'], bins=bins, alpha=0.7, color='skyblue')
    plt.title('Histogram of Fear/Greed Index')
    plt.legend(labels)
    plt.xlabel('Index Value')
    plt.ylabel('Frequency')
    plt.show()

    #check the correlation between the index and log price
    corr, p_value = spearmanr(merged_df['value'], merged_df['log_Close'])
    print(f"Spearman correlation : {corr}")
    print(f"Spearman p-value : {p_value}")

def timeseries_sarimax_prep(merged_df):
    #Fit Auto-ARIMA model to find p, d and q
    greed_fear = merged_df['value']
    model = auto_arima(merged_df['log_Close'].dropna(), exog=merged_df['value'], seasonal=False, trace=True)
    #view model summary
    print(model.summary())

    #extract best model params
    print("Best (p,d,q) values: ", model.order)
    best_order = model.order

    #prepare data for ARIMA time series
    #merged_df.set_index('timestamp', inplace=True)

    # shift the crypto fear index backward one day to predict btc-price and aligh the indices for SARIMAX
    merged_df['value'] = merged_df['value'].shift(1)
    merged_df = merged_df.dropna() # drop the row with NN after shifting

    print(merged_df.shape)

    # train test split(20-80)
    n = math.ceil(len(merged_df) * 0.8) # calculate the no. of rows for cutoff
    train_data1 = merged_df.iloc[:n]
    test_data1 = merged_df.iloc[n:]

    train_endog = train_data1['log_Close']
    train_exog = train_data1['value']
    test_endog = test_data1['log_Close']
    test_exog = test_data1['value']
    return train_data1, test_data1, train_endog, train_exog, test_endog, test_exog, n, best_order

def timeseries_sarimax(test_data1, train_endog, train_exog, test_endog, test_exog, n, best_order):
    # Fit SARIMAX model with crypto greed-fear index as exogenous variable
    new_model = ARIMA(endog=train_endog, exog=train_exog, order=best_order)
    newfitted_model = new_model.fit()

    #perform rolling forecast
    rolling_forecast1=[]
    rolling_forecast_USD1 = []
    mse_list1 =[]
    for i in range(len(test_data1)):
        try:
            # fit the model with udpated training data
            newfitted_model = ARIMA(
                endog=pd.concat([train_endog.iloc[i - n:], test_endog.iloc[:i]]),
                exog=pd.concat([train_exog.iloc[i - n:], test_exog.iloc[:i]]),
                order=best_order).fit()

            # one step ahead forecast
            exog_value = test_exog.iloc[i]
            forecast1 = newfitted_model.forecast(steps=1, exog=np.array([[exog_value]]))

            #save the forecast to the list
            rolling_forecast1.append(forecast1.values[0])

            #convert the logged forecast price back to USD
            forecast_USD1 = np.exp(forecast1)
            rolling_forecast_USD1.append(forecast_USD1.values[0])

            #evaluate model performance
            true_values = test_data1['Close'].iloc[i]
            mse1 = mean_squared_error([true_values], [forecast_USD1])
            mse_list1.append(mse1)
        #error handling
        except Exception as e:
            print("Error occured while updating the model for test data at index", i)
            print(e)
            break
    # create a new dataframe from results
    result_df1 = pd.DataFrame({
        'Timestamp': test_data1.index,
        'Real_BTC_Price': test_data1['Close'],
        'Rolling_Forecast_USD': rolling_forecast_USD1,
        'MSE': mse_list1
    })

    return result_df1

def result_analysis(*result_dfs):

    for i, result_df in enumerate(result_dfs):
        #calculate average Mean Square Error, Root mean square error and average Mean percentage Error
        average_mse = np.mean(result_df['MSE'])
        average_rmse = round(np.sqrt(average_mse), 4)
        print(f"Average RMSE {i+1}:, {average_rmse}")

        #calculate the R2 score
        r2score = r2_score(result_df['Real_BTC_Price'], result_df['Rolling_Forecast_USD'])
        print(f"R2score {i+1}: {r2score}")

        # calculate the residual of each forecast and actual price
        residuals = result_df['Real_BTC_Price'] - result_df['Rolling_Forecast_USD']
        percentage_errors = residuals / result_df['Real_BTC_Price'] * 100
        absolute_percentage_errors = np.abs(percentage_errors)
        mpe = np.mean(absolute_percentage_errors)
        print(f"Mean percentage error {i+1}: {mpe}")
        max_error = np.max(absolute_percentage_errors)
        print(f"Max percentage error {i+1}: {max_error}")

        # calculate the daily returns in %
        daily_returns = result_df['Real_BTC_Price'].pct_change().dropna() * 100
        daily_returns.to_csv('daily_returns.csv', header=['Daily_Return'])
        # calculate the average volatility
        average_volatility = daily_returns.std()
        print('average volatility for the test period: ', average_volatility)


        # plot a graph
        plt.figure(figsize=(10,6))
        plt.plot(daily_returns, color = 'red', label='Daily Close')
        plt.plot(percentage_errors, color='blue', label='%error')
        plt.title('Daily returns vs. percentage errors')
        plt.xlabel('Time')
        plt.ylabel('Percentages')
        plt.grid(True)
        plt.legend()
        plt.show()

        #plot histogram for the absolute_percentage_errors
        plt.hist(absolute_percentage_errors, bins=20, edgecolor='black')
        plt.title('Distribution of Absolute Percentage Errors')
        plt.xlabel('Absolute percentage error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # plot residuals
        plt.plot(residuals, color='orange')
        plt.xlabel('Time')
        plt.ylabel('USD')
        plt.title('Residuals between forecasted and real BTC price')
        plt.show()

        # plot a line graph to see forecasted vs. real btc prices
        plt.plot(result_df['Real_BTC_Price'], label='Actual Bitcoin prices', color='blue', linewidth=2)
        plt.plot(result_df['Rolling_Forecast_USD'], label='Forecasted Bitcoin prices', color='red', linewidth=1)
        plt.ylim([result_df[['Real_BTC_Price', 'Rolling_Forecast_USD']].min().min() - 5000,
                  result_df[['Real_BTC_Price', 'Rolling_Forecast_USD']].max().max() + 5000])
        plt.xlabel('Time')
        plt.ylabel('Bitcoin Prices')
        plt.title('Actual vs. Forecasted Bitcoin prices')
        plt.legend()
        plt.show()

# Main function
def main():
    merged_df = combine_data()
    data_explore(merged_df)
    #update the df after data preprocessing
    merged_df = data_preprocess(merged_df)
    #defining training and testing data
    train_data, test_data, n, best_order = arimatimeseries_prep(merged_df)
    # get ARIMA results
    result_df = timeseries_arima(train_data, test_data, n, best_order)
    greed_fear(merged_df)
    train_data1, test_data1, train_endog, train_exog, test_endog, test_exog, n, best_order =timeseries_sarimax_prep(merged_df)
    # get SARIMAX results including exogenous variable of greed-fear index
    result_df1 = timeseries_sarimax(test_data1, train_endog, train_exog, test_endog, test_exog, n, best_order)
    # analyse the forecast results
    result_analysis(result_df, result_df1)

if __name__ == "__main__":
    main()