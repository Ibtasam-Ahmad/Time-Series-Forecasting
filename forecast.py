# # forecast.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
# import pmdarima as pm
# from prophet import Prophet
# import tensorflow as tf

# def load_data(path='complete_datetime.csv'):
#     df = pd.read_csv(path)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.sort_values('Date')
#     return df

# def run_rnn(data):
#     y = data['Open'].fillna(method='ffill')
#     y = y.values.reshape(-1, 1)
#     # print(df)
#     # print(y)

#     # scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler = scaler.fit(y)
#     y = scaler.transform(y)

#     # generate the input and output sequences
#     n_lookback = 60  # length of input sequences (lookback period)
#     n_forecast = 30  # length of output sequences (forecast period)

#     X = []
#     Y = []

#     for i in range(n_lookback, len(y) - n_forecast + 1):
#         X.append(y[i - n_lookback: i])
#         Y.append(y[i: i + n_forecast])

#         X = np.array(X)
#         Y = np.array(Y)
#         # print(X, Y)

#         # fit the model
#         model = Sequential()
#         model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
#         model.add(SimpleRNN(units=50))
#         model.add(Dense(n_forecast))

#         model.compile(loss='mean_squared_error', optimizer='adam')
#         model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
#         model.summary()

#         # generate the forecasts
#         X_ = y[- n_lookback:]  # last available input sequence
#         X_ = X_.reshape(1, n_lookback, 1)

#         Y_ = model.predict(X_).reshape(-1, 1)
#         Y_ = scaler.inverse_transform(Y_)
#         # print(X_, Y_)

#         # organize the results in a data frame
#         df_past = data[['Open']].reset_index()
#         df_past.rename(columns={'index': 'Date', 'Open': 'Actual'}, inplace=True)
#         df_past['Date'] = pd.to_datetime(df_past['Date'])
#         df_past['Forecast'] = np.nan
#         df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
#         print(df_past)

#         df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
#         df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
#         df_future['Forecast'] = Y_.flatten()
#         df_future['Actual'] = np.nan
#         print(df_future)

#         results = pd.concat([df_past, df_future]).set_index('Date')

#         # plot the results
#         # results.plot(title='Forcast')
#         # Assuming you have two DataFrames named 'df1' and 'df2'
#         df_combined = pd.concat([df_past[['Actual', 'Forecast']], df_future[['Actual', 'Forecast']]], ignore_index=True)

#         plt.figure(figsize=(14, 4))
#         plt.plot(df_combined)
#         plt.show()

# def run_lstm(data):
#     y = data['Open'].fillna(method='ffill')
#     y = y.values.reshape(-1, 1)
#     # print(df)
#     # print(y)

#     # scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler = scaler.fit(y)
#     y = scaler.transform(y)

#     # generate the input and output sequences
#     n_lookback = 60  # length of input sequences (lookback period)
#     n_forecast = 30  # length of output sequences (forecast period)

#     X = []
#     Y = []

#     for i in range(n_lookback, len(y) - n_forecast + 1):
#         X.append(y[i - n_lookback: i])
#         Y.append(y[i: i + n_forecast])

#     X = np.array(X)
#     Y = np.array(Y)
#     # print(X, Y)

#     # fit the model
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
#     model.add(LSTM(units=50))
#     model.add(Dense(n_forecast))

#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
#     model.summary()

#     # generate the forecasts
#     X_ = y[- n_lookback:]  # last available input sequence
#     X_ = X_.reshape(1, n_lookback, 1)

#     Y_ = model.predict(X_).reshape(-1, 1)
#     Y_ = scaler.inverse_transform(Y_)
#     # print(X_, Y_)

#     # organize the results in a data frame
#     df_past = data[['Open']].reset_index()
#     df_past.rename(columns={'index': 'Date', 'Open': 'Actual'}, inplace=True)
#     df_past['Date'] = pd.to_datetime(df_past['Date'])
#     df_past['Forecast'] = np.nan
#     df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
#     print(df_past)

#     df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
#     df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
#     df_future['Forecast'] = Y_.flatten()
#     df_future['Actual'] = np.nan
#     print(df_future)

#     results = pd.concat([df_past, df_future]).set_index('Date')

#     # plot the results
#     # results.plot(title='Forcast')
#     # Assuming you have two DataFrames named 'df1' and 'df2'
#     df_combined = pd.concat([df_past[['Actual', 'Forecast']], df_future[['Actual', 'Forecast']]], ignore_index=True)

#     plt.figure(figsize=(14, 4))
#     plt.plot(df_combined)
#     plt.show()

# def run_arima(data):
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pylab as plt
#     %matplotlib inline
#     from matplotlib.pylab import rcParams

#     import warnings
#     warnings.filterwarnings('ignore')

#     from statsmodels.tsa.stattools import adfuller
#     # !pip install pmdarima --quiet
#     import pmdarima as pm
    
#     df = pd.read_csv('complete_datetime.csv')

#     dftest = adfuller(df['Open'], autolag='AIC')

#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
        
#     print(dfoutput)

#     model = pm.auto_arima(
#         y=df.Open,
#         start_p=1,   # AR search setting
#         max_p=3,
#         start_q=1,   # MA search setting
#         max_q=3,
#         test='adf',  # use Augmented Dickey-Fuller test to get best difference 'd'
#         d=None,      # let model detemine best 'd'
#         seasonal=False, # SARIMA setting. Set it to false & P D to 0, since we interested normal ARIMA
#         start_P=0,
#         D=0,
#         trace=True,
#         error_action='ignore',
#         suppress_warnings=True,
#         stepwise=True   
#     )

#     # model = pm.auto_arima(df[['Open']], 
#     #                        start_p=1, start_q=1,
#     #                        max_p=3, max_q=3, 
#     #                        d=None,  # Let auto_arima determine differencing
#     #                        seasonal=False,  # No seasonality
#     #                        trace=False,
#     #                        error_action='ignore',  
#     #                        suppress_warnings=True, 
#     #                        stepwise=True)


#     print(model.summary())

#     # model.plot_diagnostics(figsize=(20,7))
#     # plt.show()

#     # Forecasting data
#     n_periods = 30
#     # model.plot_diagnostics(figsize=(20,7))
#     # plt.show()

#     fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
#     index_of_fc = np.arange(len(df.Open), len(df.Open)+n_periods)

#     # Plot
#     fc_series = pd.Series(fc, index=index_of_fc)   # forecasted data in time series
#     lower_series = pd.Series(confint[:,0], index=index_of_fc)   # lower CI band
#     upper_series = pd.Series(confint[:,1], index=index_of_fc)   # upper CI band

#     # Plot
#     plt.figure(figsize=(14, 4))
#     plt.plot(df.Open)
#     plt.plot(fc_series, color='darkgreen')
#     plt.fill_between(lower_series.index, lower_series, upper_series, color='y', alpha=.15)

#     plt.title(f"Forecast {n_periods} data points forward")

# def run_sarima(data):
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pylab as plt
#     %matplotlib inline
#     from matplotlib.pylab import rcParams

#     import warnings
#     warnings.filterwarnings('ignore')

#     from statsmodels.tsa.stattools import adfuller
#     # !pip install pmdarima --quiet
#     import pmdarima as pm


#     df = pd.read_csv('complete_datetime.csv')

#     dftest = adfuller(df['Open'], autolag='AIC')

#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
        
#     print(dfoutput)

#     model = pm.auto_arima(
#         y=df.Open,
#         start_p=1,   # AR search setting
#         max_p=3,
#         start_q=1,   # MA search setting
#         max_q=3,
#         test='adf',  # use Augmented Dickey-Fuller test to get best difference 'd'
#         d=None,      # let model detemine best 'd'
#         seasonal=True, # SARIMA setting. Set it to True & P D to 0, since we interested normal ARIMA
#         start_P=0,
#         D=0,
#         trace=True,
#         error_action='ignore',
#         suppress_warnings=True,
#         stepwise=True   
#     )

#     # model = pm.auto_arima(df[['Open']], 
#     #                        start_p=1, start_q=1,
#     #                        max_p=3, max_q=3, 
#     #                        d=None,  # Let auto_arima determine differencing
#     #                        seasonal=True,  # Enable seasonality
#     #                        m=12,  # Seasonal period (e.g., 12 for monthly data)
#     #                        start_P=0,  # Seasonal AR order
#     #                        D=1,  # Seasonal differencing
#     #                        trace=False,
#     #                        error_action='ignore',  
#     #                        suppress_warnings=True, 
#     #                        stepwise=True)



#     print(model.summary())

#     # model.plot_diagnostics(figsize=(20,7))
#     # plt.show()

#     # Forecasting data
#     n_periods = 30
#     # model.plot_diagnostics(figsize=(20,7))
#     # plt.show()

#     fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
#     index_of_fc = np.arange(len(df.Open), len(df.Open)+n_periods)

#     # Plot
#     fc_series = pd.Series(fc, index=index_of_fc)   # forecasted data in time series
#     lower_series = pd.Series(confint[:,0], index=index_of_fc)   # lower CI band
#     upper_series = pd.Series(confint[:,1], index=index_of_fc)   # upper CI band

#     # Plot
#     plt.figure(figsize=(14, 4))
#     plt.plot(df.Open)
#     plt.plot(fc_series, color='darkgreen')
#     plt.fill_between(lower_series.index, lower_series, upper_series, color='y', alpha=.15)

#     plt.title(f"Forecast {n_periods} data points forward")

# def run_sarimax(data):
#     from datetime import datetime
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pylab as plt
#     %matplotlib inline
#     from matplotlib.pylab import rcParams

#     import warnings
#     warnings.filterwarnings('ignore')

#     from statsmodels.tsa.stattools import adfuller
#     # !pip install pmdarima --quiet
#     import pmdarima as pm


#     df = pd.read_csv('complete_datetime.csv')

#     dftest = adfuller(df['Open'], autolag='AIC')

#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
        
#     print(dfoutput)

#     # Convert dates to ordinal numbers
#     df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

#     model = pm.auto_arima(df[['Open']], 
#                             exogenous=df[['Date_Ordinal']],
#                             start_p=1, start_q=1,
#                             test='adf',
#                             max_p=3, max_q=3, m=12,
#                             start_P=0, seasonal=True,
#                             d=None, D=1, 
#                             trace=False,
#                             error_action='ignore',  
#                             suppress_warnings=True, 
#                             stepwise=True)


#     # model = pm.auto_arima(df[['Open']], 
#     #                        exogenous=df[['Date']],  # External variables
#     #                        start_p=1, start_q=1,
#     #                        max_p=3, max_q=3, 
#     #                        d=None,
#     #                        seasonal=True,  
#     #                        m=12,  
#     #                        start_P=0,  
#     #                        D=1,  
#     #                        trace=False,
#     #                        error_action='ignore',  
#     #                        suppress_warnings=True, 
#     #                        stepwise=True)


#     print(model.summary())

#     # model.plot_diagnostics(figsize=(20,7))
#     # plt.show()

#     # Forecasting data
#     n_periods = 30
#     # Generate future dates
#     future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=n_periods + 1, freq='D')[1:]

#     # Convert future dates to ordinal numbers
#     future_exog = pd.DataFrame({'Date_Ordinal': future_dates.map(pd.Timestamp.toordinal)})

#     # Make predictions with exogenous variables
#     fc, confint = model.predict(n_periods=n_periods, X=future_exog, return_conf_int=True)

#     index_of_fc = np.arange(len(df.Open), len(df.Open)+n_periods)

#     # Plot
#     fc_series = pd.Series(fc, index=index_of_fc)   # forecasted data in time series
#     lower_series = pd.Series(confint[:,0], index=index_of_fc)   # lower CI band
#     upper_series = pd.Series(confint[:,1], index=index_of_fc)   # upper CI band

#     # Plot
#     plt.figure(figsize=(14, 4))
#     plt.plot(df.Open)
#     plt.plot(fc_series, color='darkgreen')
#     plt.fill_between(lower_series.index, lower_series, upper_series, color='y', alpha=.15)

#     plt.title(f"Forecast {n_periods} data points forward")


# def run_prophet(data):
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from prophet import Prophet

#     # Load data
#     df_data = pd.read_csv("complete_datetime.csv")

#     # Sort data by date to ensure chronological order
#     df_data = df_data.sort_values(by="Date")

#     # Convert Date column to datetime format
#     df_data["Date"] = pd.to_datetime(df_data["Date"])

#     # Handle missing values
#     df_data.fillna(method='ffill', inplace=True)

#     # Calculate 70% split index dynamically
#     split_index = int(len(df_data) * 0.7)

#     # Split Train and Test Data
#     df_data_train = df_data.iloc[:split_index]  # First 70% for training
#     df_data_test = df_data.iloc[split_index:]   # Remaining 30% for testing

#     # Prepare Data for Prophet
#     df_train_prophet = df_data_train.rename(columns={"Date": "ds", "Open": "y"})

#     # Train Prophet Model
#     # model_prophet = Prophet()
#     model_prophet = Prophet(interval_width=0.80)  # Reducing the uncertainty range
#     model_prophet.fit(df_train_prophet)

#     # Get last available date and last known value
#     last_date = df_data["Date"].max()
#     last_value = df_data["Open"].iloc[-1]

#     # ✅ Create future dataframe starting from the last known date
#     df_future = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')})

#     # Make Predictions
#     forecast_prophet = model_prophet.predict(df_future)

#     # Extract forecast results
#     df_forecast = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#     df_forecast = df_forecast.rename(columns={'ds': 'Date'})  # Keep Date column name consistent
#     df_forecast['yhat_lower'] = np.maximum(df_forecast['yhat_lower'], df_forecast['yhat'].min() * 0.9)

#     # ✅ Adjust forecast to continue smoothly from last known value
#     # df_forecast['yhat'] += (last_value - df_forecast['yhat'].iloc[0])
#     adjustment = last_value - df_forecast['yhat'].iloc[0]
#     df_forecast[['yhat', 'yhat_lower', 'yhat_upper']] += adjustment

#     # Plot the results
#     plt.figure(figsize=(14, 4))

#     # Plot all previous actual values
#     plt.plot(df_data['Date'], df_data['Open'], 'b-', marker='o', markersize=2, label='Actual Data')

#     # Plot only the forecasted values (red dotted line)
#     plt.plot(df_forecast['Date'], df_forecast['yhat'], 'r--', marker='.', label='n-Day Forecast')

#     # Add confidence intervals
#     plt.fill_between(df_forecast['Date'], df_forecast['yhat_lower'], df_forecast['yhat_upper'], color='red', alpha=0.2)

#     # Add a vertical line at the last known data point
#     plt.axvline(x=last_date, color='black', linestyle='--', label='Last Known Data')

#     # Show the legend
#     plt.legend()

#     # Display the plot
#     plt.show()




# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import SimpleRNN, Dense
# # import tensorflow as tf

# # st.set_page_config(page_title="Stock Forecasting RNN", layout="wide")
# # st.title("📈 Stock Price Forecasting")

# # # Load local CSV directly
# # # (Replace the path below with your actual file path if needed)
# # csv_path = 'C:/Users/LENOVO/Downloads/Additional RND/Forcasting_Models/complete_datetime.csv'
# # data = pd.read_csv(csv_path)

# # # -----------------------------
# # # If you want to use file upload instead, uncomment below:
# # # uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
# # # if uploaded_file is not None:
# # #     data = pd.read_csv(uploaded_file)
# # # else:
# # #     st.stop()
# # # -----------------------------

# # if 'Open' not in data.columns:
# #     st.error("The CSV must contain an 'Open' column.")
# #     st.stop()

# # data['Open'] = data['Open'].fillna(method='ffill')
# # y = data['Open'].values.reshape(-1, 1)

# # # Scale the data
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # y_scaled = scaler.fit_transform(y)

# # # Parameters
# # n_lookback = st.slider("Lookback period (days)", 10, 100, 60)
# # n_forecast = st.slider("Forecast horizon (days)", 1, 60, 30)

# # # Generate sequences
# # X, Y = [], []
# # for i in range(n_lookback, len(y_scaled) - n_forecast + 1):
# #     X.append(y_scaled[i - n_lookback:i])
# #     Y.append(y_scaled[i:i + n_forecast])

# # X = np.array(X)
# # Y = np.array(Y)

# # # Build and train RNN
# # model = Sequential()
# # model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
# # model.add(SimpleRNN(units=50))
# # model.add(Dense(n_forecast))
# # model.compile(loss='mean_squared_error', optimizer='adam')

# # with st.spinner("Training the model..."):
# #     model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

# # # Forecast
# # X_ = y_scaled[-n_lookback:].reshape(1, n_lookback, 1)
# # Y_ = model.predict(X_).reshape(-1, 1)
# # Y_ = scaler.inverse_transform(Y_)

# # # Prepare past and future dataframes
# # df_past = data[['Open']].copy()
# # df_past.rename(columns={'Open': 'Actual'}, inplace=True)
# # df_past['Forecast'] = np.nan
# # df_past.loc[df_past.index[-1], 'Forecast'] = df_past.loc[df_past.index[-1], 'Actual']

# # # Handle date indexing
# # if 'Date' in data.columns:
# #     df_past['Date'] = pd.to_datetime(data['Date'])
# # else:
# #     df_past['Date'] = pd.date_range(start='2020-01-01', periods=len(df_past))

# # last_date = df_past['Date'].iloc[-1]
# # future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

# # df_future = pd.DataFrame({
# #     'Date': future_dates,
# #     'Actual': np.nan,
# #     'Forecast': Y_.flatten()
# # })

# # df_combined = pd.concat([df_past[['Date', 'Actual', 'Forecast']], df_future], ignore_index=True)
# # df_combined.set_index('Date', inplace=True)

# # # Plot results
# # st.subheader("📊 Forecast Results")
# # fig, ax = plt.subplots(figsize=(14, 5))
# # df_combined['Actual'].plot(ax=ax, label='Actual')
# # df_combined['Forecast'].plot(ax=ax, label='Forecast')
# # ax.set_title('Stock Price Forecasting')
# # ax.set_ylabel('Price')
# # ax.legend()
# # st.pyplot(fig)

# # # Optionally show the data
# # if st.checkbox("Show forecast data table"):
# #     st.dataframe(df_combined.tail(n_forecast + 10))




# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
# import tensorflow as tf

# st.set_page_config(page_title="Stock Forecasting", layout="wide")
# st.title("📈 Stock Price Forecasting with RNN / LSTM")

# # Fix seed for reproducibility
# tf.random.set_seed(0)

# # Load data (replace with your actual file path if needed)
# csv_path = 'C:/Users/LENOVO/Downloads/Additional RND/Forcasting_Models/complete_datetime.csv'
# data = pd.read_csv(csv_path)

# # Validate 'Open' column
# if 'Open' not in data.columns:
#     st.error("CSV file must contain an 'Open' column.")
#     st.stop()

# # Preprocess
# data['Open'] = data['Open'].fillna(method='ffill')
# y = data['Open'].values.reshape(-1, 1)

# # Scale
# scaler = MinMaxScaler(feature_range=(0, 1))
# y_scaled = scaler.fit_transform(y)

# # Sidebar options
# model_type = st.selectbox("Select model type:", ['Simple RNN', 'LSTM'])
# n_lookback = st.slider("Lookback period (days)", 10, 100, 60)
# n_forecast = st.slider("Forecast horizon (days)", 1, 60, 30)

# # Create sequences
# X, Y = [], []
# for i in range(n_lookback, len(y_scaled) - n_forecast + 1):
#     X.append(y_scaled[i - n_lookback:i])
#     Y.append(y_scaled[i:i + n_forecast])
# X = np.array(X)
# Y = np.array(Y)

# # Build model
# model = Sequential()
# if model_type == 'Simple RNN':
#     model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
#     model.add(SimpleRNN(units=50))
# else:
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
#     model.add(LSTM(units=50))

# model.add(Dense(n_forecast))
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Train model
# with st.spinner(f"Training {model_type} model..."):
#     model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

# # Forecast
# X_ = y_scaled[-n_lookback:].reshape(1, n_lookback, 1)
# Y_ = model.predict(X_).reshape(-1, 1)
# Y_ = scaler.inverse_transform(Y_)

# # Prepare past and future DataFrames
# df_past = data[['Open']].copy()
# df_past.rename(columns={'Open': 'Actual'}, inplace=True)
# df_past['Forecast'] = np.nan
# df_past.loc[df_past.index[-1], 'Forecast'] = df_past.loc[df_past.index[-1], 'Actual']

# # Use provided date column if available
# if 'Date' in data.columns:
#     df_past['Date'] = pd.to_datetime(data['Date'])
# else:
#     df_past['Date'] = pd.date_range(start='2020-01-01', periods=len(df_past))

# last_date = df_past['Date'].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

# df_future = pd.DataFrame({
#     'Date': future_dates,
#     'Actual': np.nan,
#     'Forecast': Y_.flatten()
# })

# # Combine for plotting
# df_combined = pd.concat([df_past[['Date', 'Actual', 'Forecast']], df_future], ignore_index=True)
# df_combined.set_index('Date', inplace=True)

# # Plot
# st.subheader("📊 Forecast Results")
# fig, ax = plt.subplots(figsize=(14, 5))
# df_combined['Actual'].plot(ax=ax, label='Actual')
# df_combined['Forecast'].plot(ax=ax, label='Forecast')
# ax.set_title(f"{model_type} Forecast")
# ax.set_ylabel("Price")
# ax.legend()
# st.pyplot(fig)

# # Show data table
# if st.checkbox("Show forecast data table"):
#     st.dataframe(df_combined.tail(n_forecast + 10))




