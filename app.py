import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from prophet import Prophet  # Import Prophet
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting")

# Load data
# csv_path = 'C:/Users/LENOVO/Downloads/Additional RND/Forcasting_Models/complete_datetime.csv'
# data = pd.read_csv(csv_path)

# -----------------------------
# If you want to use file upload instead, uncomment below:
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.stop()
# -----------------------------

# Check Open column
if 'Open' not in data.columns:
    st.error("CSV file must contain an 'Open' column.")
    st.stop()

# Sidebar model selection
model_type = st.selectbox("Select Model Type", ['Simple RNN', 'LSTM', 'ARIMA', 'SARIMA', 'SARIMAX', 'Prophet'])

# Forecast parameters
n_lookback = st.slider("Lookback period (RNN/LSTM)", 10, 100, 60)
n_forecast = st.slider("Forecast horizon (days)", 1, 60, 30)

if st.button("Generate Forecast"):

    # ------------------
    # RNN and LSTM Processing
    # ------------------------------
    if model_type in ['Simple RNN', 'LSTM']:
        data['Open'] = data['Open'].fillna(method='ffill')
        y = data['Open'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y)

        X, Y = [], []
        for i in range(n_lookback, len(y_scaled) - n_forecast + 1):
            X.append(y_scaled[i - n_lookback:i])
            Y.append(y_scaled[i:i + n_forecast])
        X = np.array(X)
        Y = np.array(Y)

        # Build model
        model = Sequential()
        if model_type == 'Simple RNN':
            model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
            model.add(SimpleRNN(units=50))
        else:
            model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
            model.add(LSTM(units=50))
        model.add(Dense(n_forecast))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train
        with st.spinner(f"Training {model_type} model..."):
            model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

        # Forecast
        X_ = y_scaled[-n_lookback:].reshape(1, n_lookback, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)

        # Dates
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        else:
            data['Date'] = pd.date_range(start='2020-01-01', periods=len(data))

        df_past = data[['Date', 'Open']].copy()
        df_past.rename(columns={'Open': 'Actual'}, inplace=True)
        df_past['Forecast'] = np.nan
        df_past.loc[df_past.index[-1], 'Forecast'] = df_past.loc[df_past.index[-1], 'Actual']

        future_dates = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future = pd.DataFrame({
            'Date': future_dates,
            'Actual': np.nan,
            'Forecast': Y_.flatten()
        })

        df_combined = pd.concat([df_past, df_future], ignore_index=True)
        df_combined.set_index('Date', inplace=True)

        st.dataframe(df_combined.tail(n_forecast + 10))

        # Plot
        st.subheader("📊 Forecast Plot")
        fig, ax = plt.subplots(figsize=(14, 5))
        df_combined['Actual'].plot(ax=ax, label='Actual')
        df_combined['Forecast'].plot(ax=ax, label='Forecast')
        ax.set_title(f"{model_type} Forecast")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # if st.checkbox("Show forecast data table"):
        #     st.dataframe(df_combined.tail(n_forecast + 10))

    # ------------------------------
    # ARIMA Processing
    # ------------------------------
    elif model_type == 'ARIMA':
        st.subheader("🔍 Augmented Dickey-Fuller Test")
        result = adfuller(data['Open'].dropna())
        adf_output = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in result[4].items():
            adf_output[f'Critical Value ({key})'] = value
        st.write(adf_output)

        st.subheader("⚙️ Fitting ARIMA Model")
        with st.spinner("Training ARIMA model..."):
            model = pm.auto_arima(
                y=data['Open'],
                start_p=1, max_p=3,
                start_q=1, max_q=3,
                d=None,
                seasonal=False,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

        st.text("Model Summary:")
        st.text(model.summary())

        # Forecast
        fc, confint = model.predict(n_periods=n_forecast, return_conf_int=True)
        index_of_fc = np.arange(len(data['Open']), len(data['Open']) + n_forecast)

        # Prepare forecast plot
        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        forecast_table = pd.DataFrame({
                'Forecast': fc,
                'Lower CI': confint[:, 0],
                'Upper CI': confint[:, 1], 
            }, index=index_of_fc)
        st.dataframe(forecast_table)

        st.subheader("📊 ARIMA Forecast Plot")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data['Open'], label='Actual')
        ax.plot(fc_series, color='darkgreen', label='Forecast')
        ax.fill_between(lower_series.index, lower_series, upper_series, color='yellow', alpha=0.2, label='Confidence Interval')
        ax.set_title("ARIMA Forecast")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # # Optional table
        # if st.checkbox("Show ARIMA forecast values"):
        #     forecast_table = pd.DataFrame({
        #         'Forecast': fc,
        #         'Lower CI': confint[:, 0],
        #         'Upper CI': confint[:, 1]
        #     })
        #     st.dataframe(forecast_table)



    # --------------------------
    # SARIMA Model
    # --------------------------
    elif model_type == 'SARIMA':
        st.subheader("🔍 Augmented Dickey-Fuller (ADF) Test")

        result = adfuller(data['Open'].dropna())
        adf_output = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in result[4].items():
            adf_output[f'Critical Value ({key})'] = value
        st.write(adf_output)

        st.subheader("⚙️ Fitting Seasonal ARIMA (SARIMA) Model")
        with st.spinner("Training SARIMA model..."):
            model = pm.auto_arima(
                y=data['Open'],
                start_p=1, max_p=3,
                start_q=1, max_q=3,
                test='adf',
                d=None,
                seasonal=True,
                start_P=0,
                D=0,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

        st.text("Model Summary:")
        st.text(model.summary())

        # Forecasting
        fc, confint = model.predict(n_periods=n_forecast, return_conf_int=True)
        index_of_fc = np.arange(len(data['Open']), len(data['Open']) + n_forecast)

        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        forecast_table = pd.DataFrame({
                'Forecast': fc,
                'Lower CI': confint[:, 0],
                'Upper CI': confint[:, 1]
            }, index=index_of_fc)
        st.dataframe(forecast_table)

        # Plot forecast
        st.subheader("📊 SARIMA Forecast Plot")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data['Open'], label='Actual')
        ax.plot(fc_series, color='darkgreen', label='Forecast')
        ax.fill_between(lower_series.index, lower_series, upper_series, color='yellow', alpha=0.2, label='Confidence Interval')
        ax.set_title("Seasonal ARIMA (SARIMA) Forecast")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # if st.checkbox("Show SARIMA forecast table"):
            # forecast_table = pd.DataFrame({
            #     'Forecast': fc,
            #     'Lower CI': confint[:, 0],
            #     'Upper CI': confint[:, 1]
            # }, index=index_of_fc)
            # st.dataframe(forecast_table)

    elif model_type == 'SARIMAX':
        st.subheader("🔍 Augmented Dickey-Fuller (ADF) Test")

        result = adfuller(data['Open'].dropna())
        adf_output = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in result[4].items():
            adf_output[f'Critical Value ({key})'] = value
        st.write(adf_output)

        st.subheader("⚙️ Fitting Seasonal ARIMAX (SARIMAX) Model")
        with st.spinner("Training SARIMA model..."):
            model = pm.auto_arima(
                y=data['Open'],
                start_p=1, max_p=3,
                start_q=1, max_q=3,
                test='adf',
                d=None,
                seasonal=True,
                start_P=0,
                D=1,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

        st.text("Model Summary:")
        st.text(model.summary())

        # Forecasting
        fc, confint = model.predict(n_periods=n_forecast, return_conf_int=True)
        index_of_fc = np.arange(len(data['Open']), len(data['Open']) + n_forecast)

        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        forecast_table = pd.DataFrame({
                'Forecast': fc,
                'Lower CI': confint[:, 0],
                'Upper CI': confint[:, 1]
            }, index=index_of_fc)
        st.dataframe(forecast_table)

        # Plot forecast
        st.subheader("📊 SARIMAX Forecast Plot")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data['Open'], label='Actual')
        ax.plot(fc_series, color='darkgreen', label='Forecast')
        ax.fill_between(lower_series.index, lower_series, upper_series, color='yellow', alpha=0.2, label='Confidence Interval')
        ax.set_title("Seasonal ARIMAX (SARIMAX) Forecast")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # if st.checkbox("Show SARIMAX forecast table"):
        #     forecast_table = pd.DataFrame({
        #         'Forecast': fc,
        #         'Lower CI': confint[:, 0],
        #         'Upper CI': confint[:, 1]
        #     }, index=index_of_fc)
        #     st.dataframe(forecast_table)

    elif model_type == 'Prophet':
        st.subheader("📊 Prophet Forecasting")

        # Prepare data
        df_data = data[['Date', 'Open']].copy()
        df_data['Date'] = pd.to_datetime(df_data['Date'])
        df_data.fillna(method='ffill', inplace=True)

        # Sort data by date to ensure chronological order
        df_data = df_data.sort_values(by="Date")

        # Calculate 70% split index dynamically
        split_index = int(len(df_data) * 0.7)

        # Split Train and Test Data
        df_data_train = df_data.iloc[:split_index]  # First 70% for training
        df_data_test = df_data.iloc[split_index:]   # Remaining 30% for testing

        # Prepare Data for Prophet
        df_train_prophet = df_data_train.rename(columns={"Date": "ds", "Open": "y"})

        # Train Prophet Model
        model_prophet = Prophet(interval_width=0.80)  # Reducing the uncertainty range
        model_prophet.fit(df_train_prophet)

        # Get last available date and last known value
        last_date = df_data["Date"].max()
        last_value = df_data["Open"].iloc[-1]

        # Create future dataframe starting from the last known date
        df_future = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast, freq='D')})

        # Make Predictions
        forecast_prophet = model_prophet.predict(df_future)

        # Extract forecast results
        df_forecast = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        df_forecast = df_forecast.rename(columns={'ds': 'Date'})  # Keep Date column name consistent
        df_forecast['yhat_lower'] = np.maximum(df_forecast['yhat_lower'], df_forecast['yhat'].min() * 0.9)

        # Adjust forecast to continue smoothly from last known value
        adjustment = last_value - df_forecast['yhat'].iloc[0]
        df_forecast[['yhat', 'yhat_lower', 'yhat_upper']] += adjustment

        st.dataframe(df_forecast)

        # Plot the results
        fig, ax = plt.subplots(figsize=(14, 5))

        # Plot all previous actual values
        ax.plot(df_data['Date'], df_data['Open'], 'b-', marker='o', markersize=2, label='Actual Data')

        # Plot only the forecasted values (red dotted line)
        ax.plot(df_forecast['Date'], df_forecast['yhat'], 'r--', marker='.', label='n-Day Forecast')

        # Add confidence intervals
        ax.fill_between(df_forecast['Date'], df_forecast['yhat_lower'], df_forecast['yhat_upper'], color='red', alpha=0.2)

        # Add a vertical line at the last known data point
        ax.axvline(x=last_date, color='black', linestyle='--', label='Last Known Data')

        # Show the legend
        ax.legend()

        # Display the plot
        st.pyplot(fig)

        # # Optional table
        # if st.checkbox("Show Prophet forecast table"):
        #     st.dataframe(df_forecast)
