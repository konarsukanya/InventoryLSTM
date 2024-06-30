import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
data = pd.read_csv('Historical_Data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data = data[['Date', 'Sold_Units']]
data.set_index('Date', inplace=True)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create dataset with look-back
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Split into train and test sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

look_back = 3  # Number of previous time steps to use as input to LSTM
train_X, train_Y = create_dataset(train_data, look_back)
test_X, test_Y = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Build LSTM model with hyperparameter tuning
model = Sequential()
model.add(LSTM(units=100, input_shape=(1, look_back)))  
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model with increased epochs and batch size
model.fit(train_X, train_Y, epochs=150, batch_size=32, verbose=2)

# Predictions
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# Evaluate model 
train_rmse = np.sqrt(mean_squared_error(train_Y[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(test_Y[0], test_predict[:,0]))
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")


# Define parameters
desired_service_level = 0.95  # 95% service level
z_score = 1.65  # Z-score for 95% service level
forecast_error_stddev = 10  # Standard deviation of forecast error
lead_time = 7  # Lead time in days

# Calculate safety stock
safety_stock = z_score * forecast_error_stddev * np.sqrt(lead_time)
print(f"Forecast Error Standard Deviation: {forecast_error_stddev}")
print(f"Safety Stock: {safety_stock}")
results = pd.DataFrame({
    'Date': data.index[-len(test_predict):],  # Use the last dates corresponding to test predictions
    'Actual': test_Y[0],
    'Predicted': test_predict[:, 0],
    'Safety_Stock': safety_stock
})

results.to_csv('forecast_results.csv', index=False)
print("Results saved to forecast_results.csv")
