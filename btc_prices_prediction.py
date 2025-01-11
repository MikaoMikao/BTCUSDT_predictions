import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input

# load data
file_path = 'btc_usdt.csv'
data = pd.read_csv(file_path)

# data normalisation
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['close', 'volume', 'volatility']])
size = len(data_scaled)

# construct time series data
def LSTM_model(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps - 1):
        x.append(data[i:i + time_steps, :])
        y.append(data[i + time_steps:i + time_steps + 1, 0].mean())
    return np.array(x), np.array(y)

# set time_steps and dataset
time_steps = 24
x, y = LSTM_model(data_scaled, time_steps)

# split dataset
train_size = -72
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# configure LSTM model
model = tf.keras.Sequential([
    Input(shape=(time_steps, x.shape[2])),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model_history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
print(model_history.history)

# get training loss and validation loss
train_loss = model_history.history['loss']
epochs = range(1, len(train_loss) + 1)

# pLot loss figure
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.title("Training Loss", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# predict next 24 hours prices
predictions = model.predict(x_test)
# expand precitions and y_test size to fit the scaled data size
predictions = np.concatenate((predictions, np.zeros((len(predictions), 2))), axis=1)
predicted_prices = scaler.inverse_transform(predictions)[:, 0]
print(predicted_prices)
real = np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 2))), axis=1)
real_prices = scaler.inverse_transform(real)[:, 0]

# plot comparison of predicted prices and real prices
plt.plot(real_prices, label='Real Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.ylabel('Prices',fontsize=12)
plt.xlabel('Hours',fontsize=12)
plt.title('BTC Prices from 2025-01-07-00:00 to 2025-01-10-00:00')
plt.legend(fontsize=12)
plt.show()