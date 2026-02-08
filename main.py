Step 1: Imports & Synthetic Data
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

timesteps = 1200
t = np.arange(timesteps)

trend = 0.01 * t
seasonal_1 = np.sin(0.02 * t)
seasonal_2 = np.cos(0.04 * t)
noise = np.random.normal(0, 0.2, timesteps)

feature_1 = trend + seasonal_1 + noise
feature_2 = seasonal_2 + 0.5 * feature_1 + noise

data = np.vstack([feature_1, feature_2]).T

Step 2: Differencing & Scaling
data_diff = np.diff(data, axis=0)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_diff)

Step 3: Windowing
def create_sequences(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

Step 4: LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X, y, epochs=10, batch_size=32, verbose=1)

Step 5: Integrated Gradients (Explainability)
def integrated_gradients(model, inputs, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros(inputs.shape)

    interpolated = [
        baseline + (float(i) / steps) * (inputs - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for x in interpolated:
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x)
        grad = tape.gradient(pred, x)
        grads.append(grad)

    avg_grads = tf.reduce_mean(tf.stack(grads), axis=0)
    return (inputs - baseline) * avg_grads

sample = X[:1]
ig_attributions = integrated_gradients(model, sample)

importance = np.mean(np.abs(ig_attributions.numpy()), axis=(0,1))
print("Feature importance:", importance)
