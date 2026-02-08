import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------- DATA GENERATION --------------------
np.random.seed(7)
N = 6000
t = np.arange(N)

trend = 0.0005 * t
seasonal_1 = np.sin(2 * np.pi * t / 50)
seasonal_2 = np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 0.3, N)

f1 = trend + seasonal_1 + noise
f2 = 0.5 * f1 + seasonal_2 + noise
f3 = np.roll(f1, 5) + noise
f4 = np.cos(seasonal_1) + noise
f5 = 0.3 * f2 + 0.2 * f3 + noise

data = np.column_stack([f1, f2, f3, f4, f5])

# -------------------- PREPROCESSING --------------------
data_diff = np.diff(data, axis=0)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_diff)

# -------------------- WINDOWING --------------------
def make_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(data_scaled)

train_end = int(0.7 * len(X))
val_end = int(0.85 * len(X))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# -------------------- MODEL --------------------
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=15, batch_size=32, verbose=0)

# -------------------- EVALUATION --------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("RMSE:", rmse)
print("MAE:", mae)

# -------------------- INTEGRATED GRADIENTS --------------------
def integrated_gradients(model, x, steps=50):
    baseline = tf.zeros_like(x)
    interpolated = [baseline + (i / steps) * (x - baseline) for i in range(steps + 1)]
    grads = []
    for inp in interpolated:
        with tf.GradientTape() as tape:
            tape.watch(inp)
            pred = model(inp)
        grads.append(tape.gradient(pred, inp))
    avg_grads = tf.reduce_mean(tf.stack(grads), axis=0)
    return (x - baseline) * avg_grads

sample = X_test[:1]
ig = integrated_gradients(model, sample)
feature_importance = np.mean(np.abs(ig.numpy()), axis=(0, 1))

print("Feature Importance:", feature_importance)
