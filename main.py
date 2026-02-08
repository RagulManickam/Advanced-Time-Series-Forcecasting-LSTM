import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# -----------------------------
# 1. DATA GENERATION (5 FEATURES)
# -----------------------------
np.random.seed(42)
t = np.arange(0, 1500)

f1 = np.sin(0.02 * t)                          # seasonality
f2 = np.cos(0.015 * t)                         # second seasonality
f3 = 0.001 * t                                 # trend
f4 = np.sin(0.01 * t) * np.cos(0.02 * t)       # nonlinear interaction
f5 = np.random.normal(0, 0.1, len(t))          # noise

y = 0.4*f1 + 0.3*f2 + 0.2*f3 + 0.1*f4 + np.random.normal(0,0.05,len(t))

X = np.column_stack([f1,f2,f3,f4,f5])

# -----------------------------
# 2. DIFFERENCING (NON-STATIONARITY HANDLING)
# -----------------------------
X_diff = np.diff(X, axis=0)
y_diff = np.diff(y)

# -----------------------------
# 3. SCALING
# -----------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_diff)
y_scaled = scaler_y.fit_transform(y_diff.reshape(-1,1))

# -----------------------------
# 4. WINDOWING
# -----------------------------
def create_sequences(X, y, window):
    xs, ys = [], []
    for i in range(len(X)-window):
        xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(xs), np.array(ys)

window_size = 20
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# -----------------------------
# 5. LSTM MODEL
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)

# -----------------------------
# 6. HYPERPARAMETER TUNING
# -----------------------------
hidden_sizes = [32,64]
learning_rates = [0.001,0.0005]

best_rmse = float("inf")
best_model = None

for h in hidden_sizes:
    for lr in learning_rates:
        model = LSTMModel(5,h)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(10):
            for xb,yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred,yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).numpy()
            true = y_test_t.numpy()

        rmse = np.sqrt(mean_squared_error(true,preds))
        mae = mean_absolute_error(true,preds)

        print(f"Hidden={h}, LR={lr}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

# -----------------------------
# 7. FINAL EVALUATION
# -----------------------------
best_model.eval()
with torch.no_grad():
    y_pred = best_model(X_test_t).numpy()
    y_true = y_test_t.numpy()

rmse = np.sqrt(mean_squared_error(y_true,y_pred))
mae = mean_absolute_error(y_true,y_pred)

print("\nFinal Model Performance")
print("RMSE:", rmse)
print("MAE:", mae)

plt.plot(y_true[:100], label="True")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("Forecast vs True")
plt.show()

# -----------------------------
# 8. EXPLAINABILITY (Integrated Gradients on one window)
# -----------------------------
ig = IntegratedGradients(best_model)

sample = X_test_t[10:11]   # specific forecast window
attr = ig.attribute(sample)

attr = attr.detach().numpy().squeeze()

plt.bar(range(5), np.mean(attr, axis=0))
plt.xticks(range(5), ["f1","f2","f3","f4","f5"])
plt.title("Integrated Gradients Feature Importance (Single Window)")
plt.show()
