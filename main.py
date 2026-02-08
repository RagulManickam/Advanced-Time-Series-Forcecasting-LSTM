# ============================================================
# Advanced Time Series Forecasting with LSTM + Explainability
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients

# ------------------------------------------------------------
# 1. Synthetic Multivariate Time Series Generation
# ------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

timesteps = 1200
t = np.arange(timesteps)

feature_1 = np.sin(0.02 * t) + np.random.normal(0, 0.2, timesteps)
feature_2 = np.cos(0.015 * t) + np.random.normal(0, 0.2, timesteps)
feature_3 = 0.001 * t + np.random.normal(0, 0.1, timesteps)

target = (
    0.5 * feature_1
    + 0.3 * feature_2
    + 0.2 * feature_3
    + np.random.normal(0, 0.1, timesteps)
)

data = np.column_stack([feature_1, feature_2, feature_3, target])

# ------------------------------------------------------------
# 2. Scaling
# ------------------------------------------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ------------------------------------------------------------
# 3. Windowing Function
# ------------------------------------------------------------
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :-1])
        y.append(data[i + window_size, -1])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_sequences(data_scaled, WINDOW_SIZE)

# ------------------------------------------------------------
# 4. Train / Test Split
# ------------------------------------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# ------------------------------------------------------------
# 5. LSTM Model Definition
# ------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size=3, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# 6. Training Loop
# ------------------------------------------------------------
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}")

# ------------------------------------------------------------
# 7. Evaluation
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

plt.figure(figsize=(10, 4))
plt.plot(y_test.numpy(), label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("LSTM Time Series Forecast")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 8. Explainability with Integrated Gradients
# ------------------------------------------------------------
ig = IntegratedGradients(model)

sample_input = X_test[0].unsqueeze(0)
attributions, _ = ig.attribute(sample_input, target=0, return_convergence_delta=True)

attr_mean = attributions.mean(dim=1).squeeze().detach().numpy()

feature_names = ["Feature 1 (sin)", "Feature 2 (cos)", "Feature 3 (trend)"]

plt.figure(figsize=(6, 4))
plt.bar(feature_names, attr_mean)
plt.title("Feature Importance (Integrated Gradients)")
plt.ylabel("Attribution Score")
plt.show()

# ------------------------------------------------------------
# 9. Print Explanation Summary
# ------------------------------------------------------------
for name, score in zip(feature_names, attr_mean):
    print(f"{name}: {score:.4f}")

print("\nModel successfully trained and explained!")
