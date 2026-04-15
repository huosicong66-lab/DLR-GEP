import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df = pd.read_csv("beijing_aqi_2022_2024_combined.csv")
df = df[['date', 'AQI']]
df['AQI_t-1'] = df['AQI'].shift(1)
df['AQI_t-2'] = df['AQI'].shift(2)
df.dropna(inplace=True)

split_index = int(len(df) * 0.8)
X = df[['AQI_t-1', 'AQI_t-2']].values
y = df['AQI'].values
date = pd.to_datetime(df['date'].values)

X_train = torch.tensor(X[:split_index], dtype=torch.float32)
X_test = torch.tensor(X[split_index:], dtype=torch.float32)
y_train = torch.tensor(y[:split_index], dtype=torch.float32)
y_test = torch.tensor(y[split_index:], dtype=torch.float32)
date_test = date[split_index:]


num_nodes = X.shape[0]
edge_index_list = []

for i in range(2, num_nodes):
    edge_index_list.append([i, i - 1])
    edge_index_list.append([i, i - 2])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()


x_all = torch.tensor(X, dtype=torch.float32)
y_all = torch.tensor(y, dtype=torch.float32)
data = Data(x=x_all, edge_index=edge_index, y=y_all)


import torch.nn as nn
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x.view(-1)


model = GCN(in_channels=2, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

EPOCHS = 300
patience = 20
best_loss = float('inf')
pat_counter = 0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[:split_index], y_train)
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        pat_counter = 0
    else:
        pat_counter += 1
    if pat_counter >= patience:
        print(f"ealy {epoch} ")
        break

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    y_pred_all = model(data.x, data.edge_index)
    y_pred = y_pred_all[split_index:]

mse = mean_squared_error(y_test, y_pred.numpy())
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred.numpy())
r2 = np.corrcoef(y_test.numpy(), y_pred.numpy())[0, 1] ** 2

print("\nGNN ：")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")


results = pd.DataFrame({
    'date': date_test,
    'True AQI': y_test.numpy(),
    'Predicted AQI': y_pred.numpy()
})
results.to_csv("gnn_selfreg_aqi.csv", index=False, encoding='utf-8-sig')


plt.figure(figsize=(14, 6))
plt.plot(date_test, y_test, label='True AQI', marker='o')
plt.plot(date_test, y_pred, label='Predicted AQI (GNN)', linestyle='--', marker='x')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("GNN Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gnn_selfreg_aqi_plot.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title(f'GNN $R^2$ = {r2:.4f}')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.grid(True)
plt.tight_layout()
plt.savefig("gnn_selfreg_r2_scatter.png", dpi=300)
plt.show()
