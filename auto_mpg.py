import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

data = pd.read_csv(url,
                   na_values='?',
                   comment='\t',
                   header=None,
                   sep=r'\s+',
                   names=['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                          'Weight', 'Acceleration', 'Model Year',
                          'Origin', 'Name'])

data = data.dropna()
data = data.drop('Name', axis=1)
data = pd.get_dummies(data, columns=['Origin'], drop_first=True)

X = data.drop('MPG', axis=1)
y = data['MPG']

X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=True)
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

class MLPModel(nn.Module):
    def __init__(self):
      super(MLPModel, self).__init__()
      self.flatten = nn.Flatten()
      self.linear_relu_stack = nn.Sequential(
          nn.Linear(X_train_tensor.shape[1], 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
      )
    def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

model = MLPModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


num_epochs = 100
model.train()
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    val_loss = criterion(val_predictions, y_val_tensor)
    print(f"\nValidation MSE: {val_loss.item():.4f}")