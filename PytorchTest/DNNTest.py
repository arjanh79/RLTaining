import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

class LinearRegression(nn.Module):
    def __init__(self, lr: float, input_features: int):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_features, out_features=1)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, data: tensor) -> tensor:
        x = self.fc1(data)
        return x

    def learn(self, data: tensor, labels: tensor) -> float:
        self.train()
        self.optimizer.zero_grad()

        predictions = self(data)
        batch_loss = self.loss_fn(predictions, labels)

        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.item()


X = torch.randn(1000, 1)
y = 2 * X + 10 + (torch.rand_like(X) * 2)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = LinearRegression(lr=0.03, input_features=1)

summary(model, input_size=(32, 1))

n_epochs = 25
for epoch in range(n_epochs):
    running_loss = 0.0
    for batch_data, batch_labels in train_loader:
        loss = model.learn(batch_data, batch_labels)
        running_loss += loss
    print(f'Weight: {model.fc1.weight.data.item():.3f}'
          f' Bias: {model.fc1.bias.data.item():.3f}')
print('Loss:', running_loss)