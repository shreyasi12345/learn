import torch
import torch.nn as nn
import torch.optim as optim

# Define Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize Model
model = NeuralNet()
criterion = nn.MSELoss()  # Binary classification loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example Data
X = torch.tensor([[0.5, 0.8], [0.2, 0.4]], dtype=torch.float32)
y = torch.tensor([[1], [0]], dtype=torch.float32)

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

print("Training Complete!")
