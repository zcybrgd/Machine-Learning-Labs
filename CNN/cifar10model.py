import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
  def __init__ (self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  #(3 channels RGB qnd 6 filters in this layer)
    self.pool = nn.MaxPool2d(2, 2)                 # 2x2 pooling
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 6 input channels qnd 16 filters
    self.fc1 = nn.Linear(16 * 5 * 5, 120)          # depends on input size
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)   # 10 output classes
    
  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
    x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
    x = x.view(-1, 16 * 5 * 5)         # flatten
    x = F.relu(self.fc1(x))              # (the dense layer)fully connected layer
    x = F.relu(self.fc2(x))
    x = self.fc3(x)     # Final layer (no softmax — use CrossEntropyLoss)
    return x



model = SimpleCNN()
