import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet_default(nn.Module):
    def __init__(self):
      super(NeuralNet_default, self).__init__()
      self.fc1 = nn.Linear(in_features=8, out_features=64)
      self.fc2 = nn.Linear(in_features=64, out_features=64)
      self.fc3 = nn.Linear(in_features=64, out_features=1)


    def forward(self, x):
      # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
      x = torch.flatten(x, start_dim=1)

      # Pass data through 1st fully connected layer
      x = self.fc1(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 2nd fully connected layer
      x = self.fc2(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 3rd fully connected layer
      x = self.fc3(x)

      return x


class NeuralNet_deep(nn.Module):
    def __init__(self):
      super(NeuralNet_deep, self).__init__()
      self.fc1 = nn.Linear(in_features=8, out_features=64)
      self.fc2 = nn.Linear(in_features=64, out_features=128)
      self.fc3 = nn.Linear(in_features=128, out_features=64)
      self.fc4 = nn.Linear(in_features=64, out_features=1)


    def forward(self, x):
      # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
      x = torch.flatten(x, start_dim=1)

      # Pass data through 1st fully connected layer
      x = self.fc1(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 2nd fully connected layer
      x = self.fc2(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 3rd fully connected layer
      x = self.fc3(x)
      x = F.relu(x)

      x = self.fc4(x)

      return x


class NeuralNet_wide(nn.Module):
    def __init__(self):
      super(NeuralNet_wide, self).__init__()
      self.fc1 = nn.Linear(in_features=8, out_features=128)
      self.fc2 = nn.Linear(in_features=128, out_features=128)
      self.fc3 = nn.Linear(in_features=128, out_features=1)


    def forward(self, x):
      # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
      x = torch.flatten(x, start_dim=1)

      # Pass data through 1st fully connected layer
      x = self.fc1(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 2nd fully connected layer
      x = self.fc2(x)
      # Apply ReLU non-linearity
      x = F.relu(x)

      # Pass data through 3rd fully connected layer
      x = self.fc3(x)

      return x


class NeuralNet_deeper_wide(nn.Module):
    def __init__(self):
        super(NeuralNet_deeper_wide, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return x


class NeuralNet_deep_wider(nn.Module):
    def __init__(self):
        super(NeuralNet_deep_wider, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)


    def forward(self, x):
        # Flatten the data (B, 1, 28, 28) => (B, 784), where B is the batch size
        x = torch.flatten(x, start_dim=1)

        # Pass data through 1st fully connected layer
        x = self.fc1(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 2nd fully connected layer
        x = self.fc2(x)
        # Apply ReLU non-linearity
        x = F.relu(x)

        # Pass data through 3rd fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x