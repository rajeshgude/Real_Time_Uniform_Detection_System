# uniform_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniformModel(nn.Module):
    def __init__(self):
        super(UniformModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)  # Adjust size based on input resolution
        self.fc2 = nn.Linear(128, 2)  # Binary classification: Uniform / Non-Uniform

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, img_tensor):
        with torch.no_grad():
            output = self.forward(img_tensor)
            _, predicted = torch.max(output, 1)
            return "Uniform" if predicted.item() == 1 else "Non-Uniform"
