# train_uniform.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from uniform_model import UniformModel

# Dataset Path
data_dir = 'C:\\Users\\19\\Music\\Uniform_Detection\\dress_codes'  # Update your dataset path

# Hyperparameters
batch_size = 8
learning_rate = 0.0005
num_epochs = 5  # Reduced epochs for faster training

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Smaller frame size for faster processing
    transforms.ToTensor(),
])

# Dataset and DataLoader
train_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
model = UniformModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    torch.save(model.state_dict(), 'optimized_uniform_model.pth')

if __name__ == "__main__":
    train()
