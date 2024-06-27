# %% [markdown]
# # Name: Abhay Sharma
# # Roll: 22CH10001

# %%
import os
import torch
import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# %% [markdown]
# Experiment 1

# %%
device = torch.device('cpu')
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
np.random.seed(2022)
random.seed(2022)

# %% [markdown]
# Experiment 2

# %%
dataset_path = 'data'

class FaceAgeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        self.image_paths = []
        self.ages = []
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if os.path.isdir(folder_path):
                age = int(folder)
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    self.image_paths.append(file_path)
                    self.ages.append(age)
        
        data = list(zip(self.image_paths, self.ages))
        random.shuffle(data)
        self.image_paths, self.ages = zip(*data)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        age = self.ages[idx]
        
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, age

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = FaceAgeDataset(dataset_path, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_test_dataset = train_test_split(dataset, train_size=train_size)
val_dataset, test_dataset = train_test_split(val_test_dataset, train_size=val_size)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Overall dataset size: {len(dataset)}")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

# %%
dataiter = iter(train_loader)
images, labels = next(dataiter)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

for i in range(10):
    axs[i].imshow(images[i].numpy().transpose((1, 2, 0)), cmap='gray')
    axs[i].set_title('Label: ' + str(labels[i].item()))
    axs[i].axis('off')
plt.show()

# %% [markdown]
# Experiment 3

# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(32 * 8 * 4, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.bn1(self.relu(self.conv1(x))))
        x = self.pool(self.bn2(self.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = Net().to(device)

# %%
from torchsummary import summary

summary(net, input_size=(3, 32, 32))


# %% [markdown]
# Experiment 4

# %%
dataset_path = 'lab_test_2_dataset'

model = Net()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

num_epochs = 25
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# %% [markdown]
# Experiment 5

# %%
plt.figure(figsize=(8, 6))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# %%
import torch
import matplotlib.pyplot as plt

model = Net()
model.load_state_dict(torch.load('model_epoch_25.pth'))
model.eval()

device = torch.device('cpu')

test_loss = 0.0
predicted_ages = []
ground_truth_ages = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        test_loss += loss.item() * inputs.size(0)

        predicted_ages.extend(outputs.cpu().numpy().squeeze())
        ground_truth_ages.extend(labels.cpu().numpy())

test_loss /= len(test_loader.dataset)
print(f'Test MSE Loss: {test_loss:.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(ground_truth_ages, predicted_ages, alpha=0.5)
plt.xlabel('Ground Truth Age')
plt.ylabel('Predicted Age')
plt.title('Scatter Plot of Predicted vs. Ground Truth Ages')
plt.grid(True)
plt.show()


