import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

batch_size = 128
num_epochs = 20
learning_rate = 0.01
test_batch_size = 128
initial_ratio = 0.0
final_ratio = 0.8
add_end_epoch = 50
rm_epoch_first = 60
rm_epoch_second = 80
rm_easy_ratio_first = 0.3
rm_easy_ratio_second = 0.5
alpha = 0.2

csv_file = 'F:\\GRAD STUDIES\\ECE1512\\Project A\\ECE1512_2024F_ProjectA_submission_files\\submission_files\\mhist_dataset\\annotations.csv'
img_dir = 'F:\\GRAD STUDIES\\ECE1512\\Project A\\ECE1512_2024F_ProjectA_submission_files\\submission_files\\mhist_dataset\\images\\images'


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = 1 if self.annotations.iloc[idx, 1] == 'SSA' else 0
        if self.transform:
            image = self.transform(image)
        return image, label

class ConvNet7(nn.Module):
    def __init__(self):
        super(ConvNet7, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512*16*16, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def calculate_el2n(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            true_probs = probabilities.gather(1, labels.view(-1, 1)).squeeze()
            el2n_score = (1 - true_probs).cpu().numpy()
            scores.extend(el2n_score)
    return np.array(scores)

def manage_samples(el2n_scores, dataset, epoch):
    num_samples = len(el2n_scores)
    ratio = initial_ratio + (final_ratio - initial_ratio) * min(epoch / add_end_epoch, 1)
    threshold = int(num_samples * ratio)
    difficult_indices = np.argsort(el2n_scores)[-threshold:]
    return torch.utils.data.Subset(dataset, difficult_indices)

def calculate_trajectory_distance(trajectory1, trajectory2):
    distance = 0.0
    for layer1, layer2 in zip(trajectory1, trajectory2):
        distance += torch.norm(layer1 - layer2).item()
    return distance / len(trajectory1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet7().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
train_dataset = CustomDataset(csv_file, img_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Adjust hard sample ratio
    if epoch < add_end_epoch or epoch in [rm_epoch_first, rm_epoch_second]:
        el2n_scores = calculate_el2n(model, train_loader)
        current_subset = manage_samples(el2n_scores, train_dataset, epoch)
        current_data_loader = DataLoader(current_subset, batch_size=batch_size, shuffle=True)
        hard_sample_ratio = len(current_subset) / len(train_dataset) * 100
    else:
        current_data_loader = train_loader
        hard_sample_ratio = 100

    print(f"Epoch [{epoch+1}/{num_epochs}], Hard Sample Ratio: {hard_sample_ratio:.2f}%")

    for images, labels in current_data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(current_data_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

condensed_model = ConvNet7().to(device)
optimizer = optim.SGD(condensed_model.parameters(), lr=learning_rate, momentum=0.9)
condensed_loader = current_data_loader 

for epoch in range(num_epochs):
    condensed_model.train()
    for images, labels in condensed_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = condensed_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
test_dataset = CustomDataset(csv_file, img_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

condensed_model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = condensed_model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

test_accuracy = total_correct / total_samples
print(f"Test Accuracy on Real Data: {test_accuracy:.4f}")
