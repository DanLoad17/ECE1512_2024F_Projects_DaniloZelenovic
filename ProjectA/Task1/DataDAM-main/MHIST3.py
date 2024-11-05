import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np

class ConvNet7(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = {label: idx for idx, label in enumerate(self.annotations.iloc[:, 1].unique())}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.annotations.iloc[idx, 1]
        label = torch.tensor(self.label_mapping[label], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

def initialize_synthetic_data(num_classes, ipc, device):
    total_images = num_classes * ipc  
    image_syn = torch.randn((total_images, 3, 224, 224), requires_grad=True, device=device)
    label_syn = torch.tensor([np.full(ipc, i) for i in range(num_classes)], dtype=torch.long, device=device).view(-1)
    return image_syn, label_syn

def calculate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def attention_matching_training_loop(model, image_syn, label_syn, criterion, K, T, eta_s, zeta_s, eta_theta, zeta_theta, device, val_loader):
    optimizer = optim.SGD(model.parameters(), lr=eta_theta, momentum=0.9, weight_decay=0.0005)
    synthetic_data_optimizer = optim.SGD([image_syn], lr=eta_s)
    
    for it in range(T):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        
        for k in range(K):
            optimizer.zero_grad()
            outputs = model(image_syn)
            loss = criterion(outputs, label_syn)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += label_syn.size(0)
            correct_train += (predicted == label_syn).sum().item()

        
        for i in range(len(image_syn)):
            synthetic_data_optimizer.zero_grad()
            outputs = model(image_syn[i:i+1])
            attention_loss = criterion(outputs, label_syn[i:i+1])
            attention_loss.backward()
            synthetic_data_optimizer.step()
        
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = calculate_accuracy(model, val_loader, device)
        
        print(f"Iteration {it + 1}/{T} - Loss: {running_loss/K:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    return model

def train_model_with_synthetic_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    ipc = 50 

    
    convnet_model = ConvNet7(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    image_syn, label_syn = initialize_synthetic_data(num_classes, ipc, device)

    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset(csv_file='C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv', img_dir='C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\images\\images', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    
    attention_matching_training_loop(convnet_model, image_syn, label_syn, criterion, K=200, T=10, eta_s=0.1, zeta_s=1, eta_theta=0.01, zeta_theta=50, device=device, val_loader=val_loader)

    
    lenet_model = LeNet(num_classes).to(device)
    optimizer = optim.SGD(lenet_model.parameters(), lr=0.01, momentum=0.9)
    lenet_model.train()
    
    
    for epoch in range(10):  
        total_loss = 0
        for i in range(len(image_syn)):
            optimizer.zero_grad()
            outputs = lenet_model(image_syn[i:i+1])
            loss = criterion(outputs, label_syn[i:i+1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1} - Loss: {total_loss/len(image_syn):.4f}")


    lenet_accuracy = calculate_accuracy(lenet_model, val_loader, device)
    print(f"LeNet Validation Accuracy: {lenet_accuracy:.2f}%")

if __name__ == "__main__":
    train_model_with_synthetic_data()
