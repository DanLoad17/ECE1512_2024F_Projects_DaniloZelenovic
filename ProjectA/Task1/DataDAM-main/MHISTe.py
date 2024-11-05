import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import time


class ConvNet7(nn.Module):
    def __init__(self, channels=3, num_classes=2):
        super(ConvNet7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_synthetic_mhist_data():
    synthesized_images = torch.rand(100, 3, 32, 32)  
    synthesized_labels = torch.randint(0, 2, (100,))  
    return TensorDataset(synthesized_images, synthesized_labels)


def load_real_mhist_test_data(data_dir, csv_file):
    data = pd.read_csv(csv_file)
    images, labels = [], []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])

    for _, row in data.iterrows():
        if row['Partition'] == 'test':  
            img_path = os.path.join(data_dir, row['Image Name'])
            label = 1 if row['Majority Vote Label'] == 'SSA' else 0  

            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            images.append(image)
            labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return TensorDataset(images, labels)


def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main():
    
    K = 200
    T = 10
    eta_S = 0.1
    zeta_S = 1
    eta_theta = 0.01
    zeta_theta = 50
    num_images_per_class = 50
    batch_size = 128

   
    train_dataset = load_synthetic_mhist_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

   
    data_dir = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\images\\images'  
    csv_file = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv' 
    test_dataset = load_real_mhist_test_data(data_dir, csv_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    model = ConvNet7()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=eta_theta)

   
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    end_time = time.time()
    
   
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy on real data: {test_accuracy:.2f}%")
    print(f"Training Time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
