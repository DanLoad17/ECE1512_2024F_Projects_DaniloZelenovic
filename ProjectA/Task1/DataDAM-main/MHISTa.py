import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
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
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
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

def get_dataloaders(batch_size):
    mhist_img_dir = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\images\\images'
    mhist_csv_file = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv'

    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(csv_file=mhist_csv_file, img_dir=mhist_img_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def compute_flops(model, input_size):
    
    dummy_input = torch.randn(1, 3, *input_size).to(next(model.parameters()).device)
    flops = 0

    def flops_hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            batch_size, in_channels, in_height, in_width = input[0].size()
            out_channels, out_height, out_width = output.size(1), output.size(2), output.size(3)
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            flops += batch_size * out_channels * out_height * out_width * (in_channels * kernel_size)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(flops_hook))

    model(dummy_input)

    for hook in hooks:
        hook.remove()

    return flops

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    input_size = (224, 224)  
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        
        flops = compute_flops(model, input_size)
        print(f"Epoch [{epoch + 1}/{num_epochs}], FLOPs: {flops}")

        
        val_accuracy = evaluate_model(model, val_loader, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
              f'Training Accuracy: {epoch_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(pd.read_csv('C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv').iloc[:, 1].unique())
    batch_size = 32
    num_epochs = 20

    model = ConvNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loader, val_loader = get_dataloaders(batch_size)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()
