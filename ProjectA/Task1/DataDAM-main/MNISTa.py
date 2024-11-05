import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from thop import profile


class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),  
            nn.ReLU(),
            nn.Linear(256, 10)  
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x


def calculate_flops(model, input_size):
    input_tensor = torch.randn(input_size)  
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        
        flops, params = calculate_flops(model, (1, 1, 28, 28))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {100 * correct / total:.2f}%, "
              f"FLOPs: {flops}, Params: {params}")


def main():
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
    ])

    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    
    model = ConvNet3()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20)

if __name__ == "__main__":
    main()
