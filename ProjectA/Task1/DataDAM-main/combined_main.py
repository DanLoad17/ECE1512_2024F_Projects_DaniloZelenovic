import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
import os

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for the datasets
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize for compatibility
    transforms.ToTensor(),
])

mhist_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset class for MHIST
class MHISTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.annotations.iloc[idx, 1]  # Assuming the label is in the second column
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Define function to get dataloaders for MNIST and MHIST
def get_dataloaders(batch_size):
    # Load MNIST dataset
    train_dataset_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    test_dataset_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=batch_size, shuffle=True)
    test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=batch_size, shuffle=False)

    # Load MHIST dataset
    mhist_img_dir = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\images\\images'
    mhist_csv_file = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv'
    train_dataset_mhist = MHISTDataset(csv_file=mhist_csv_file, root_dir=mhist_img_dir, transform=mhist_transform)

    train_loader_mhist = DataLoader(train_dataset_mhist, batch_size=batch_size, shuffle=True)

    return train_loader_mnist, test_loader_mnist, train_loader_mhist

# Function to calculate FLOPs
def count_flops(model, input_tensor):
    flops = 0
    def count_conv_flops(module, input, output):
        nonlocal flops
        n_out = output.size(1)  # Number of output channels
        n_in = input[0].size(1)  # Number of input channels
        h_out, w_out = output.size(2), output.size(3)  # Output height and width
        kernel_size = module.kernel_size[0] * module.kernel_size[1]  # Kernel size
        flops += (n_in * n_out * kernel_size * h_out * w_out)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(count_conv_flops))

    # Forward pass to count FLOPs
    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return flops

# Define the ConvNet architecture
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        emb = out.reshape(out.size(0), -1)
        out = self.classifier(emb)
        return emb, out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return nn.SiLU()  # Swish is now SiLU in PyTorch
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=1)]  # Adjust padding to keep the spatial dimensions
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

# Define ConvNet3 and ConvNet7 for both datasets
class ConvNet3(ConvNet):
    def __init__(self, channel, num_classes, net_norm, net_act, net_pooling, im_size=(32, 32)):
        super(ConvNet3, self).__init__(channel, num_classes, net_width=32, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

class ConvNet7(ConvNet):
    def __init__(self, channel, num_classes, net_norm, net_act, net_pooling, im_size=(224, 224)):
        super(ConvNet7, self).__init__(channel, num_classes, net_width=32, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

# Initialize models for MNIST and MHIST
mnist_model_convnet3 = ConvNet3(channel=1, num_classes=10, net_norm='batchnorm', net_act='relu', net_pooling='maxpooling').to(device)

mhist_model_convnet7 = ConvNet7(channel=3, num_classes=5, net_norm='batchnorm', net_act='relu', net_pooling='maxpooling').to(device)  # Adjust class count as necessary

# Initialize criterion and optimizers for each model
criterion = nn.CrossEntropyLoss()
optimizer_mnist_convnet3 = optim.SGD(mnist_model_convnet3.parameters(), lr=0.01)
scheduler_mnist = optim.lr_scheduler.CosineAnnealingLR(optimizer_mnist_convnet3, T_max=20)

optimizer_mhist_convnet7 = optim.SGD(mhist_model_convnet7.parameters(), lr=0.01)
scheduler_mhist = optim.lr_scheduler.CosineAnnealingLR(optimizer_mhist_convnet7, T_max=20)

# Load data
train_loader_mnist, test_loader_mnist, train_loader_mhist = get_dataloaders(batch_size=32)

# Function to evaluate model accuracy
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Example training loop (for illustration, customize as needed)
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # Evaluate accuracy after each epoch
        accuracy = evaluate_model(model, test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')

    return model

# Train models
train_model(mnist_model_convnet3, train_loader_mnist, test_loader_mnist, criterion, optimizer_mnist_convnet3, scheduler_mnist)
train_model(mhist_model_convnet7, train_loader_mhist, train_loader_mhist, criterion, optimizer_mhist_convnet7, scheduler_mhist)  # Using train_loader for evaluation as example

# Calculate and print FLOPs for each model
dummy_input_mnist = torch.randn(1, 1, 32, 32).to(device)
dummy_input_mhist = torch.randn(1, 3, 224, 224).to(device)

flops_mnist = count_flops(mnist_model_convnet3, dummy_input_mnist)
flops_mhist = count_flops(mhist_model_convnet7, dummy_input_mhist)

print(f'MNIST ConvNet3 FLOPs: {flops_mnist}')
print(f'MHIST ConvNet7 FLOPs: {flops_mhist}')
