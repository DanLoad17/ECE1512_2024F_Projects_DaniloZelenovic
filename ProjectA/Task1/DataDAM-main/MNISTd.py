import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from thop import profile
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


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

def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, skip=False):
    net = net.to(args['device'])    
    images_train = images_train.to(args['device'])
    labels_train = labels_train.to(args['device'])
    
    lr = 0.01
    Epoch = 50
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args['device'])

    dst_train = data.TensorDataset(images_train, labels_train)
    trainloader = data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)

    start = time.time()
    acc_test, loss_train, acc_train = 0, 0, 0
    if not skip:
        for ep in range(Epoch + 1):
            net.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in trainloader:
                images, labels = images.to(args['device']), labels.to(args['device'])
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            acc_train = 100. * correct / total
            loss_train = running_loss / len(trainloader)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
        time_train = time.time() - start
        net.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for images, labels in testloader:
                images, labels = images.to(args['device']), labels.to(args['device'])
                outputs = net(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            acc_test = 100. * correct / total
        print(f'Iter {it_eval}: Train Acc = {acc_train:.2f}%, Test Acc = {acc_test:.2f}%')

    return net, acc_train, acc_test

def add_gaussian_noise(images, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=images.size()).to(images.device)
    return images + noise


def main():

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

  
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

   
    num_classes = 10
    channel, im_size = 1, (28, 28)
    ipc = 10  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_syn = torch.randn(size=(num_classes * ipc, channel, *im_size), dtype=torch.float, requires_grad=True, device=device)
    label_syn = torch.tensor([np.ones(ipc) * i for i in range(num_classes)], dtype=torch.long, device=device).view(-1)
    
    for c in range(num_classes):
        idx = (train_dataset.targets == c).nonzero(as_tuple=True)[0]
        image_syn.data[c * ipc:(c + 1) * ipc] = train_dataset.data[idx[:ipc]].float().div(255).unsqueeze(1).to(device)

    
    image_syn = add_gaussian_noise(image_syn)

    
    K, T, eta_s, zeta_s, eta_theta, zeta_theta = 100, 10, 0.1, 1, 0.01, 50
    lambda_bal = 0.01  

    
    model = ConvNet3().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=eta_theta, momentum=0.9, weight_decay=0.0005)

    
    for it in range(T):
        model.train()
        for k in range(K):  
            optimizer.zero_grad()
            outputs = model(image_syn)
            loss = criterion(outputs, label_syn)  

            loss.backward()
            optimizer.step()

        
        net, acc_train, acc_test = evaluate_synset(it, model, image_syn, label_syn, test_loader, args={'device': device})
        print(f'Iteration {it}, Train Accuracy: {acc_train:.2f}%, Test Accuracy: {acc_test:.2f}%')

    
    visualize_condensed_images(image_syn, label_syn, num_classes)

def visualize_condensed_images(images, labels, num_classes):
    condensed_images = []
    for cls in range(num_classes):
        class_images = images[labels == cls]
        condensed_image = class_images.mean(dim=0).detach().cpu().numpy()
        condensed_images.append(condensed_image)

    
    plt.figure(figsize=(15, 8))
    for cls, img in enumerate(condensed_images):
        plt.subplot(2, 5, cls + 1)
        plt.imshow(img[0], cmap='gray')  
        plt.title(f'Class {cls}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
