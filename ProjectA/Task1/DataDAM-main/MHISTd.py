import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import time
import matplotlib.pyplot as plt
from torch.utils import data


class ConvNet7(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet7, self).__init__()
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
            nn.Linear(256 * 16 * 16, 256), 
            nn.ReLU(),
            nn.Linear(256, num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x


def load_mhist_dataset(image_folder, annotations_csv, img_size=(128, 128)):
    df = pd.read_csv(annotations_csv)
    images, labels = [], []
    
    label_mapping = {label: idx for idx, label in enumerate(df['Majority Vote Label'].unique())}

    for _, row in df.iterrows():
        img_name, label = row['Image Name'], row['Majority Vote Label']
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert('L') 
        image = image.resize(img_size)  
        images.append(np.array(image))
        numerical_label = label_mapping[label]
        labels.append(numerical_label)
    
    images = np.array(images).astype(np.float32) / 255.0
    labels = np.array(labels)

    return images, labels


def add_gaussian_noise(images, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=images.size()).to(images.device)
    return images + noise


def evaluate_synset(it_eval, net, images_train, labels_train, args):
    net = net.to(args['device'])    
    images_train = images_train.to(args['device'])
    labels_train = labels_train.to(args['device'])
    
    lr = 0.01
    Epoch = 50
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args['device'])

    dst_train = data.TensorDataset(images_train, labels_train)
    trainloader = data.DataLoader(dst_train, batch_size=128, shuffle=True, num_workers=0)

    start = time.time()
    acc_test, loss_train, acc_train = 0, 0, 0
    
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
    
    net.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for images, labels in trainloader:
            images, labels = images.to(args['device']), labels.to(args['device'])
            outputs = net(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        acc_test = 100. * correct / total
    print(f'Iter {it_eval}: Train Acc = {acc_train:.2f}%, Test Acc = {acc_test:.2f}%')

    return acc_train, acc_test


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


def main():
    image_folder = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\images\\images'
    annotations_csv = 'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv'

    images, labels = load_mhist_dataset(image_folder, annotations_csv)
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  
    labels = torch.tensor(labels, dtype=torch.long)  

    images = add_gaussian_noise(images)

    num_classes = len(np.unique(labels.numpy()))
    K, T, eta_theta = 200, 10, 0.01  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ipc = 50 
    image_syn = torch.randn(size=(num_classes * ipc, 1, 128, 128), dtype=torch.float, requires_grad=True, device=device)
    label_syn = torch.tensor(np.repeat(np.arange(num_classes), ipc), dtype=torch.long, device=device)

    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        image_syn.data[c * ipc:(c + 1) * ipc] = images[idx[:ipc]].float().to(device)

    model = ConvNet7(num_classes=num_classes).to(device)
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

        acc_train, acc_test = evaluate_synset(it, model, image_syn, label_syn, args={'device': device})
        print(f'Iteration {it}, Train Accuracy: {acc_train:.2f}%, Test Accuracy: {acc_test:.2f}%')

    visualize_condensed_images(image_syn, label_syn, num_classes)

if __name__ == "__main__":
    main()
