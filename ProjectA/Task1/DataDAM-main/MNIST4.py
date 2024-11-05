import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
from tqdm import tqdm
from thop import profile
from argparse import Namespace


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


class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel == 1 else 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc_1(x))
        x = torch.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


def epoch(mode, dataloader, net, optimizer, criterion, args, aug=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp
    return loss_avg, acc_avg


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

# Evaluate function for synthetic dataset
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)

    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = data.TensorDataset(images_train, labels_train)
    trainloader = data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_test = 0
    loss_train = 0
    time_train = 0
    acc_train = 0

    for ep in tqdm(range(Epoch + 1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    time_train = time.time() - start

    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' %
          (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test


def main():
    
    num_classes, ipc = 10, 10
    T, K = 10, 100
    batch_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Namespace(
        device=device,
        lr_net=0.01,
        epoch_eval_train=50,
        batch_train=256
    )

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    
    image_syn = torch.randn((num_classes * ipc, 1, 28, 28), requires_grad=True, device=device)
    label_syn = torch.arange(num_classes).repeat_interleave(ipc).to(device)

    for c in range(num_classes):
        idx = (train_dataset.targets == c).nonzero(as_tuple=True)[0][:ipc]
        image_syn.data[c * ipc:(c + 1) * ipc] = train_dataset.data[idx].float().div(255).unsqueeze(1).to(device)

    model = ConvNet3().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    
    for it in range(T):
        model.train()
        for k in range(K):
            optimizer.zero_grad()
            outputs = model(image_syn)
            loss = criterion(outputs, label_syn)
            loss.backward()
            optimizer.step()

        
        torch.save((image_syn.detach().cpu(), label_syn.cpu()), f"condensed_data_{it}.pth", _use_new_zipfile_serialization=False)

    
    print("\nTraining LeNet on Condensed Datasets")
    lenet = LeNet(channel=1, num_classes=10).to(device)
    lenet_optimizer = optim.SGD(lenet.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for it in range(T):
        condensed_data = torch.load(f"condensed_data_{it}.pth", map_location=device)
        condensed_loader = data.DataLoader(data.TensorDataset(*condensed_data), batch_size=32, shuffle=True)

        for epoch in range(10):
            lenet.train()
            correct, total = 0, 0
            for images, labels in condensed_loader:
                images, labels = images.to(device), labels.to(device)
                lenet_optimizer.zero_grad()
                outputs = lenet(images)
                loss = criterion(outputs, labels)
                loss.backward()
                lenet_optimizer.step()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            print(f'Iteration {it}, Epoch [{epoch+1}/10], Train Accuracy: {100 * correct / total:.2f}%')

        
        evaluate_synset(it, lenet, image_syn, label_syn, test_loader, args)

if __name__ == '__main__':
    main()
