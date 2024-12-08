from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from torchvision import models
from typing import Union
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from PIL import Image
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Change paths as required
image_dir = "F:\\GRAD STUDIES\\ECE1512\\Project B\\NIH_Dataset\\images"
csv_file = "F:\\GRAD STUDIES\\ECE1512\\Project B\\NIH_Dataset\\Data_Entry_2017_v2020.csv"
labels_file = "F:\\GRAD STUDIES\\ECE1512\\Project B\\NIH_Dataset\\labels.csv"

data_df = pd.read_csv(csv_file)
labels_df = pd.read_csv(labels_file)
labels_dict = dict(zip(labels_df['Image Index'], labels_df['Label']))

@dataclass
class ModelArgs:
    d_model: int 
    n_layer: int 
    vocab_size: int 
    d_state: int = 16  
    expand: int = 2 
    dt_rank: Union[int, str] = 'auto'  
    d_conv: int = 4  
    conv_bias: bool = True 
    bias: bool = False 
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs, num_classes: int):
        """Adapted Mamba model for image classification."""
        super().__init__()
        self.args = args
        
        # Use a pretrained ResNet for feature extraction from images
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()  
        
        self.args.d_model = 512  

        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        
        self.lm_head = nn.Linear(args.d_model, num_classes)

    def forward(self, x):
        """Forward pass for image data."""
        # Extract features from image using ResNet
        x = self.feature_extractor(x) 
        x = rearrange(x, 'b d_model -> b 1 d_model') 
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x.squeeze(1)) 
        
        return logits

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Residual block with Mamba block and normalization."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """Forward pass with residual connection."""
        return self.mixer(self.norm(x)) + x

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Single Mamba block, adapted for image feature size."""
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(in_channels=args.d_inner, out_channels=args.d_inner, bias=args.conv_bias, kernel_size=args.d_conv, groups=args.d_inner, padding=args.d_conv - 1)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward."""
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        """State space model (SSM) pass."""
        b, l, d_in = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Selective scan implementation."""
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        """Root Mean Square Normalization."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
])

train_dataset = ImageFolder(root=image_dir, transform=transform)
test_dataset = ImageFolder(root=image_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

args = ModelArgs(d_model=512, n_layer=6, vocab_size=1000)
model = Mamba(args, num_classes=14) 
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

log_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Validation Accuracy'])

for epoch in range(5): 
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = correct / total
    log_df = pd.concat([log_df, pd.DataFrame({'Epoch': [epoch+1], 'Train Loss': [avg_loss], 'Validation Accuracy': [validation_accuracy]})], ignore_index=True)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

log_df.to_csv('training_logs.csv', index=False)

