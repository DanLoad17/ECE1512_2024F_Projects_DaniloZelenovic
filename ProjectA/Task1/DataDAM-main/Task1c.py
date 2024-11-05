import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_mnist_samples(num_samples=5):
  
    from torchvision.datasets import MNIST
    mnist = MNIST(root='./data', train=True, download=True)
    classes = np.unique(mnist.targets.numpy())
    
    samples_per_class = {}
    for cls in classes:
        idxs = np.where(mnist.targets.numpy() == cls)[0]
        samples_per_class[cls] = np.random.choice(idxs, size=num_samples, replace=False)
    
    return mnist.data, mnist.targets, samples_per_class

def load_mhist_samples(csv_file, img_dir, num_samples=5):
  
    annotations = pd.read_csv(csv_file)
    classes = annotations.iloc[:, 1].unique()
    
    samples_per_class = {}
    for cls in classes:
        idxs = annotations[annotations.iloc[:, 1] == cls].index
        samples_per_class[cls] = np.random.choice(idxs, size=num_samples, replace=False)
    
    return annotations, img_dir, samples_per_class

def display_condensed_images(dataset, samples_per_class, title):
    plt.figure(figsize=(12, 10))
    
   
    for i, (cls, idxs) in enumerate(samples_per_class.items()):
        plt.subplot(3, 4, i + 1) 
        if title == "MNIST":
            images = dataset[0][idxs].numpy()
            plt.imshow(np.mean(images, axis=0), cmap='gray')
        else:
            images = [Image.open(os.path.join(dataset[1], dataset[0].iloc[idx, 0])) for idx in idxs]
            condensed_image = np.mean(np.array([np.array(img) for img in images]), axis=0).astype(np.uint8)
            plt.imshow(condensed_image)
        
        plt.title(f"Class {cls}")
        plt.axis('off')
    
    plt.suptitle(f'Condensed Images per Class - {title}')
    plt.show()


mnist_data, mnist_targets, mnist_samples = load_mnist_samples()
mhist_annotations, mhist_img_dir, mhist_samples = load_mhist_samples(
    'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\annotations.csv',
    'C:\\Users\\goran\\venv\\DataDAM-main\\mhist_dataset\\images\\images'
)


display_condensed_images((mnist_data, mnist_targets), mnist_samples, "MNIST")
display_condensed_images((mhist_annotations, mhist_img_dir), mhist_samples, "MHIST")
