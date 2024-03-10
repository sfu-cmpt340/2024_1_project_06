import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(data_dir="data/lung_image_sets", batch_size=32): # default values for now
    transform = transforms.Compose([
        transforms.Resize((150, 150)), 
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalizer - default ImageNet values
    ])

    # split dataset into training and test set
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # split the dataset into training and testing sets
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # define train and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, dataset.classes

def show_loaded_images(train_loader, class_names):
    """
    Displays the first few images from a DataLoader. Primarily for visualization and confirmation and not required for actual training process
    """
    images, labels = next(iter(train_loader))
    out = torchvision.utils.make_grid(images[:4])  # displaying the first 4 images
    npimg = out.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    labels_list = [class_names[labels[j]] for j in range(len(labels[:4]))] # display class label

    # inverse of normalization formula
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # perform the inverse of normalization 
    for i in range(3): 
        npimg[i] = npimg[i] * std[i] + mean[i]
    
    # display
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(labels_list)
    plt.show()

class LungPathologyModel(nn.Module): # convolutional neural network
    def __init__(self):
        super(LungPathologyModel, self).__init__()

        # convolutional layers, 3 layers for now
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # 3 ch RGB, 16 output ch
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # increase depth from 16 to 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # increase depth to 64

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # fully connected layers
        size_after_conv = 768 // 2**3 # considering input images are 768 x 768
        self.fc1 = nn.Linear(64 * size_after_conv * size_after_conv, 1024)
        self.fc2 = nn.Linear(1024, 3)  # 3 output neurons/classes [lung_aca, lung_n, lung_scc]

    def forward(self, x): # define flow of input through model
        # apply ReLU activation to each convolutional layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten tensor
        x = x.view(-1, 64 * (768 // 2**3) * (768 // 2**3))
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_loader, model):
    pass

if __name__ == '__main__':
    train_loader, test_loader, class_names = load_data()
    model = LungPathologyModel()
    train_model(train_loader, model)
    show_loaded_images(train_loader, class_names)
    print(os.listdir("/data/lung_image_sets"))
