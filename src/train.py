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
    plt.title(labels_list)
    plt.show()


def define_model():
    pass

def train_model(train_loader, model):
    pass

if __name__ == '__main__':
    train_loader, test_loader, class_names = load_data()
    model = define_model()
    train_model(train_loader, model)
    show_loaded_images(test_loader, class_names) # use test dataset for classification/prediction
    print(os.listdir("C:\data\lung_image_sets"))
