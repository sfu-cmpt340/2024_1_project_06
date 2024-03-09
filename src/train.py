import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def show_loaded_images(train_loader):
    """
    Displays the first few images from a DataLoader. Primarily for visualization and confirmation and not required for actual training process
    """
    images, labels = next(iter(train_loader))
    out = torchvision.utils.make_grid(images[:4])  # displaying the first 4 images
    
    # inverse of normalization formula
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    npimg = out.numpy()
    
    # perform the inverse of normalization 
    for i in range(3): 
        npimg[i] = npimg[i] * std[i] + mean[i]
    
    # display
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_data(data_dir="data/lung_image_sets", batch_size=32): # default values for now
    transform = transforms.Compose([
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalizer - default ImageNet values
    ])

    # load dataset from directory
    train_dataset = datasets.ImageFolder(data_dir, transform=transform) 

    # create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def define_model():
    pass

def train_model(train_loader, model):
    pass

if __name__ == '__main__':
    data = load_data()
    model = define_model()
    train_model(data, model)
    show_loaded_images(data)
