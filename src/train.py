import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir="data/lung_image_sets", batch_size=32): # default values for now
    transform = transforms.Compose([
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalizer - default ImageNet values
    ])

    # load dataset from directory
    train_dataset = datasets.ImageFolder(data_dir, transform=transform) 

    # Creating a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def define_model():
    pass

def train_model():
    pass

if __name__ == '__main__':
    data = load_data()
    model = define_model()
    train_model(data, model)
