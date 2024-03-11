import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_dir="C:/data/lung_image_sets", batch_size=16, subset_size=100): # try to load a subset of the dataset for quicker training
    transform = transforms.Compose([
        transforms.Resize((150, 150)), # resize the image for faster performance when training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the dataset with transforms
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    subset_indices = torch.randperm(len(dataset))[:subset_size] # create a random subset of the images to make training faster
    subset = torch.utils.data.Subset(dataset, subset_indices) # create a dataset that contains the subsetted images
    
    # split the training and testing subset data
    total_size = len(subset)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(subset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, dataset.classes

def show_loaded_images_and_predictions(model, loader, class_names, device):
    model.eval()  # set the model to evaluation mode
    images, labels = next(iter(loader))
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images.cpu()

    # display images and their predictions
    plt.figure(figsize=(10, 10))

    for i in range(min(len(images), 4)):  # show up to 4 images
        plt.subplot(2, 2, i + 1)
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Actual: {class_names[labels[i]]}\nPredicted: {class_names[predicted[i]]}") # Show actual and predicted cancer class
        plt.axis("off")
    plt.show()

class LungPathologyModel(nn.Module):
    def __init__(self):
        super(LungPathologyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # convolutional layers 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # outputted number of channels are 16, so fc1's input will be 16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 37 * 37, 120) # images start being 150 x 150 pixels, after first convolutional layer, size is /2 (75 x 75) and after second convolutional layer, size is /2 (37x37)
        self.fc2 = nn.Linear(120, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 37 * 37) # images start being 150 x 150 pixels, after first convolutional layer, size is /2 (75 x 75) and after second convolutional layer, size is /2 (37x37)
        self.fc2 = nn.Linear(120, 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# try cross entropy loss for loss function, since we are distinguishing between distinct classes
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.001

def train_model(model, train_loader, optimizer, num_epochs=3):
    model.train() # set model to training mode

    # loop over the dataset for each epoch
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device) 
            # clear the gradients and set them to 0 to prevent previous gradients from adding up
            optimizer.zero_grad()
            # pass the images through the model - Forward pass
            outputs = model(images)
            # calculate loss based on predictions of model and actual output
            loss = loss_fn(outputs, labels)
            # find gradient of the loss - Backward pass
            loss.backward()
            # optimiziation step - try Adam optimization in main function
            optimizer.step()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = load_data(subset_size=100)  # reduce dataset size to a subet of 100 images instead to make training quicker
    model = LungPathologyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, train_loader, optimizer)
    show_loaded_images_and_predictions(model, test_loader, class_names, device)
