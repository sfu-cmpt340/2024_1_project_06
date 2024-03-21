import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from sklearn.metrics import classification_report, accuracy_score

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        # Original ImageFolder __getitem__ returns (image, target)
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]  # Get the image path
        # Return image, target, and path
        return (original_tuple[0], original_tuple[1], path)
    
def load_data(data_dir="C:/data/lung_image_sets", batch_size=16, subset_size=100): # try to load a subset of the dataset for quicker training
    transform = transforms.Compose([
        transforms.Resize((150, 150)), # resize the image for faster performance when training
        RandomHorizontalFlip(),  # augment by flipping images horizontally to try to deal with histopathological images where there are variations in cell apperance
        RandomRotation(10),  # Augment by rotating images by up to 10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the dataset with transforms
    dataset = CustomImageFolder(data_dir, transform=transform)
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
    data_iter = iter(loader)
    images, labels, paths = next(iter(loader))
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images.cpu()

    # display images and their predictions
    plt.figure(figsize=(10, 10))

    for i in range(min(len(images), 4)):  # show up to 4 images
        cell_count = count_cells_in_image(paths[i])
        plt.subplot(2, 2, i + 1)
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Actual: {class_names[labels[i]]}\nPredicted: {class_names[predicted[i]]}\nCell Count: {cell_count}") # Show actual and predicted cancer class
        plt.axis("off")
    plt.show()

class LungPathologyModel(nn.Module):
    def __init__(self):
        super(LungPathologyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # convolutional layers 
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # outputted number of channels are 16, so fc1's input will be 16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 18 * 18, 512) # images start being 150 x 150 pixels, after first convolutional layer, size is /2 (75 x 75) and after second convolutional layer, size is /2 (37x37)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 18 * 18) # images start being 150 x 150 pixels, after first convolutional layer, size is /2 (75 x 75) and after second convolutional layer, size is /2 (37x37)
        #self.fc2 = nn.Linear(120, 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# try Cross entropy loss for loss function, since we are classifying cancer
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.001

def train_model(model, train_loader, optimizer, scheduler, num_epochs=3):
    model.train() # set model to training mode

    # loop over the dataset for each epoch
    for epoch in range(num_epochs):
        for i, (images, labels, _) in enumerate(train_loader):
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

            scheduler.step()  # Adjust the learning rate

def count_cells_in_image(image_path):
    # read the image
    image = cv2.imread(image_path)
    # convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # reduce noise by applying the gaussian blur function
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Otsu's thresholding to try and identify cells
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find contours which could correspond to cell nuclei
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # filter out small contours that are not cells
    cells = [c for c in contours if cv2.contourArea(c) > 50]  # 50 is an arbitrary value for minimum cell area

    # show the cell count on the image
    output_img = image.copy()
    for c in cells:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output_img, center, radius, (0, 255, 0), 2)
    cv2.imwrite("/mnt/data/output.png", output_img)

    return len(cells)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            # transfer images and labels to the current computing device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # get the predictions from the maximum value of the output logits
            _, predicted = torch.max(outputs, 1)
            # collect the predictions and true labels
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
    
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.2f}")
    return all_labels, all_preds # Return the lists of labels and predictions for further analysis if needed
       
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = load_data(subset_size=100)  # reduce dataset size to a subet of 100 images instead to make training quicker
    model = LungPathologyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # Decays LR by a factor of 0.1 every 7 epochs
    train_model(model, train_loader, optimizer, scheduler)
    evaluate_model(model, test_loader, device)
    show_loaded_images_and_predictions(model, test_loader, class_names, device)
