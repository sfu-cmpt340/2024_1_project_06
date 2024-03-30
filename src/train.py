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
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PIL import Image

class CustomPreprocessingTransform:
    def __init__(self, edge_detection=True, contrast_enhancement=True):
        self.edge_detection = edge_detection
        self.contrast_enhancement = contrast_enhancement
    
    def __call__(self, img):
        # convert image to OpenCV format
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        if self.contrast_enhancement:
            # apply CLAHE to help with distinguishing key features like cell boundaries and tissue structures to more accurately distinguish between types of images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_cv)
            l = clahe.apply(l)
            img_cv = cv2.merge((l, a, b))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)

        if self.edge_detection:
            # convert images to grayscale and apply Canny edge detector
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            # convert edges back to real colour
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            img_cv = cv2.addWeighted(img_cv, 0.8, edges, 0.2, 0)

        # convert back to PIL image
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img_pil  
    
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        # original ImageFolder __getitem__ returns (image, target)
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]  # Get the image path
        # return image, target, and path
        return (original_tuple[0], original_tuple[1], path)
    
data_pathway = "data/lung_image_sets" # "C:\data\lung_image_sets"
def load_data(data_dir=data_pathway, batch_size=16, total_subset_size=100): # try to load a subset of the dataset for quicker training
    custom_preprocess = CustomPreprocessingTransform(edge_detection=True, contrast_enhancement=True)
    
    train_transform = transforms.Compose([ # create a seperate training and testing transform (training has image augmentation, testing does not)
        custom_preprocess,
        transforms.Resize((150, 150)), # resize the image for faster performance when training
        RandomHorizontalFlip(),  # augment by flipping images horizontally to try to deal with histopathological images where there are variations in cell apperance
        RandomRotation(10),  # augment by rotating images by up to 10 degrees
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)), 
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),                
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # New augmentation

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # create seperate test transform without image augmentation 
    test_transform = transforms.Compose([
        custom_preprocess,
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the dataset with the training transforms
    full_dataset = CustomImageFolder(data_dir, transform=train_transform)

    assert len(full_dataset.classes) == 3, "Dataset must contain exactly three classes."
    
    # generate a subset of indices and create the training subset
    indices = torch.randperm(len(full_dataset))[:total_subset_size]
    
    # split training and testing subsets
    train_size = int(len(indices) * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # create train subset 
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # create test subset to display images from testing data without iamge augmentation 
    full_dataset.transform = test_transform
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_dataset.classes

# plot the confusion matrix given actual and predicted values
def plot_confusion_matrix(labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

def show_loaded_images_and_predictions(model, loader, class_names, device):
    model.eval()  # set the model to evaluation mode

    images, labels, paths = next(iter(loader))
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Get the actual and predicted values
    #y_true = []
    #y_pred = []
    #y_true.extend(labels.numpy())
    #y_pred.extend(predicted.cpu().numpy())
    images = images.cpu()

    # display images and their predictions
    plt.figure(figsize=(10, 10))

    for i in range(min(len(images), 4)):  # show up to 4 images
        print(paths[i])
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
    #plot_confusion_matrix(y_true, y_pred, class_names)
    plt.show()

class LungPathologyModel(nn.Module):
    def __init__(self):
        super(LungPathologyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # convolutional layers 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # outputted number of channels are 16, so fc1's input will be 16
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(self.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(self.leaky_relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 512 * 4 * 4)  # images start being 150 x 150 pixels, after first convolutional layer, size is /2 (75 x 75) and after second convolutional layer, size is /2 (37x37)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# try Cross entropy loss for loss function, since we are classifying cancer
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.001

def train_model(model, train_loader, optimizer, scheduler, num_epochs=10):
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
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # reduce noise by applying the gaussian blur function

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    equalized_hist = cv2.equalizeHist(blurred)
    block_size = 43
    const_val = 7
    # use adaptive thresholding to try and identify cells to classify each type of image
    thresh = cv2.adaptiveThreshold(equalized_hist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, const_val)

    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.dilate(morph, kernel, iterations=1)

    # find contours which could correspond to cell nuclei
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter out small contours that are not cells
    cells = []
    area_total = []
    for c in contours:
        area_total.append(cv2.contourArea(c))

    for c in contours:
        area = cv2.contourArea(c)
        
        if area > (np.mean(area_total) - np.std(area_total)):
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
        
            if 0.5 < circularity < 1.5:  # range for near-circular shapes
                cells.append(c)

    # show the cell count on the image
    output_img = image.copy()
    for c in cells:
        # Use bounding rectangles to draw an approximate region around each cell
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Img with bounding boxes", output_img)
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
    train_loader, test_loader, class_names = load_data(total_subset_size=100)  # reduce dataset size to a subet of 100 images instead to make training quicker
    model = LungPathologyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # Decays LR by a factor of 0.1 every 7 epochs
    train_model(model, train_loader, optimizer, scheduler, num_epochs=10)
    cm_labels, cm_preds = evaluate_model(model, test_loader, device)
    #plot_confusion_matrix(cm_labels, cm_preds, class_names)
    show_loaded_images_and_predictions(model, test_loader, class_names, device)
