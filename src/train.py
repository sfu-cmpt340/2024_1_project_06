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
from configparser import ConfigParser

# Defines a class that preprocesses images by enhancing contrast and performing edge detection based on specified parameters. 
# Utilizes OpenCV for image processing operations and PIL for image format conversion.
class CustomPreprocessingTransform:
    def __init__(self, edge_detection=True, contrast_enhancement=True):
        self.edge_detection = edge_detection
        self.contrast_enhancement = contrast_enhancement
    
    # Convert image to OpenCV format
    def __call__(self, img):
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Apply CLAHE to aid distinguishing key features (cell boundaries, tissue structures) for the purpose of improving accuracy of image type classification.
        if self.contrast_enhancement:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_cv)
            l = clahe.apply(l)
            img_cv = cv2.merge((l, a, b))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)

        # Apply canny edge detection on a gray img, then restore img color.
        if self.edge_detection:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            img_cv = cv2.addWeighted(img_cv, 0.8, edges, 0.2, 0)

        # Convert CV2 image back to PIL format
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img_pil  

# Enhance functionality of ImageFolder class with added capability to retrieve img, labels, and path when accessed from the dataset.   
class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return (original_tuple[0], original_tuple[1], path)

# Retrieve config values from config file with optional default values for cases where the specified values are not found.
def get_config_value(section='DEFAULT', option='data_pathway', default=None):
    config = ConfigParser()
    config.read('config.ini')
    try:
        return config[section][option]
    except KeyError:
        return default

# Prepares training and testing datasets with specific transformations and subset sizes. 
# Detailed step by step instructions of this process are outlined below.
data_pathway = get_config_value(section='DEFAULT', option='data_pathway', default='data/lung_image_sets')
def load_data(data_dir=data_pathway, batch_size=16, total_subset_size=100): # Load a subset of the dataset for quicker training.
    custom_preprocess = CustomPreprocessingTransform(edge_detection=True, contrast_enhancement=True)
    
    train_transform = transforms.Compose([ # Create a seperate training and testing transform (training has image augmentation, testing does not.)
        custom_preprocess,
        transforms.Resize((150, 150)), # Resize the image for faster performance when training.
        RandomHorizontalFlip(),  # Augment by flipping images horizontally to try to deal with histopathological images where there are variations in cell apperance.
        RandomRotation(10),  # Augment by rotating images by up to 10 degrees.
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)), 
        #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),                 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # New augmentation.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create seperate test transform without image augmentation. 
    test_transform = transforms.Compose([
        custom_preprocess,
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset with the training transforms.
    full_dataset = CustomImageFolder(data_dir, transform=train_transform)

    assert len(full_dataset.classes) == 3, "Dataset must contain exactly three classes."

    # Generate a subset of indices and create the training subset.
    indices = torch.randperm(len(full_dataset))[:total_subset_size]

    # Partition training and testing subsets
    train_size = int(len(indices) * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create a train subset. 
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # Create test subset to display images from testing data without image augmentation.
    full_dataset.transform = test_transform
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_dataset.classes

# Plot the confusion matrix with given actual and predicted values.
def plot_confusion_matrix(labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

def show_loaded_images_and_predictions(model, loader, class_names, device):
    model.eval()  # Set the model to evaluation mode.

    images, labels, paths = next(iter(loader))
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images.cpu()

    # Display images and their predictions.
    plt.figure(figsize=(10, 10))

    for i in range(min(len(images), 4)):  # Display up to 4 images.
        print(paths[i])
        cell_count = count_cells_in_image(paths[i])
        plt.subplot(2, 2, i + 1)
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Actual: {class_names[labels[i]]}\nPredicted: {class_names[predicted[i]]}\nCell Count: {cell_count}") # show actual and predicted cancer class.
        plt.axis("off")
    plt.show()

class LungPathologyModel(nn.Module):
    def __init__(self):
        super(LungPathologyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Convolutional layers.
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Outputted number of channels are 16, so fc1's input will be 16.
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
        x = x.view(-1, 512 * 4 * 4)  # Images start being 150 x 150 pixels, after first convolutional layer, size is /2 (75 x 75) and after second convolutional layer, size is /2 (37x37).
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Try Cross entropy loss for loss function, since we are classifying cancer.
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.001

def train_model(model, train_loader, optimizer, scheduler, num_epochs=10):
    model.train() # Set model to training mode.
    # Initialize arrays for plotting.
    matrix_acc = np.zeros((1, 2))
    matrix_err = np.zeros((1, 2))

    # Iterate over dataset for each epoch.
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device) 
            # Clear the gradients and set them to 0 to prevent previous gradients from adding up.
            optimizer.zero_grad()
            # Pass the images through the model - Forward pass.
            outputs = model(images)
            # Calculate loss based on predictions of model and actual output.
            loss = loss_fn(outputs, labels)
            # Find gradient of the loss - Backward pass.
            loss.backward()
            # Optimiziation step - try Adam optimization in main function.
            optimizer.step()

            # Update total loss.
            total_loss += loss.item()
            # Calculate accuracy.
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
        scheduler.step()  # Adjust the learning rate.

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions * 100
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Learning Rate: {scheduler.get_last_lr()[0]}')
        # Append data to plot arrays
        acc_tuple = np.array([[matrix_acc[-1,0]+1, epoch_acc]])
        err_tuple = np.array([[matrix_err[-1,0]+1, epoch_loss]])
        matrix_acc = np.concatenate((matrix_acc, acc_tuple), axis=0)
        matrix_err = np.concatenate((matrix_err, err_tuple), axis=0)

    # Plot the accuracy trend.
    plt.figure(figsize=(8, 8))
    plt.plot(matrix_acc[1:, 0], matrix_acc[1:, 1])
    plt.xlabel('Numbers of Epochs Iterated')
    plt.ylabel('Accuracy Score Percentage')
    plt.title('Accuracy Trend of Trainer per Epoch')

    # Plot the loss trend.
    plt.figure(figsize=(8, 8))
    plt.plot(matrix_err[1:, 0], matrix_err[1:, 1])
    plt.xlabel('Numbers of Epochs Iterated')
    plt.ylabel('Loss Value')
    plt.title('Loss Trend of Trainer per Epoch')

def count_cells_in_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Guassian blur noise reduction.
    #equalized_hist = cv2.equalizeHist(blurred)
    block_size = 43
    const_val = 7
    # Use adaptive thresholding to try and identify cells to classify each type of image.
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, const_val)

    # Apply morphological operations to remove small noise.
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.dilate(morph, kernel, iterations=1)

    # Find contours which could correspond to cell nuclei.
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    area_total = []
    for c in contours:
        area_total.append(cv2.contourArea(c))

    # Contours smaller than one standard of deviation below the mean are filtered out, unlikely to be cells.
    for c in contours:
        area = cv2.contourArea(c)
        
        if area > (np.mean(area_total) - np.std(area_total)):
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
        
            if 0.5 < circularity < 1.5:  # range for near-circular shapes
                cells.append(c)
          
    # Display the cell count on the image.
    # Bounding boxes are drawn in an approximate region around each cell.
    output_img = image.copy()
    for c in cells:
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
    train_loader, test_loader, class_names = load_data(total_subset_size=400)  
    model = LungPathologyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # Decays LR by a factor of 0.1 every 7 epochs
    train_model(model, train_loader, optimizer, scheduler, num_epochs=20)
    cm_labels, cm_preds = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(cm_labels, cm_preds, class_names)
    show_loaded_images_and_predictions(model, test_loader, class_names, device)
