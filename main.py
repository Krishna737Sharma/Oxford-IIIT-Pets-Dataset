!pip install torchinfo
!pip install torchcam

import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from torchcam.methods import SmoothGradCAMpp
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
import torchvision.models as models
from PIL import Image

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import classification_report
from torchinfo import summary
from torchvision import transforms
from PIL import Image

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets
import torch.optim as optim

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

"""# **Question 1 & Question 4**"""

# Define the transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()  # Convert images to tensor
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize validation/test images to 224x224
    transforms.ToTensor()  # Convert images to tensor
])

# Download the Oxford IIIT Pet Dataset using the in-built PyTorch datasets library
raw_train_dataset = torchvision.datasets.OxfordIIITPet(root='./data/oxford-pets', download=True, transform=train_transforms)

# Download the test split of the dataset
raw_test_dataset = torchvision.datasets.OxfordIIITPet(root='./data/oxford-pets', split='test', download=True, transform=val_test_transforms)

print(len(raw_train_dataset))
print(len(raw_test_dataset))

classes = raw_train_dataset.classes

# Print class names
print("Class labels and their corresponding names:")
for idx, class_name in enumerate(classes):
    print(f"Label {idx}: {class_name}")

breed_to_image = {}

for image_file, breed in raw_train_dataset:
    if breed not in breed_to_image:
        breed_to_image[breed] = image_file


print(len(breed_to_image))

print(type(breed_to_image[1]))
# Convert the tensor to H, W, C format (from C, H, W)
image = breed_to_image[0].permute(1, 2, 0)

# Display the image
plt.imshow(image)
plt.show()

import matplotlib.pyplot as plt

# Plot one image per class
plt.figure(figsize=(15, 10))
for breed in breed_to_image:
    # Convert from (C, H, W) to (H, W, C) by permuting the tensor
    image = breed_to_image[breed].permute(1, 2, 0)

    plt.subplot(5, 8, breed+1)
    plt.imshow(image)
    plt.title(classes[breed])
    plt.axis('off')

plt.tight_layout()
plt.show()

from collections import Counter
import os

# Get the class names (using the full train dataset)
class_names = raw_train_dataset.classes
print("Class Names:", class_names)

# Initialize a counter for the number of images in each class
class_counts = Counter()

# Iterate through the images in the dataset (use train_dataset, not train_set)
for _, label in raw_train_dataset:
    class_counts[label] += 1

# Display the class counts
for class_id, count in class_counts.items():
    class_name = class_names[class_id]
    print(f"Class '{class_name}': {count} images")

"""# **Question 2**"""

# Shuffle and split the train set into 80% training and 20% validation
train_size = int(0.8 * len(raw_train_dataset))  # 80% for training
val_size = len(raw_train_dataset) - train_size  # 20% for validation
train_set, val_set = random_split(raw_train_dataset, [train_size, val_size])

print(len(train_set))
print(len(val_set))
print(len(raw_test_dataset))

"""# **Question 3**"""

# Get the class names (using the full train dataset)
class_names = raw_train_dataset.classes

# Initialize a counter for the number of images in each class
class_counts = Counter()

# Iterate through the images in the training set
for _, label in train_set:
    class_counts[label] += 1

# Prepare data for plotting
class_labels = [class_names[i] for i in class_counts.keys()]
counts = list(class_counts.values())

# Plot the class distribution as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(class_labels, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Training Set (80%)')
plt.xticks(rotation=90)
plt.show()

"""# **Question 4**"""

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(raw_test_dataset, batch_size=32, shuffle=False)

# Example: Checking the shape of a batch from the train_loader
for images, labels in train_loader:
    print(f'Batch of images shape: {images.shape}')
    break

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""# **Question 5** **& Question 6**"""

import torchvision.models as models

# Load ResNet-18 pretrained on ImageNet
model = models.resnet18(pretrained=True)

import torch.nn as nn

# Get the number of classes from the dataset
num_classes = len(raw_train_dataset.classes)

# Get the input features for the fully connected layer (Linear layer)
in_features = model.fc.in_features

# Replace the fully connected layer with a Conv2d layer
model.fc = nn.Conv2d(in_channels=in_features, out_channels=num_classes, kernel_size=1)

# Override the forward method of the ResNet model
class MyResNet(nn.Module):
    def __init__(self, original_model):
        super(MyResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # Everything except the last layer
        self.fc = model.fc  # Your new Conv2d layer

    def forward(self, x):
        x = self.features(x)
        # Do not flatten here
        x = self.fc(x)
        x = torch.flatten(x, 1) # Flatten after your custom layers
        return x

model = MyResNet(model)
model = model.to(device)  # Move the model to the GPU

print(model)

from torchinfo import summary

# Summarize the model
model_summary = summary(model, input_size=(1, 3, 224, 224), verbose=1)

"""# **Question 7 **"""

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Lists to store the losses
train_losses = []
val_losses = []

# Training loop with loss tracking
num_epochs = 10
for epoch in range(num_epochs):
    # Training loop
    model.train()
    epoch_train_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')

"""# **Question 8**"""

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.legend()
plt.grid(True)

# Add value labels on the points
for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
    plt.annotate(f'{train_loss:.3f}', (i+1, train_loss), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'{val_loss:.3f}', (i+1, val_loss), textcoords="offset points", xytext=(0,-15), ha='center')

plt.tight_layout()
plt.show()

"""# **Question 9**"""

# Evaluate the model on the test set
model.eval()
y_true = []
y_pred = []
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

def plot_advanced_confusion_matrix(y_true, y_pred, class_names, normalize=False):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    # Convert to DataFrame
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Create figure
    plt.figure(figsize=(20, 16))

    # Custom colormap with better visibility
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    # Create heatmap with custom settings
    sns.heatmap(cm_df,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
                annot_kws={'size': 8},
                # Use bool instead of np.bool
                mask=np.zeros_like(cm, dtype=bool))

    # Customize the plot
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=45, ha='right', fontsize=10)

    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)

    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    plt.title(title, fontsize=14, pad=20)

    # Add a text box with metrics
    accuracy = np.trace(cm) / np.sum(cm)
    stats_text = f'Accuracy: {accuracy:.2%}\n'
    plt.figtext(0.99, 0.01, stats_text, fontsize=10, ha='right')

    plt.tight_layout()

    # Save with appropriate filename
    filename = 'normalized_confusion_matrix.png' if normalize else 'confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Plot both normalized and non-normalized versions
plot_advanced_confusion_matrix(y_true, y_pred, class_names, normalize=False)
plot_advanced_confusion_matrix(y_true, y_pred, class_names, normalize=True)

# Print detailed metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print per-class metrics
print("\nPer-class Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

"""# **Question 10**"""

import random

# Define transformation for resizing and converting images to tensor
transform = transforms.Compose([
])

# Load pretrained ResNet-18 model and set it to evaluation mode
model = models.resnet18(pretrained=True)
model.to(device)
model.eval()

# Set requires_grad=True only for the layer4 block
for name, param in model.named_parameters():
    if 'layer4' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Initialize the CAM extractor with the specified layer ('layer4')
cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')

# Select 5 random images from the test set
sample_indices = random.sample(range(len(raw_test_dataset)), 3)

# Process and display CAM for each random image
for idx in sample_indices:
    # Load and preprocess the sample image
    image, label = raw_test_dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)

    # Display the original image
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Original Image (Class: {label})")
    plt.show()

    # Perform a forward pass to get model output
    output = model(input_tensor)
    predicted_class = output.argmax().item()

    # Generate the CAM for the predicted class
    activation_map = cam_extractor(predicted_class, output)

    # Resize the activation map to match the input image dimensions
    activation_map_resized = transforms.functional.resize(activation_map[0].unsqueeze(0), (224, 224)).squeeze()

    # Convert the activation map to a numpy array for visualization
    activation_map_resized = activation_map_resized.cpu().detach().numpy()

    # Plot the original image and overlay the CAM
    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.imshow(activation_map_resized, cmap='jet', alpha=0.5)  # Overlay CAM with transparency
    ax.axis('off')
    plt.show()
