# Image Classification Assignment - Oxford-IIIT Pets Dataset

This repository contains solutions for the image classification assignment based on the Oxford-IIIT Pets dataset. The assignment covers various tasks such as data loading, splitting, model training using ResNet-18, and evaluating the model's performance with metrics like accuracy, F1 score, and confusion matrix. Additionally, the assignment includes visualizing the model's decisions using Grad-CAM.

## Tasks

### Task 1: Load the Oxford-IIIT Pets Dataset
- **Objective**: Load the Oxford-IIIT Pets dataset using Torchvision’s dataset API, including both the train and test partitions.

### Task 2: Split the Train Partition
- **Objective**: Randomly shuffle the train partition and split it further into training and validation sets with an 80%-20% ratio.

### Task 3: Class Distribution in the Training Partition
- **Objective**: Show the class distribution in the training partition.

### Task 4: Define Transform and Dataloaders
- **Objective**: Define a transform to resize the images to a size of 224x224. 
  - Create Dataloaders for the training, validation, and test sets with this transform.

### Task 5: Load Pretrained ResNet-18
- **Objective**: Load the pretrained ResNet-18 model with the default weights (trained on ImageNet).

### Task 6: Modify ResNet-18 for the Dataset
- **Objective**: Extract features after the GAP layer of the ResNet-18 model and replace the final fully connected (FC) layer with one or more 1x1 convolutional layers. The number of filters in the final layer should match the number of classes in the Oxford-IIIT Pets dataset.

### Task 7: Train the Model
- **Objective**: Train the model using Stochastic Gradient Descent (SGD) with a suitable learning rate.

### Task 8: Show Training and Validation Losses
- **Objective**: Show the training and validation losses as a function of the number of epochs.

### Task 9: Report Classification Performance
- **Objective**: Report the classification performance on the test partition of the dataset.
  - Show the confusion matrix, accuracy, and F1 score.

### Task 10: Visualize Class Activation Maps (Grad-CAM)
- **Objective**: Use Grad-CAM to visualize the class activation maps on a few randomly sampled test images (5 images).

## Requirements
- Python 3.x
- PyTorch
- torchvision
- Matplotlib
- torchcam (for Grad-CAM visualization)
- scikit-learn

## Installation

Clone the repository:

```bash
git clone https://github.com/Krishna737Sharma/image-classification-assignment.git
cd image-classification-assignment
