# Multiclass Image Classification using CNN

This project contains a Python implementation of a Convolutional Neural Network (CNN) for multiclass image classification using the [CIFAR10](https://www.tensorflow.org/datasets/catalog/cifar10) dataset. 

The code allows you to accurately classify images into one of the ten predefined classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

## Code Explanation

- **Imports**: The necessary libraries and modules are imported, including PyTorch (torch), neural network modules (nn), data-related modules (DataLoader, datasets), utilities (os, time), visualization libraries (matplotlib.pyplot, seaborn), and evaluation metrics (confusion_matrix, classification_report).

- **CUDA Optimization**: CUDA backend optimization flags are set using torch.cuda.empty_cache() and cudnn.benchmark = True. These flags optimize the use of GPU memory and speed up computations if a compatible GPU is available.

- **Normalization Parameters**: The code defines the normalization parameters mean and std for the dataset. These parameters are used to normalize the image data during preprocessing.

- **Image Transformations**: The code defines a transform object that performs image transformations using transforms.ToTensor(). This converts the images into tensors suitable for input to the neural network.

- **Dataset Loading**: The CIFAR10 dataset is loaded for training and testing using datasets.CIFAR10. The transform object is applied to the dataset to perform the defined transformations.

- **Data Loaders**: Data loaders are created for training and testing using DataLoader. These data loaders provide batches of data to the neural network during training and evaluation. The batch size, number of workers, and other parameters are set for efficient data loading.

- **Class Labels**: The class labels for CIFAR10 are defined as LABELS. These labels represent the different object classes in the dataset, such as 'plane', 'car', 'bird', etc.

- **Display Training Images**: The code displays a batch of training images using make_grid and plt.imshow. The inv_normalize transformation is used to inverse normalize the images for visualization.

- **CNN Model Definition**: The CNN model is defined as a subclass of nn.Module. It consists of several convolutional layers for feature extraction and fully connected layers for classification. Leaky ReLU activation and batch normalization are applied to the convolutional layers.

- **Model Initialization**: An instance of the CNN model is created and moved to the GPU using .cuda(). This ensures that the model parameters and computations are performed on the GPU if available.

- **Loss Function and Optimizer**: The loss function is defined as cross-entropy loss (nn.CrossEntropyLoss). The stochastic gradient descent (SGD) optimizer is used with specific learning rate, momentum, and weight decay values.

- **Training Loop**: The code iterates over the specified number of epochs. Within each epoch, it iterates over the training dataset batches, performs forward pass, computes loss, backpropagates gradients, and updates the model parameters using the optimizer. Training accuracy and losses are calculated.

- **Validation**: At the end of each epoch, the model is evaluated on the test dataset. Validation accuracy and loss are computed to monitor the model's performance during training.

- **Training and Validation Metrics**: The training and validation losses and accuracies are plotted using matplotlib.pyplot to visualize the learning progress during training.

- **Inference on Test Set**: Inference is performed on the entire test set using torch.no_grad(). Predictions are made for the test images, and the number of correct predictions is recorded.

- **Confusion Matrix**: The confusion matrix is computed using confusion_matrix and visualized using seaborn.heatmap. The confusion matrix provides insights into the model's performance across different classes.

- **Classification Report**: The classification report is printed using classification_report. It displays precision, recall, F1-score, and support for each class.

- **Model Saving**: The trained model is saved to a file using torch.save.

## Requirements

- Python 3+
- Pytorch
- Numpy
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/christianadebambo/cifar10-image-classification-DL.git
   ```
   
## Usage

```cd cifar10-image-classification-DL```

```python cifar10.py```
