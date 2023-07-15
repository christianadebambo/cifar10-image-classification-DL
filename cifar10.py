## Multiclass Image Classification using CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import os
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set CUDA backend optimization flags
torch.cuda.empty_cache()
cudnn.benchmark = True 

# Define normalization parameters for the dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Define the image transformations
transform = transforms.ToTensor()

# Create a directory to store the dataset if it doesn't exist
if not os.path.exists('bleh'):
    os.mkdir('bleh')

# Load the CIFAR10 dataset for training and testing
train = datasets.CIFAR10(root='bleh', train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transform, normalize,]))
test = datasets.CIFAR10(root='bleh', train=False, download=True, transform=transforms.Compose([transform, normalize,]))

# Create data loaders for training and testing
train_loader = DataLoader(train, batch_size=10, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test, batch_size=10, shuffle=False, num_workers=4, pin_memory=True)

# Define the class labels
LABELS = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display a batch of training images
for images, labels in train_loader:
    break 

plt.figure(figsize=(10,40))
img = make_grid(images, nrow=16)

# Define the inverse normalization transformation to display the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

img = inv_normalize(img)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        
        # Feature extraction layers
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.MaxPool2d(2, 2))
        
        # Output layers
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(1 * 1 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        
    def forward(self, X):
        X = self.conv_layer1(X)
        X = self.conv_layer2(X)
        X = self.conv_layer3(X)
        X = self.conv_layer4(X)
        X = self.conv_layer5(X)
        
        X = X.reshape(X.size(0), -1)
        
        X = self.dropout(X)
        X = F.leaky_relu(self.fc1(X), negative_slope=0.3)
        X = self.dropout(X)
        X = F.leaky_relu(self.fc2(X), negative_slope=0.3)
        X = self.fc3(X)
        
        return F.log_softmax(X, dim=1)


# Create an instance of the CNN model and move it to the GPU
model = CNN().cuda()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss().cuda()
learning_rate = 0.0003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Define the number of epochs for training
epochs = 60

losses = []
val_losses = []
correct = []
val_correct = []

start_time = time.time()

# Training loop
for epoch in range(epochs):    
    train_corr = 0
    val_corr = 0
    epoch_start_time = time.time()
    
    # Iterate over the training dataset in batches
    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        batch_idx += 1
        
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        
        # Perform a forward pass
        y_hat = model(X_train)
        loss = criterion(y_hat, y_train)
        
        pred = torch.max(y_hat.data, 1)[1]
        batch_corr = (pred == y_train).sum()
        train_corr += batch_corr
        
        # Perform backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    accuracy = (train_corr.item() * 100) / (10 * batch_idx)
            
    losses.append(loss)
    correct.append(train_corr)
    
    # Perform validation at the end of each epoch
    with torch.no_grad():
        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            
            y_val = model(X_test)
            pred = torch.max(y_val, 1)[1]
            val_corr += (pred == y_test).sum()
    
    val_loss = criterion(y_val, y_test)
    val_losses.append(val_loss)
    val_correct.append(val_corr)

    # Display training metrics
    if (epoch % 10 == 0) or (epoch == 0) or (epoch == (epochs - 1)):
        epoch_end_time = time.time()
        print("Epoch {}".format(epoch+1, batch_idx))
        print("Accuracy: {:.4f}  Loss: {:.4f}  Validation Loss: {:.4f}  Duration: {:.2f} minutes".format(
            accuracy, loss, val_loss, ((epoch_end_time - epoch_start_time) / 60)))

end_time = time.time() - start_time

print("\nTraining Duration {:.4f} minutes".format(end_time / 60))
print("GPU memory used: {} kb".format(torch.cuda.memory_allocated()))
print("GPU memory cached: {} kb".format(torch.cuda.memory_cached()))

# Plot the training and validation losses
plt.plot(range(epochs), losses, label='training loss')
plt.plot(range(epochs), val_losses, label='validation loss')
plt.title('Loss Metrics')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Plot the training and validation accuracies
acc = [(train_corr / 500) for train_corr in correct]
val_acc = [(test_corr / 100) for test_corr in val_correct]
plt.plot(range(epochs), acc, label='training accuracy')
plt.plot(range(epochs), val_acc, label='validation accuracy')
plt.title('Accuracy Metrics')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Perform inference on the test set
test_load_all = DataLoader(test, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

# Plot the confusion matrix
arr = confusion_matrix(y_test.view(-1).cpu(), predicted.view(-1).cpu())
df_cm = pd.DataFrame(arr, LABELS, LABELS)
plt.figure(figsize=(9, 6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
plt.xlabel("Prediction")
plt.ylabel("Target")
plt.show()

# Print the classification report
print(f"Classification Report\n\n{classification_report(y_test.view(-1).cpu(), predicted.view(-1).cpu())}")

# Save the model
torch.save(model.state_dict(), 'cifar.pt')
