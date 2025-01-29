import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import optuna

num_classes = 102 


# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")
        
        # Read the corresponding label from the text file
        label_name = os.path.splitext(self.image_filenames[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, 'r') as f:
            label = int(f.readline().strip())

        if self.transform:
            image = self.transform(image)

        return image, label-1

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 5, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = CustomImageDataset(
        image_dir='vgg_data/images/train',
        label_dir='vgg_data/labels/train',
        transform=transform
    )
    val_dataset = CustomImageDataset(
        image_dir='vgg_data/images/val',
        label_dir='vgg_data/labels/val',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained VGG19 model
    model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, num_classes)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        # Early stopping if accuracy does not improve
        if epoch_accuracy >= 95:  # You can adjust this threshold
            break

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy  # The objective to maximize

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and best accuracy
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)
