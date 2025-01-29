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

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Hyperparameters
batch_size = 32
num_classes = 102 
num_epochs = 10
learning_rate = 0.001

# Load the model from the saved state dict
model = models.vgg19(pretrained=True)

# Modify the classifier for the number of classes
model.classifier[6] = nn.Linear(4096, num_classes)

# Load the saved state dict
model.load_state_dict(torch.load('models/vgg19_transfer_learning.pth'))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Load the test dataset
test_dataset = CustomImageDataset(
    image_dir='vgg_data/images/test',
    label_dir='vgg_data/labels/test',
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
test_loss = 0.0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()


with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate average loss and accuracy
average_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total

print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
