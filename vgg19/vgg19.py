import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd

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

# Hyperparameters
batch_size = 32
num_classes = 102 
num_epochs = 10
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
train_dataset = CustomImageDataset(
    image_dir='vgg_data/images/train',
    label_dir='vgg_data/labels/train',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomImageDataset(
    image_dir='vgg_data/images/val',
    label_dir='vgg_data/labels/val',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = CustomImageDataset(
    image_dir='vgg_data/images/test',
    label_dir='vgg_data/labels/test',
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained VGG19 model
model = models.vgg19(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier
model.classifier[6] = nn.Linear(4096, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(num_epochs):
    # Training loop
    total_train = 0
    correct_train = 0
    model.train()
    running_loss = 0.0
    print(f"Train Number {epoch+1}")
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        running_loss += loss.item()
    
    train_loss.append(running_loss / len(train_loader))
    train_accuracy.append(100 * correct_train / correct_train)

    # Validation + test loop
    model.eval()
    running_val_loss = 0.0
    running_test_loss = 0.0
    correct_val = 0
    correct_test = 0
    total_val = 0
    total_test = 0
    with torch.no_grad():
        print(f"Val Number {epoch+1}")
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
        print(f"Test Number {epoch+1}")
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        
    val_loss.append(running_val_loss / len(val_loader))
    val_accuracy.append(100 * correct_val / total_val)
    test_loss.append(running_test_loss / len(test_loader))
    test_accuracy.append(100 * correct_test / total_test)
    
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss[-1]:.4f} | Train Accuracy: {train_accuracy[-1]:.2f}% | Val Loss: {val_loss[-1]:.4f} | Val Accuracy: {val_accuracy[-1]:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {test_accuracy[-1]:.2f}%")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_loss,
    'Train Accuracy (%)': train_accuracy,
    'Val Loss': val_loss,
    'Val Accuracy (%)': val_accuracy,
    'Test Loss': test_loss,
    'Test Accuracy (%)': test_accuracy
})
metrics_df.to_csv("vgg_training_metrics.csv", index=False)

# Save the model
torch.save(model.state_dict(), 'vgg19_transfer_learning.pth')
