import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Dataset paths


TRAIN_DIR = "storage/data/train"
VAL_DIR   = "storage/data/val"
TEST_DIR  = "storage/data/test"

# 3. Remove empty class folders 

def remove_empty_class_folders(root_dir):
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path) and len(os.listdir(class_path)) == 0:
            print(f"Removing empty class folder: {class_path}")
            os.rmdir(class_path)

remove_empty_class_folders(VAL_DIR)
remove_empty_class_folders(TEST_DIR)


# 4. Data augmentation

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. Load datasets

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=val_test_transforms)

num_classes = len(train_dataset.classes)
print("Number of classes used for training:", num_classes)


# 6. Data loaders (WINDOWS SAFE → num_workers=0)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# 7. Model (ResNet18 – fine-tuning)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# 8. Loss & optimizer

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# 9. Training 

EPOCHS = 10
total_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    # Training 
    model.train()
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 50 == 0:
            print(f"    Batch {i}/{len(train_loader)}")

    train_loss /= len(train_loader)

    # Validation 
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100*correct / total

    epoch_time = (time.time() - epoch_start) / 60
    elapsed_time = (time.time() - total_start_time) / 60
    avg_epoch_time = elapsed_time / (epoch + 1)
    remaining_time = avg_epoch_time * (EPOCHS - epoch - 1)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Epoch Time: {epoch_time:.2f} min")
    print(f"Estimated Remaining Time: {remaining_time:.2f} min")

# 10. Save artifacts

os.makedirs("artifacts", exist_ok=True)

torch.save(model.state_dict(), "artifacts/breed_model.pth")

with open("artifacts/class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)

print("\nModel saved to artifacts/breed_model.pth")
print("Class names saved to artifacts/class_names.json")

# 11. Final test accuracy

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"\n Test Accuracy: {test_accuracy:.2f}%")
