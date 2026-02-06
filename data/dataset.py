import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import numpy as np
from torchvision import models

def get_embeddings_and_labels(data_root, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=data_root, transform=transform)
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    embeddings, labels = [], []
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    with torch.no_grad():
        for images, targets in dataloader:
            emb = resnet(images.to(device))
            embeddings.append(emb.cpu().numpy())
            labels.append(targets.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

def get_data_loaders(data_root, device, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    embeddings, labels = get_embeddings_and_labels(data_root, device)
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)
    return train_loader, val_loader, y_val