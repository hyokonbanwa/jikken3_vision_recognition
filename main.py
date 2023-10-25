import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import random

from src.vgg import VGG
from src.train import train
from src.eval import eval
from src.test import test

def main(model='VGG16', image_size=64):

    # Fixing the seed for reproducibility
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # Checking if GPU is available and setting the seed for GPU operations
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    # Preprocess and augmentation for training data
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Preprocess for test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Loading and splitting the dataset into train, validation, and test sets
    dataset = datasets.ImageFolder(root='./data_clean', transform=test_transform)
    train_len = int(len(dataset)*0.8)
    val_len = int(len(dataset)*0.1)
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    train_dataset.transform = train_transform

    # Preparing data loaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Creating the model, loss function, and optimizer
    model = VGG('VGG16', classes=3, image_size=image_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop for 10 epochs
    for epoch in range(1, 11):
        train(epoch, model, optimizer, criterion, train_loader, device)
        eval(epoch, model, criterion, val_loader, device)

    # Testing the model after training
    test(model, criterion, test_loader, device)

    # Saving the trained model weights
    torch.save(model.state_dict(), f'./final_weight.pth')
    print(f'Saved model to ./final_weight.pth')

if __name__ == '__main__':
    main()
