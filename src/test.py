import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import random
from src.vgg import VGG

def test(model, criterion, loader, device):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Test_Loss: {test_loss/len(loader)} | Test_Accuracy: {100.*correct/total}')


if __name__ == '__main__':
    image_size = 64

    # fix seed
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = datasets.ImageFolder(root='.././data_clean', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = VGG('VGG16', classes=3, image_size=image_size).to(device)
    model.load_state_dict(torch.load('.././final_weight.pth'))
    criterion = nn.CrossEntropyLoss().to(device)

    test(model, criterion, test_loader, device)
