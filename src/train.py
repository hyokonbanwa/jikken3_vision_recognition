# fix the code
from tqdm import tqdm
def train(epoch, model, optimizer, criterion, loader, device):

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(loader)
    for (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Epoch: {epoch} | Train_Loss: {train_loss/len(loader)} | Train_Accuracy: {100.*correct/total}')