import torch

def eval(epoch, model, criterion, loader, device):

    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Epoch: {epoch} | Eval_Loss: {eval_loss/len(loader)} | Eval_Accuracy: {100.*correct/total}')
