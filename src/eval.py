import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def eval(epoch: int, 
         model: Module, 
         criterion: Module, 
         loader: DataLoader, 
         device: torch.device) -> None:
    """
    Evaluate the model's performance on a dataset.

    Parameters:
    - epoch (int): Current epoch number.
    - model (torch.nn.Module): The model to be evaluated.
    - criterion (torch.nn.Module): The loss function used for evaluation.
    - loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated on.
    - device (torch.device): The device to which tensors should be moved before computation.
    """
    
    # Set the model to evaluation mode. In this mode, certain operations like dropout are disabled.
    model.eval()
    
    eval_loss = 0  # Accumulated evaluation loss
    correct = 0    # Count of correctly predicted samples
    total = 0      # Total samples processed
    
    # Disable gradient computations. Since we're in evaluation mode, we don't need gradients.
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):  # Loop through batches of data
            # Move data and target tensors to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute model predictions
            outputs = model(inputs)
            
            # Compute loss for the batch
            loss = criterion(outputs, targets)

            # Accumulate loss and update counts
            eval_loss += loss.item()
            _, predicted = outputs.max(1)  # Get the index of the max log-probability as prediction
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Print evaluation results for the epoch
    print(f'Epoch: {epoch} | Eval_Loss: {eval_loss/len(loader)} | Eval_Accuracy: {100.*correct/total}')
