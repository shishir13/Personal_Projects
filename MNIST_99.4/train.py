import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.model import FastMNIST

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders():
    """Prepare MNIST data loaders with augmentation"""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, scheduler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def plot_metrics(train_losses, train_accs, test_losses, test_accs):
    """Plot training and testing metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Testing Losses')
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Testing Accuracies')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed
    set_seed()
    
    # Create model
    model = FastMNIST().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=19,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100
    )
    
    # Training loop
    best_accuracy = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    print("\nStarting Training:")
    print("-" * 60)
    
    for epoch in range(1, 20):
        print(f'\nEpoch: {epoch}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_loss, test_acc = test(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best accuracy: {best_accuracy:.2f}%')
        
        print("-" * 60)
        plot_metrics(train_losses, train_accs, test_losses, test_accs)
    
    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")
    print(f"Final parameter count: {total_params:,}")

if __name__ == '__main__':
    main()
