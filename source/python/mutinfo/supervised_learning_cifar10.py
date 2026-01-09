"""
Supervised Learning on CIFAR-10 with ResNet18
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import json


def adapt_backbone_to_CIFAR(backbone):
    """
    SimCLR paper, page 18: replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1,
    remove the first max pooling operation
    """
    backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
    backbone.maxpool = torch.nn.Identity()
    return backbone


def get_cifar10_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Create CIFAR-10 train and test dataloaders with standard augmentations
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader


class ResNetClassifier(nn.Module):
    """
    ResNet backbone with classification head for CIFAR-10
    """
    def __init__(self, backbone_name='resnet18', num_classes=10, embeddings_dim=128):
        super().__init__()
        # Load backbone
        backbone = getattr(torchvision.models, backbone_name)(num_classes=embeddings_dim)
        self.backbone = adapt_backbone_to_CIFAR(backbone)
        
        # Replace the final FC layer for CIFAR-10 classification
        self.backbone.fc = nn.Linear(512, num_classes)  # ResNet18 has 512 features
        
    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, trainloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / len(trainloader), 100. * correct / total


def evaluate(model, testloader, criterion, device):
    """
    Evaluate model on test set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(testloader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, train_acc, test_acc, checkpoint_dir, is_best=False):
    """
    Save model checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f'Saved best model: {best_path}')
    
    return checkpoint_path


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    # Create model
    print(f'Creating {args.backbone} model...')
    model = ResNetClassifier(
        backbone_name=args.backbone,
        num_classes=10,
        embeddings_dim=args.embeddings_dim
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_test_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f'\nStarting training for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f'\nTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save checkpoint
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc
        
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_acc, test_acc,
                args.checkpoint_dir, is_best=is_best
            )
    
    # Save training history
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nSaved training history: {history_path}')
    
    print(f'\nTraining completed!')
    print(f'Best test accuracy: {best_test_acc:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 Supervised Learning')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--embeddings-dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='./checkpoints/cifar10_supervised',
                        help='Checkpoint directory (default: ./checkpoints/cifar10_supervised)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    main(args)
