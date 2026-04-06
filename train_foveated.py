# coding=utf-8
"""
FoveatedViT Training Script for OCT-MNIST Classification
"""

import os
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CIFAR10JPGDataset(Dataset):
    """Custom Dataset for OCT-MNIST format JPG images."""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        split_dir = self.root_dir / self.split
        for idx, class_name in enumerate(self.classes):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                class_dir = split_dir / f"{idx:02d}_{class_name}"
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        random.shuffle(self.samples)
        logger.info(f"Loaded {len(self.samples)} images from {self.split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # expand 1-ch grayscale to 3-ch RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AverageMeter:
    """Computes and stores average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(args):
    """Create data loaders for training and validation."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # grayscale→RGB: mean=0.5, std=0.5
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # grayscale→RGB: mean=0.5, std=0.5
    ])
    
    train_dataset = CIFAR10JPGDataset(
        root_dir=args.data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = CIFAR10JPGDataset(
        root_dir=args.data_dir,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def validate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    val_loss_sum = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits, _ = model(images)

            loss = criterion(logits, labels)
            val_loss_sum += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader)
    accuracy = correct / total

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_accuracy = {}
    for class_idx in range(4):
        mask = all_labels == class_idx
        if mask.sum() > 0:
            class_accuracy[class_idx] = (all_preds[mask] == all_labels[mask]).mean()
        else:
            class_accuracy[class_idx] = 0.0

    return val_loss, accuracy, class_accuracy


def train(args, model, train_loader, val_loader, device):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    best_accuracy = 0.0
    best_class_accuracy = None
    
    for epoch in range(args.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            loss = model(images, labels)
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        scheduler.step()
        
        # Validation
        val_loss, val_acc, class_acc = validate(model, val_loader, device)
        
        classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*50}")
        print(f"  Train Loss: {loss_meter.avg:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc*100:.2f}%")
        print(f"  Per-class Accuracy:")
        for idx, cls_name in enumerate(classes):
            print(f"    {cls_name}: {class_acc[idx]*100:.2f}%")
        print(f"{'='*50}\n")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_class_accuracy = class_acc
            save_model(args, model, epoch+1, val_acc)
    
    return best_accuracy, best_class_accuracy


def save_model(args, model, epoch, accuracy):
    """Save model checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'accuracy': accuracy,
    }, checkpoint_path)


def main():
    import sys
    from pathlib import Path

    # 解决 models 模块导入问题，确保可以找到 models.foveated_vit
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(description='FoveatedViT Training')

    # autodl 平台路径配置
    # 数据存放路径（相对于脚本位置）
    # 训练输出路径: ./output
    parser.add_argument('--data_dir', type=str,
                       default='data/octmnist_jpg_subset',
                       help='Data directory (relative to script location)')
    parser.add_argument('--output_dir', type=str,
                       default='output',
                       help='Output directory (relative to script location)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(args)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    from models.foveated_vit import FoveatedViT, get_foveated_b16_full_config
    config = get_foveated_b16_full_config()
    model = FoveatedViT(
        config=config,
        img_size=224,
        num_classes=4,
        zero_head=True
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {num_params:.2f}M")

    # Log token info
    token_info = model.get_token_info()
    logger.info(f"Token info: center={token_info['num_center_tokens']}, "
                 f"outer={token_info['num_outer_tokens']}, "
                 f"total={token_info['num_total_tokens']} (excluding cls)")

    # Train
    best_acc, best_class_acc = train(args, model, train_loader, val_loader, device)
    
    # Print final results
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    logger.info("=" * 50)
    logger.info("Training Completed!")
    logger.info(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    logger.info("Per-class Accuracy:")
    for idx, cls_name in enumerate(['CNV', 'DME', 'DRUSEN', 'NORMAL']):
        logger.info(f"  {cls_name}: {best_class_acc[idx]*100:.2f}%")


if __name__ == '__main__':
    main()
