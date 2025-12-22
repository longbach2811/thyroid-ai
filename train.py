from loss.loss import get_loss
from model.model import get_model
from utils.dataset import ImageFolder
from utils.process import training_loops
from utils.transforms import _get_train_transforms, _get_test_transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import argparse
import torch.optim.lr_scheduler as lr_scheduler


def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset = ImageFolder(
        root_dir=args.train_data, transform=_get_train_transforms(imgsz=args.img_size)
    )
    val_dataset = ImageFolder(
        root_dir=args.val_data, transform=_get_test_transforms(imgsz=args.img_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    num_classes = len(train_dataset.class_names)

    # Initialize model
    model = get_model(
        model_name=args.model, num_classes=num_classes, weights=args.weights
    )
    model = model.to(device)

    # Loss function
    criterion = get_loss(
        loss_name=args.loss,
        labels=[label.item() for _, label in train_dataset],
        num_classes=num_classes,
        gamma=args.focal_gamma,
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Scheduler
    scheduler = (
        lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        if args.use_scheduler
        else None
    )

    # Training loops
    training_loops(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
        args.save_path,
        period=args.save_period,
        scheduler=scheduler,
        patience=args.patience,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data",
        default=None,
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data",
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model architecture",
        default="resnet18",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to pre-trained weights"
    )
    parser.add_argument(
        "--loss", type=str, default="focal", help="Loss function"
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Gamma value for Focal Loss"
    )
    parser.add_argument("--img_size", type=int, default=300, help="Input image size")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=None,
        help="Period (in epochs) to save checkpoints",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Use learning rate scheduler",
        default=True,
    )
    parser.add_argument(
        "--patience", type=int, default=7, help="Patience for early stopping"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
