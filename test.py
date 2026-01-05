from model.model import get_model
from utils.dataset import ImageFolder
from utils.process import training_loops
from utils.transforms import _get_test_transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import argparse
import csv


def _init_model(args, num_classes):
    model = get_model(model_name=args.model, num_classes=num_classes, weights=None)
    
    if os.path.isfile(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Weights file not found at {args.weights}")
        
    return model

def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ImageFolder(
        root_dir=args.test_data, transform=_get_test_transforms(imgsz=args.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    num_classes = len(test_dataset.class_names)
    
    # Init model
    model = _init_model(args, num_classes)
    model = model.to(device)
    model.eval()
    
    results = []
    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            img_path = test_dataset.image_paths[i]
            img_name = os.path.basename(img_path)
            pred_class = test_dataset.class_names[preds.item()]
            
            results.append([img_name, pred_class])
            
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "prediction"])
        writer.writerows(results)
        
    print(f"Predictions saved to {args.output}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="TN5000_split/test", help="Path to test data")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--img_size", type=int, default=300, help="Input image size")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_model(args)
