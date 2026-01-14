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
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_curve, roc_auc_score, confusion_matrix


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
    
    probs_list = []
    targets_list = []
    img_names = []
    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            probs_list.append(probs.cpu().numpy())
            targets_list.append(labels.numpy())

            img_path = test_dataset.image_paths[i]
            img_name = os.path.basename(img_path)
            img_names.append(img_name)

    probs_array = np.concatenate(probs_list)
    targets_array = np.concatenate(targets_list)
    preds_array = np.argmax(probs_array, axis=1)

    if probs_array.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(targets_array, probs_array[:, 1])
        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]
        print(f"Best threshold (Youden): {best_threshold:.4f}")
        preds_array = (probs_array[:, 1] >= best_threshold).astype(int)

    # Calculate metrics
    if probs_array.shape[1] == 2:
        mcm = multilabel_confusion_matrix(targets_array, preds_array, labels=[0, 1])
        tn, fp, fn, tp = mcm[1].ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        auc = roc_auc_score(targets_array, probs_array[:, 1])
        print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, PPV: {ppv:.4f}, NPV: {npv:.4f}, AUC: {auc:.4f}")
    else:
        auc = roc_auc_score(targets_array, probs_array, multi_class="ovr", average="macro")
        print(f"AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(targets_array, preds_array, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(targets_array, preds_array))

    results = []
    for i, img_name in enumerate(img_names):
        pred_class = test_dataset.class_names[preds_array[i]]
        true_class = test_dataset.class_names[targets_array[i]]
        if probs_array.shape[1] == 2:
            prob = probs_array[i, 1]
        else:
            prob = np.max(probs_array[i])
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
