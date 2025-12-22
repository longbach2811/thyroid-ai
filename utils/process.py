import os
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import torch.optim.lr_scheduler as lr_scheduler

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    probs_list = []
    preds_list = []
    targets_list = []

    for images, labels in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        probs_list.append(probs.detach().cpu().numpy())
        preds_list.append(preds.detach().cpu().numpy())
        targets_list.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    probs_array = np.concatenate(probs_list)
    preds_array = np.concatenate(preds_list)
    targets_array = np.concatenate(targets_list)

    precision = precision_score(targets_array, preds_array, average='macro', zero_division=0)
    recall = recall_score(targets_array, preds_array, average='macro', zero_division=0)
    
    try:
        if probs_array.shape[1] == 2:
            auc = roc_auc_score(targets_array, probs_array[:, 1])
        else:
            auc = roc_auc_score(targets_array, probs_array, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    return epoch_loss, precision, recall, auc

def validate_one_epoch(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    probs_list = []
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    
    probs_array = np.concatenate(probs_list)
    preds_array = np.concatenate(preds_list)
    targets_array = np.concatenate(targets_list)

    precision = precision_score(targets_array, preds_array, average='macro', zero_division=0)
    recall = recall_score(targets_array, preds_array, average='macro', zero_division=0)
    try:
        if probs_array.shape[1] == 2:
            auc = roc_auc_score(targets_array, probs_array[:, 1])
        else:
            auc = roc_auc_score(targets_array, probs_array, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    return epoch_loss, precision, recall, auc

def training_loops(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, save_path, period=1, scheduler=None, patience=None):
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'runs'))
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_precision, train_recall, train_auc = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch, num_epochs
        )
        val_loss, val_precision, val_recall, val_auc = validate_one_epoch(
            model, val_dataloader, criterion, device, epoch, num_epochs
        )

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"    [Train] Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, AUC: {train_auc:.4f}")
        print(f"    [Val]   Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")

        # Log scalars to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print("Best model saved.")
        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if period is not None and (epoch + 1) % period == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint at epoch {epoch+1}.")

    print("Training complete.")
    writer.close()
    return model
