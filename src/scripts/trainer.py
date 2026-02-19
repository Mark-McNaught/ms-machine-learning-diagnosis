import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier


def save_weights(model, save_path):
    """Save model state dictionary."""
    torch.save(model.state_dict(), save_path)
    print(f"save_weights()>>> Model weights saved to {save_path}")

def load_weights(model, save_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Load model state dictionary."""
    weights = torch.load(save_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    print("load_weights()>>> Model loaded successfully and set to evaluation mode.")
    return model


def get_features(model, loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Generates feature vectors and labels from a DataLoader using the model."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feat = model(imgs, return_features=True)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

def get_nca_features(X_train_features, y_train_features_labels, X_test_features, 
                      TARGET_DIM=256, SEED=42, MAX_ITER=500, TOL=1e-5):
    """Applies NCA to reduce feature dimensions."""
    nca = NeighborhoodComponentsAnalysis(
        n_components=TARGET_DIM, 
        random_state=SEED, 
        max_iter=MAX_ITER,
        tol=TOL
    )
    print(f"get_nca_features()>>> Fitting NCA to reduce 512 features to {TARGET_DIM}...")
    
    nca.fit(X_train_features, y_train_features_labels)
    
    X_train_selected = nca.transform(X_train_features)
    X_test_selected = nca.transform(X_test_features)

    print(f"get_nca_features()>>> Reduced Train Feature Shape: {X_train_selected.shape}")
    print(f"get_nca_features()>>> Reduced Test Feature Shape: {X_test_selected.shape}")

    return X_train_selected, X_test_selected

def get_and_train_knn(X_train_selected, y_train_features_labels, NUM_NEIGHBOURS=20):
    """Trains kNN classifier on NCA-transformed features."""
    knn_classifier = KNeighborsClassifier(
        n_neighbors=NUM_NEIGHBOURS, 
        weights='distance'
    ) 

    print("get_and_train_knn()>>> Training kNN classifier on NCA selected deep features...")
    knn_classifier.fit(X_train_selected, y_train_features_labels)

    return knn_classifier

def validate_model(model, val_loader, criterion, 
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                   decision_threshold=0.5):
    """
    Evaluates model on validation set without updating weights.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds = (torch.sigmoid(outputs) > decision_threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_loss = running_loss / total
    val_acc = correct / total
    
    return val_loss, val_acc


def freeze_module(module):
    """Freeze all parameters in the given module."""
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    """Unfreeze all parameters in the given module."""
    for p in module.parameters():
        p.requires_grad = True

def get_trainable_parameters(model, param_mode, verbose=False):
    """
    Select parameters based on training mode.
    Modes:
        "head_and_attention" - For CNN models (ResNet18 + SE/CBAM)
            Freeze: backbone conv/bn layers
            Train: attention modules + classifier head
        
        "head" - For ViT models (DeiT, EfficientFormer)
            Freeze: transformer encoder
            Train: classifier head only
        
        "all" - For Phase 2 fine-tuning (all models)
            Train: everything
    """

    if param_mode == "head_and_attention":
        # For CNN models with attention
        for name, param in model.model.named_parameters():
            # Freeze backbone layers
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2', 
                                                 'layer3', 'layer4', 'downsample']):
                # But NOT if they're part of attention modules
                if not any(attn in name for attn in ['cbam', 'se', '.ca.', '.sa.', 'cbam_iso']):
                    param.requires_grad = False
                    continue
            
            # Explicitly unfreeze attention modules and classifier head
            if any(attn in name for attn in ['cbam', 'se', '.ca.', '.sa.', 'cbam_iso', 'fc']):
                param.requires_grad = True

        if verbose:
            print("\n=== Phase 1: head_and_attention mode ===")
            print("TRAINABLE parameters:")
            for name, param in model.model.named_parameters():
                if param.requires_grad:
                    print(f"  ✓ {name}")
            print("\nFROZEN parameters:")
            for name, param in model.model.named_parameters():
                if not param.requires_grad:
                    print(f"  ✗ {name}")
            print()

        return filter(lambda p: p.requires_grad, model.model.parameters())
    
    elif param_mode == "head":
        # For ViT models: freeze transformer, train only head
        freeze_module(model.model)
        unfreeze_module(model.head)
        
        if verbose:
            print("\n=== Phase 1: head mode (ViT) ===")
            print("Transformer encoder frozen, head trainable")
            print()
        
        return model.head.parameters()

    elif param_mode == "all":
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        
        if verbose:
            print("\n=== Phase 2: all mode ===")
            print("All parameters trainable")
            print()
        
        return model.parameters()

    else:
        raise ValueError(f"Unknown parameter mode: '{param_mode}'. "
                         f"Valid options: 'head_and_attention', 'head', 'all'")


def train_model(model, train_loader, val_loader, config_name, train_configs, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                decision_threshold=0.5, verbose=False, early_stopping_patience=None):
    """
    Train model with validation monitoring and optional early stopping.
    """
    
    if config_name not in train_configs:
        raise ValueError(f"Config '{config_name}' not found")

    config = train_configs[config_name]

    num_epochs = config["num_epochs"]
    lr = config["lr"]
    criterion = config["criterion"]
    optimiser_class = config["optimiser"]
    param_mode = config["parameters"]

    trainable_params = get_trainable_parameters(model, param_mode, verbose=verbose)
    optimiser = optimiser_class(trainable_params, lr=lr)

    model.to(device)

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # ========== TRAINING PHASE ==========
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimiser.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > decision_threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # ========== VALIDATION PHASE ==========
        if val_loader is not None:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device, decision_threshold)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(
                f"[{config_name}] "
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Train Loss: {epoch_loss:.4f} "
                f"- Train Acc: {epoch_acc:.4f} "
                f"- Val Loss: {val_loss:.4f} "
                f"- Val Acc: {val_acc:.4f}"
            )
            
            # ========== EARLY STOPPING CHECK ==========
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs (patience={early_stopping_patience})")
                        break
        else:
            print(
                f"[{config_name}] "
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Loss: {epoch_loss:.4f} "
                f"- Acc: {epoch_acc:.4f}"
            )

    return losses, accuracies, val_losses, val_accuracies


def plot(losses, accuracies, config_name):
    """Plots training loss over epochs."""
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, losses, color='red', marker='o', 
             linewidth=2, label='Training Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Training Loss: {config_name}')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_training_validation(
    losses_p1, val_losses_p1, accs_p1, val_accs_p1,
    losses_p2, val_losses_p2, accs_p2, val_accs_p2,
    model_name: str = "model",
    save_dir: str = "training-imgs",
    show: bool = True,
):
    """
    Plots training and validation curves for both phases and saves the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=16, fontweight="bold", y=1.01)

    # ── Phase 1 Loss ──────────────────────────────────────────────────────────
    axes[0, 0].plot(losses_p1,     "b-", marker="o", label="Train Loss")
    axes[0, 0].plot(val_losses_p1, "r-", marker="s", label="Val Loss")
    axes[0, 0].set_title("Phase 1: Loss",     fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ── Phase 1 Accuracy ──────────────────────────────────────────────────────
    axes[0, 1].plot(accs_p1,     "b-", marker="o", label="Train Acc")
    axes[0, 1].plot(val_accs_p1, "r-", marker="s", label="Val Acc")
    axes[0, 1].set_title("Phase 1: Accuracy", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ── Phase 2 Loss ──────────────────────────────────────────────────────────
    axes[1, 0].plot(losses_p2,     "b-", marker="o", label="Train Loss")
    axes[1, 0].plot(val_losses_p2, "r-", marker="s", label="Val Loss")
    axes[1, 0].set_title("Phase 2: Loss",     fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ── Phase 2 Accuracy ──────────────────────────────────────────────────────
    axes[1, 1].plot(accs_p2,     "b-", marker="o", label="Train Acc")
    axes[1, 1].plot(val_accs_p2, "r-", marker="s", label="Val Acc")
    axes[1, 1].set_title("Phase 2: Accuracy", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(save_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_utils] Saved → {os.path.abspath(out_path)}")

    if show:
        plt.show()
    plt.close(fig)

    return out_path