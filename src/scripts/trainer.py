import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier

########################################################################################################
########################################### Module Selector ###########################################
########################################################################################################

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

        "bridge" - For CNNViTHybrid Phase 1 (SRQ3)
            Freeze: CNN backbone + DeiT encoder blocks
            Train: token_proj, cls_token, pos_embed, head

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

    elif param_mode == "bridge":
        # For CNNViTHybrid Phase 1: freeze backbone + encoder, train bridge only
        for p in model.parameters():
            p.requires_grad = False
        for p in model.token_proj.parameters():
            p.requires_grad = True
        model.cls_token.requires_grad = True
        model.pos_embed.requires_grad  = True
        for p in model.head.parameters():
            p.requires_grad = True

        if verbose:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total     = sum(p.numel() for p in model.parameters())
            print("\n=== Phase 1: bridge mode (CNNViTHybrid) ===")
            print(f"Backbone + DeiT encoder frozen")
            print(f"Trainable: token_proj, cls_token, pos_embed, head "
                  f"({n_trainable:,} / {n_total:,} params)")
            print()

        return filter(lambda p: p.requires_grad, model.parameters())

    elif param_mode == "bridge_and_encoder":
        # For CNNViTHybrid Phase 2: keep CNN backbone frozen, train everything else
        # Freeze backbone only
        for p in model.backbone.parameters():
            p.requires_grad = False
        # Unfreeze bridge + encoder
        for p in model.token_proj.parameters():
            p.requires_grad = True
        model.cls_token.requires_grad = True
        model.pos_embed.requires_grad  = True
        for p in model.encoder_blocks.parameters():
            p.requires_grad = True
        for p in model.encoder_norm.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True

        if verbose:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total     = sum(p.numel() for p in model.parameters())
            print("\n=== Phase 2: bridge_and_encoder mode (CNNViTHybrid) ===")
            print("CNN backbone frozen — training: token_proj, cls_token, pos_embed, "
                  "encoder_blocks, encoder_norm, head")
            print(f"Trainable: {n_trainable:,} / {n_total:,} params")
            print()

        return filter(lambda p: p.requires_grad, model.parameters())

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
                         f"Valid options: 'head_and_attention', 'head', 'bridge', "
                         f"'bridge_and_encoder', 'all'")

########################################################################################################
########################################### NCA-kNN Pipeline ###########################################
########################################################################################################

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

########################################################################################################
########################################### Regular Pipeline ###########################################
########################################################################################################

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
    weight_decay = config.get("weight_decay", 0.0)  # optional; default 0 preserves existing behaviour

    trainable_params = get_trainable_parameters(model, param_mode, verbose=verbose)
    optimiser = optimiser_class(trainable_params, lr=lr, weight_decay=weight_decay)

    model.to(device)

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
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

        # Validation phase (if val_loader is provided)
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
            
            # Early stopping logic
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