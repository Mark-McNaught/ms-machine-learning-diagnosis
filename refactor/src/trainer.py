import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier



def get_features(model, loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """ Generates feature vectors and labels from a DataLoader using the model."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feat = model(imgs, return_features=True) #uses internal flag to extract features and not get final classifications
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

def get_nca_features(X_train_features, y_train_features_labels, X_test_features, TARGET_DIM=256, SEED=42, MAX_ITER=500, TOL=1e-5):
    """ Applies NCA to reduce feature dimensions. """
    nca = NeighborhoodComponentsAnalysis(
        n_components=TARGET_DIM, 
        random_state=SEED, 
        max_iter=MAX_ITER,
        tol=TOL
    )
    print(f"get_nca_features()>>> Fitting NCA to reduce 512 features to {TARGET_DIM}...")
    
    # NCA is fitted ONLY on the training data
    nca.fit(X_train_features, y_train_features_labels) 

    # Transform both train and test features
    X_train_selected = nca.transform(X_train_features)
    X_test_selected = nca.transform(X_test_features)

    print(f"get_nca_features()>>> Reduced Train Feature Shape: {X_train_selected.shape}")
    print(f"get_nca_features()>>> Reduced Test Feature Shape: {X_test_selected.shape}")

    return X_train_selected, X_test_selected

def get_and_train_knn(X_train_selected, y_train_features_labels, NUM_NEIGHBOURS=20):
    knn_classifier = KNeighborsClassifier(
        n_neighbors=NUM_NEIGHBOURS, 
        weights='distance'
        ) 

    print("get_and_train_knn()>>> Training kNN classifier on NCA selected deep features...")
    knn_classifier.fit(X_train_selected, y_train_features_labels)

    return knn_classifier


def freeze_module(module):
    # Freeze all parameters in the given module
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    # Unfreeze all parameters in the given module
    for p in module.parameters():
        p.requires_grad = True

def get_trainable_parameters(model, param_mode):
    """
    Select parameters based on training mode.

    Assumes:
      - model.model is a torchvision ResNet18
      - model.model.fc is the modified classification head
      - CBAM modules live inside model.model
    """

    if param_mode == "head":
        # Freeze entire ResNet backbone (including CBAM)
        freeze_module(model.model)

        # Unfreeze classifier head
        unfreeze_module(model.model.fc)

        return model.model.fc.parameters()

    elif param_mode == "all":
        # Train everything
        unfreeze_module(model.model)
        return model.model.parameters()

    else:
        raise ValueError(f"Unknown parameter mode: {param_mode}")


def train_model(model, train_loader, config_name, train_configs, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), decision_threshold=0.5):
    # Validate config
    if config_name not in train_configs:
        raise ValueError(f"Config '{config_name}' not found")

    # Establishing training regime based on selected config
    config = train_configs[config_name]

    num_epochs = config["num_epochs"]
    lr = config["lr"]
    criterion = config["criterion"]
    optimiser_class = config["optimiser"]
    param_mode = config["parameters"]


    # Initial optimiser setup
    trainable_params = get_trainable_parameters(model, param_mode)
    optimiser = optimiser_class(trainable_params, lr=lr)

    model.to(device)

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
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

        print(
            f"[{config_name}] "
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Loss: {epoch_loss:.4f} "
            f"- Acc: {epoch_acc:.4f}"
        )

    return losses, accuracies


def plot(losses, accuracies, config_name):
    """
    Plots training loss over epochs given a training config.
    """
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(10, 5))

    # Plot Loss in Red
    plt.plot(epochs, losses, color='red', marker='o', 
             linewidth=2, label='Training Loss')

    # Formatting the axis
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Training Loss: {config_name}')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()