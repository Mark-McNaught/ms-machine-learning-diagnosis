import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), criterion=None, optimiser=None, num_epochs=15):
    """
    NEED TO REWORK THE OPTIMISER LOGIC IN ORDER TO ALLOW FOR DIFFERENT PARAMS TO BE TRAINED
    CURRENTLY ONLY TRAINING THE HEAD THROUGH MODEL.MODEL.FC.PARAMETERS(), ALSO THEREFORE WANT
    A SECOND EPOCH PARAMATER TO ALLOW FOR FINE-TUNING IF DESIRED
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    if optimiser is None:
        optimiser = optim.Adam(model.model.fc.parameters(), lr=1e-3)
    
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

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    return losses, accuracies
