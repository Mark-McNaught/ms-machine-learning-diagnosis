import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def evaluate_model(model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.cpu().numpy().astype(int))
            y_pred.extend(preds)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))


def predict_model(model, X_test, y_test, test_transform, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Visualize predictions on random test images
    num_samples = 10
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    model.eval()
    for i, idx in enumerate(indices):
        img_path = X_test[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            pred_label = "MS" if prob > 0.5 else "Control"
        true_label = "MS" if y_test[idx] == 1 else "Control"
        row = i // 5
        col = i % 5
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"Actual: {true_label}\nPred: {pred_label}", fontsize=11)
        axes[row, col].axis('off')
    plt.suptitle("Model Predictions on Test Images", fontsize=18)
    plt.show()