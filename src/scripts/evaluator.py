import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchmetrics.classification import BinaryCalibrationError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score


def evaluate_model(model=None, test_loader=None, y_true=None, y_pred=None, y_probs=None,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Evaluates model on test set and prints metrics.

    Can accept either:
    - model + test_loader  (runs inference, computes predictions + probabilities)
    - y_true + y_pred + y_probs  (uses pre-computed values; y_probs needed for AUC/ECE)

    Returns:
        acc, prec, rec, f1, auc, ece, conf, report
    """
    if model is not None:
        model.eval()
        y_true  = []
        y_pred  = []
        y_probs = []   # collect sigmoid probabilities for AUC-ROC and ECE

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.clone().detach().float().to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                y_true.extend(labels.cpu().numpy().astype(int))
                y_pred.extend(preds)
                y_probs.extend(probs)   # store probabilities

    # Core metrics
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred)
    rec    = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    conf   = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # AUC-ROC: uses probabilities, not hard predictions
    # roc_auc_score needs continuous scores (not thresholded 0/1) to compute
    # the area under the ROC curve - a threshold-independent performance measure.
    auc = roc_auc_score(y_true, y_probs) if y_probs is not None else float("nan")

    # ECE: calibration quality via torchmetrics BinaryCalibrationError
    # Equal-width binning (n_bins=10), weighted average of confidence-accuracy gaps.
    # A perfectly calibrated model has ECE = 0.
    if y_probs is not None:
        ece_metric = BinaryCalibrationError(n_bins=10, norm='l1')
        ece = ece_metric(
            torch.tensor(y_probs, dtype=torch.float32),
            torch.tensor(y_true,  dtype=torch.long)
        ).item()
    else:
        ece = float("nan")

    # Print and return results
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"ECE       : {ece:.4f}  (lower = better calibrated; 0 = perfect)")
    print("Confusion Matrix:\n", conf)
    print("Classification Report:\n", report)

    return acc, prec, rec, f1, auc, ece, conf, report


def predict_model(model, X_test, y_test, test_transform, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Visualises predictions on random test images."""
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
            pred_label = "MS" if prob >= 0.5 else "Control"
        true_label = "MS" if y_test[idx] == 1 else "Control"
        row = i // 5
        col = i % 5
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"Actual: {true_label}\nPred: {pred_label}", fontsize=11)
        axes[row, col].axis('off')
    plt.suptitle("Model Predictions on Test Images", fontsize=18)
    plt.show()

def predict_knn(knn_classifier, X_test_selected):
    """Predicts test set labels using kNN classifier."""
    print("predict_knn()>>> Predicting test set labels using kNN classifier...")
    y_test_preds = knn_classifier.predict(X_test_selected)
    return y_test_preds