import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold


#####################################################################################################
######################## Functions for downloading and processing MS dataset ########################
#####################################################################################################

def get_dataset(DATA_DIR):
    """
    Downloads the MS dataset from Kagglehub if it is not already downloaded.
    Returns the path to the dataset as well as a list of available subdatasets.
    """
    
    # Set custom download directory
    os.environ["KAGGLEHUB_CACHE"] = DATA_DIR

    # Download dataset if not already present
    if not os.listdir(DATA_DIR):
        path = kagglehub.dataset_download("buraktaci/multiple-sclerosis")
        print("get_dataset()>>> Dataset downloaded to:", path)
    else:
        print(f"get_dataset()>>> Dataset already exists in {DATA_DIR}")
        path = os.path.join(DATA_DIR, "datasets/buraktaci/multiple-sclerosis/versions/1/MS/")
    
    # List available categories
    categories = os.listdir(path)
    sorted_categories = sorted(categories)
    print("get_dataset()>>> Available categories:", sorted_categories)

    return path, sorted_categories

def get_classes(path, categories, axial=True, saggital=True, visualise=True, num_samples=3):
    """
    Returns selected classes from the dataset, and if specified, visualises sample images.
    Axial and Saggital flags to determine what parts of the dataset to include.
    """
    classes = []
    if axial:
        classes.extend([categories[0], categories[2]])
    if saggital:
        classes.extend([categories[1], categories[3]])
    
    if visualise:
        print("get_classes()>>> Visualising sample images from each category...")
        for cat in classes:
            image_dir = os.path.join(path, cat)
            category_images = os.listdir(image_dir)
            fig, ax = plt.subplots(1, num_samples, figsize=(15, 5))
            fig.suptitle(f"{cat} - Sample images", fontsize=18)
            for i in range(num_samples):
                idx = np.random.randint(0, len(category_images))
                img = np.array(Image.open(os.path.join(image_dir, category_images[idx])))
                ax[i].imshow(img, cmap='gray')
                ax[i].axis('off')
            plt.show()

    return classes


#####################################################################################################
############################### Functions for handling the MS dataset ###############################
#####################################################################################################

class MRIDataset(Dataset):
    """Custom Dataset class for loading MRI images and their labels."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_paths_and_labels(path, classes):
    """Collects image file paths and their corresponding labels from the dataset."""
    image_paths = []
    labels = []

    for label, cat in enumerate(classes):
        image_dir = os.path.join(path, cat)
        for fname in os.listdir(image_dir):
            img_path = os.path.join(image_dir, fname)
            image_paths.append(img_path)
            labels.append(label)

    print(f"get_paths_and_labels()>>> Total images: {len(image_paths)}")
    return image_paths, labels


def get_transforms(IMG_SIZE=(224, 224)):
    """
    Returns data augmentation and normalization transforms for training and testing datasets.
    Images resized to 224x224 to match input size of pre-trained models.
    """
    
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def _worker_init_fn(worker_id):
    """
    Seed each DataLoader worker for reproducibility.
    Called automatically by DataLoader.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def get_trainval_test_split(image_paths, labels, test_split=0.20, SEED=42):
    """
    Performs the outer stratified split to produce a held-out test set and a
    train+val pool. Called ONCE before any k-fold or training begins.
    The test set is never used during training, validation, or hyperparameter
    selection — only for final evaluation after all decisions are made.

    Args:
        image_paths: List of all image file paths
        labels:      List of corresponding labels (0 or 1)
        test_split:  Proportion to hold out as test set (default 0.20)
        SEED:        Random seed — fix this and never change it

    Returns:
        X_trainval, y_trainval  — pool used for all k-fold CV
        X_test,     y_test      — held-out test set, set aside until final eval
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        image_paths, labels,
        test_size=test_split,
        random_state=SEED,
        stratify=labels
    )

    total = len(image_paths)
    print(f"get_trainval_test_split()>>> Train+Val pool : {len(X_trainval)} "
          f"({len(X_trainval)/total:.1%})")
    print(f"get_trainval_test_split()>>> Held-out test  : {len(X_test)} "
          f"({len(X_test)/total:.1%})")

    tv_ms  = sum(y_trainval)
    tst_ms = sum(y_test)
    print(f"get_trainval_test_split()>>> TrainVal class ratio — "
          f"MS: {tv_ms}  Non-MS: {len(y_trainval)-tv_ms}")
    print(f"get_trainval_test_split()>>> Test     class ratio — "
          f"MS: {tst_ms}  Non-MS: {len(y_test)-tst_ms}")

    return X_trainval, y_trainval, X_test, y_test


def get_fold_loaders(X_trainval, y_trainval, fold_idx,
                     train_transform, test_transform,
                     n_splits=5, batch_size=32, SEED=42):
    """
    Returns train and validation DataLoaders for a single fold of stratified k-fold CV.
    Call this inside the fold loop — produces a different train/val split each time.

    Fold assignments are deterministic for a given (n_splits, SEED) pair,
    so results are fully reproducible.

    Args:
        X_trainval:       List of image paths in the train+val pool
        y_trainval:       Corresponding labels
        fold_idx:         Which fold to use as validation (0-indexed, 0 to n_splits-1)
        train_transform:  Augmentation transforms for training split
        test_transform:   No-augmentation transforms for validation split
        n_splits:         Number of folds (3 for grid search, 5 for final eval)
        batch_size:       Batch size for both loaders
        SEED:             Must match the seed used in get_trainval_test_split()

    Returns:
        train_loader, val_loader
    """
    X_arr = np.array(X_trainval)
    y_arr = np.array(y_trainval)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_arr, y_arr))

    train_idx, val_idx = splits[fold_idx]

    X_train, y_train = X_arr[train_idx].tolist(), y_arr[train_idx].tolist()
    X_val,   y_val   = X_arr[val_idx].tolist(),   y_arr[val_idx].tolist()

    print(f"get_fold_loaders()>>> Fold {fold_idx+1}/{n_splits} — "
          f"Train: {len(X_train)},  Val: {len(X_val)}")

    train_dataset = MRIDataset(X_train, y_train, transform=train_transform)
    val_dataset   = MRIDataset(X_val,   y_val,   transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, worker_init_fn=_worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, worker_init_fn=_worker_init_fn
    )

    return train_loader, val_loader


def get_test_loader(X_test, y_test, test_transform, batch_size=32):
    """
    Wraps the held-out test set in a DataLoader for final evaluation.
    Only call this after all training and model selection is complete.

    Args:
        X_test:          List of test image paths (from get_trainval_test_split)
        y_test:          Corresponding labels
        test_transform:  No-augmentation transforms
        batch_size:      Batch size

    Returns:
        test_loader
    """
    test_dataset = MRIDataset(X_test, y_test, transform=test_transform)
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, worker_init_fn=_worker_init_fn
    )
    print(f"get_test_loader()>>> Test loader ready — {len(X_test)} samples")
    return test_loader