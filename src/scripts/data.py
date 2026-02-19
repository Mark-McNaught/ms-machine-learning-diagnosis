import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


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

def get_classes(path, categories, axial=True, saggital=False, visualise=True, num_samples=3):
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


# Worker initialisation for reproducibility 
def _worker_init_fn(worker_id):
    """
    Seed each DataLoader worker for reproducibility.
    Called automatically by DataLoader.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def get_data_loaders(image_paths, labels, train_transform, test_transform, 
                      val_split=0.15, test_split=0.20, batch_size=32, SEED=42):
    """
    Splits the dataset into train/validation/test sets with stratification.
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (0 or 1)
        train_transform: Augmentation transforms for training
        test_transform: Transforms for validation and test (no augmentation)
        val_split: Proportion for validation set (default 0.15 = 15%)
        test_split: Proportion for test set (default 0.20 = 20%)
        batch_size: Batch size for dataloaders
        SEED: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, (X_test, y_test)
    
    Default split: 65% train, 15% val, 20% test
    """
    
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        image_paths, labels, 
        test_size=test_split, 
        random_state=SEED, 
        stratify=labels
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=val_size_adjusted, 
        random_state=SEED, 
        stratify=y_trainval
    )
    
    print(f"get_data_loaders()>>> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"get_data_loaders()>>> Proportions: Train {len(X_train)/len(image_paths):.1%}, "
          f"Val {len(X_val)/len(image_paths):.1%}, Test {len(X_test)/len(image_paths):.1%}")
    
    # Create datasets
    train_dataset = MRIDataset(X_train, y_train, transform=train_transform)
    val_dataset = MRIDataset(X_val, y_val, transform=test_transform)
    test_dataset = MRIDataset(X_test, y_test, transform=test_transform)
    
    # Create dataloaders with worker seeding
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=_worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=_worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=_worker_init_fn
    )
    
    return train_loader, val_loader, test_loader, (X_test, y_test)