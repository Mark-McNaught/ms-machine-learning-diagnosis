import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    print("get_dataset()>>> Available categories:", categories)

    return path, categories

def get_classes(path, categories, axial=True, saggital=False, visualise=True, num_samples=3):
    """
    Returns selected classes from the dataset, and if specified, visualises sample images.
    Axial and Saggital flags to determine what parts of the dataset to include.
    """
    classes = []
    if axial:
        classes.extend([categories[0], categories[2]]) # Control & MS Axial_crop
    if saggital:
        classes.extend([categories[1], categories[3]]) # Control & MS Saggital_crop
    
    if visualise:
        # Visualise random images from each category
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
    """
    Custom Dataset class for loading MRI images and their labels.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Converting to RGB to ensure compatibility with pre-trained models
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_paths_and_labels(path, classes):
    """Collects image file paths and their corresponding labels from the dataset."""

    image_paths = []
    labels = []

    # Iterate through classes and collect image paths and labels
    for label, cat in enumerate(classes):
        image_dir = os.path.join(path, cat)
        for fname in os.listdir(image_dir):
            img_path = os.path.join(image_dir, fname)
            image_paths.append(img_path)
            labels.append(label)

    print(f"get_paths_and_labels()>>> Total images: {len(image_paths)}")
    return image_paths, labels


def get_transforms(IMG_SIZE = (224, 224)):
    """
    Returns data augmentation and normalization transforms for training and testing datasets.
    Images resized to 224x224 to match input size of pre-trained models.
    """
    
    # Define transforms
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

def get_data_loaders(image_paths, labels, train_transform, test_transform, split=0.2, batch_size = 32, SEED=42):
    """
    Splits the dataset into training and testing sets, createing datasets and dataloaders.
    Default configuration uses 80-20 split, random seed of 42, and batch size of 32.
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=split, random_state=SEED, stratify=labels)
    print(f"get_data_loaders()>>> Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Create datasets
    train_dataset = MRIDataset(X_train, y_train, transform=train_transform)
    test_dataset = MRIDataset(X_test, y_test, transform=test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, (X_test, y_test)