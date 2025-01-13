"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from collections import defaultdict

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        train_transform: transforms.Compose, 
        test_transform: transforms.Compose,
        batch_size: int, 
        num_train_samples: int = None, 
        num_test_samples: int = None,
        num_workers: int=NUM_WORKERS
    ):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        train_transform: torchvision transforms to perform on training data.
        test_transform: torchvision transforms to perform on test data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_train_samples: Number of samples to include in the training dataset (None for all samples).
        num_test_samples: Number of samples to include in the test dataset (None for all samples).
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    # Resample training data if num_train_samples is specified
    if num_train_samples is not None:
        if num_train_samples > len(train_data):
            # Oversample by repeating indices
            indices = random.choices(range(len(train_data)), k=num_train_samples)
        else:
            # Undersample by selecting a subset of indices
            indices = random.sample(range(len(train_data)), k=num_train_samples)
        train_data = Subset(train_data, indices)

    # Resample testing data if num_test_samples is specified
    if num_test_samples is not None:
        if num_test_samples > len(test_data):
            # Oversample by repeating indices
            indices = random.choices(range(len(test_data)), k=num_test_samples)
        else:
            # Undersample by selecting a subset of indices
            indices = random.sample(range(len(test_data)), k=num_test_samples)
        test_data = Subset(test_data, indices)

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, #enables fast data transfre to CUDA-enable GPU
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True, #enables fast data transfre to CUDA-enable GPU
    )

    return train_dataloader, test_dataloader, class_names


def create_dataloaders_with_resampling(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    train_minority_class_samples: int = None,
    train_majority_class_samples: int = None,
    test_minority_class_samples: int = None,
    test_majority_class_samples: int = None,
    num_workers: int = 0
):
    """
    Creates training and testing DataLoaders with optional class balancing. It works for binary classification

    Args:
        train_dir (str): Path to the training directory.
        test_dir (str): Path to the testing directory.
        train_transform (transforms.Compose): Transformations for training data.
        test_transform (transforms.Compose): Transformations for test data.
        batch_size (int): Number of samples per batch in DataLoaders.
        train_minority_class_samples (int, optional): Target number of samples for minority class in training data.
        train_majority_class_samples (int, optional): Target number of samples for majority class in training data.
        test_minority_class_samples (int, optional): Target number of samples for minority class in testing data.
        test_majority_class_samples (int, optional): Target number of samples for majority class in testing data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: train_dataloader, test_dataloader, class_names
    """
    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    def resample_data(data, minority_samples, majority_samples):
        """Resamples the dataset to balance classes."""
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(data):
            class_indices[label].append(idx)
        
        sampled_indices = []
        for label, indices in class_indices.items():
            if majority_samples and len(indices) > majority_samples:
                sampled_indices.extend(np.random.choice(indices, majority_samples, replace=False))
            elif minority_samples and len(indices) < minority_samples:
                sampled_indices.extend(np.random.choice(indices, minority_samples, replace=True))
            else:
                sampled_indices.extend(indices)
        
        return torch.utils.data.Subset(data, sampled_indices)

    # Resample train dataset
    if train_minority_class_samples or train_majority_class_samples:
        balanced_train_data = resample_data(
            train_data,
            minority_samples=train_minority_class_samples,
            majority_samples=train_majority_class_samples
        )
    else:
        balanced_train_data = train_data

    # Resample test dataset
    if test_minority_class_samples or test_majority_class_samples:
        balanced_test_data = resample_data(
            test_data,
            minority_samples=test_minority_class_samples,
            majority_samples=test_majority_class_samples
        )
    else:
        balanced_test_data = test_data

    # Create DataLoaders
    train_dataloader = DataLoader(
        balanced_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        balanced_test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


def create_dataloaders_for_vit(
        vit_model: str="bitbase16",
        train_dir: str="./",
        test_dir: str="./",
        batch_size: int=64,
        num_train_samples: int = None, 
        num_test_samples: int = None,
        aug: bool=True,
        num_workers: int=os.cpu_count()
        ):
    
    """
    Creates data loaders for the training and test datasets to be used to traing visiton transformers.

    Args:
        vit_model (str): The name of the ViT model to use. Default is "bitbase16".
            -bitbase16: ViT-Base/16-224
            -bitbase16_2: ViT-Base/16-384
            -bitlarge16: ViT-Large/16-224
            -bitlarge32: ViT-Large/32-224
        train_dir (str): The path to the training dataset directory. Default is TRAIN_DIR.
        test_dir (str): The path to the test dataset directory. Default is TEST_DIR.
        batch_size (int): The batch size for the data loaders. Default is BATCH_SIZE.
        num_train_samples: Number of samples to include in the training dataset (None for all samples).
        num_test_samples: Number of samples to include in the test dataset (None for all samples).
        aug (bool): Whether to apply data augmentation or not. Default is True.
        display_imgs (bool): Whether to display sample images or not. Default is True.

    Returns:
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
        class_names (list): A list of class names.
    """

    IMG_SIZE = 224
    IMG_SIZE_2 = 384
    IMG_SIZE_3 = 518

    # Manual transforms for the training dataset
    manual_transforms = v2.Compose([           
        v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),    
    ])

    # ViT-Base/16 transforms
    if vit_model == "vitbase16":

        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitb = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((256, 256)),
                v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitb = v2.Compose([
                v2.Resize((256, 256)),
                v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitb = v2.Compose([    
            v2.Resize((256, 256)),
            v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Base
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitb,
            test_transform=manual_transforms_test_vitb,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
            )
    
    if vit_model == "vitbase16_2":

        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitb = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((IMG_SIZE_2, IMG_SIZE_2)),
                v2.CenterCrop((IMG_SIZE_2, IMG_SIZE_2)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitb = v2.Compose([
                v2.Resize((IMG_SIZE_2, IMG_SIZE_2)),
                v2.CenterCrop((IMG_SIZE_2, IMG_SIZE_2)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitb = v2.Compose([    
            v2.Resize((IMG_SIZE_2, IMG_SIZE_2)),
            v2.CenterCrop((IMG_SIZE_2, IMG_SIZE_2)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Base
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitb,
            test_transform=manual_transforms_test_vitb,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
            )

    # ViT-Large/16 transforms
    elif vit_model == "vitlarge16":

        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitl = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((242, 242)),
                v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitl = v2.Compose([
                v2.Resize((242, 242)),
                v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitl = v2.Compose([    
            v2.Resize((242, 242)),
            v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Large/16
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitl,
            test_transform=manual_transforms_test_vitl,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
        )

    # ViT-Large/32 transforms
    elif vit_model == "vitlarge32":
        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitl = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((256, 256)),
                v2.RandomCrop((IMG_SIZE, IMG_SIZE)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitl = v2.Compose([
                v2.Resize((256, 256)),
                v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitl = v2.Compose([    
            v2.Resize((256, 256)),
            v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Large/32
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitl,
            test_transform=manual_transforms_test_vitl,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
        )
    # Vit-Huge/14 transforms
    else:
        # Manual transforms for the training dataset
        if aug:
            manual_transforms_train_vitl = v2.Compose([    
                v2.TrivialAugmentWide(),
                v2.Resize((518, 518)),
                v2.RandomCrop((IMG_SIZE_3, IMG_SIZE_3)),    
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
            ])
        else:
            manual_transforms_train_vitl = v2.Compose([
                v2.Resize((518, 518)),
                v2.CenterCrop((IMG_SIZE_3, IMG_SIZE_3)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Manual transforms for the test dataset
        manual_transforms_test_vitl = v2.Compose([    
            v2.Resize((518, 518)),
            v2.CenterCrop((IMG_SIZE_3, IMG_SIZE_3)),    
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
        ])

        # Create data loaders for ViT-Large/32
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=manual_transforms_train_vitl,
            test_transform=manual_transforms_test_vitl,
            batch_size=batch_size,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            num_workers=num_workers
        )

    return train_dataloader, test_dataloader, class_names
