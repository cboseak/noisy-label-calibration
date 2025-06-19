"""
Dataset module: Handles CIFAR-10 dataset loading and label noise injection.
"""
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import copy

# CIFAR-10 classes
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10_Noisy(Dataset):
    """
    CIFAR-10 dataset with injected label noise.
    """
    def __init__(self, root, train=True, transform=None, download=False, 
                 noise_type='symmetric', noise_level=0.2, random_state=42):
        """
        Args:
            root: Root directory for the dataset
            train: Whether to use training set (True) or test set (False)
            transform: Image transformations to apply
            download: Whether to download the dataset if not found
            noise_type: Type of noise to inject ('symmetric', 'asymmetric', or 'none')
            noise_level: Proportion of labels to corrupt (between 0 and 1)
            random_state: Random seed for reproducibility
        """
        self.dataset = datasets.CIFAR10(root=root, train=train, transform=transform, 
                                       download=download)
        self.train = train
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.num_classes = 10
        
        if self.train and noise_type != 'none' and noise_level > 0:
            # Set random seed for reproducibility
            np.random.seed(random_state)
            
            # Get clean labels before corruption
            self.clean_labels = copy.deepcopy(self.dataset.targets)
            
            # Apply label noise
            if noise_type == 'symmetric':
                self.apply_symmetric_noise()
            elif noise_type == 'asymmetric':
                self.apply_asymmetric_noise()
        else:
            # For test set or when no noise is applied
            self.clean_labels = self.dataset.targets
    
    def apply_symmetric_noise(self):
        """
        Apply symmetric label noise - randomly flip labels to any of the other classes
        with uniform probability.
        """
        # Get labels and convert to numpy array if needed
        if isinstance(self.dataset.targets, list):
            labels = np.array(self.dataset.targets)
        else:
            labels = self.dataset.targets.numpy() if isinstance(self.dataset.targets, torch.Tensor) else self.dataset.targets
        
        # Determine which labels to corrupt
        mask = np.random.rand(len(labels)) <= self.noise_level
        num_to_corrupt = mask.sum()
        
        # For each label to corrupt, randomly choose a different class
        for idx in np.where(mask)[0]:
            # Select a random class different from the original class
            possible_classes = list(range(self.num_classes))
            possible_classes.remove(labels[idx])
            new_label = np.random.choice(possible_classes)
            labels[idx] = new_label
        
        # Update dataset labels
        if isinstance(self.dataset.targets, list):
            self.dataset.targets = labels.tolist()
        else:
            self.dataset.targets = torch.tensor(labels) if isinstance(self.dataset.targets, torch.Tensor) else labels
        
        return mask

    def apply_asymmetric_noise(self):
        """
        Apply asymmetric label noise - flip labels based on class similarity.
        For CIFAR-10, we use the following mapping for similar classes:
        - plane -> bird
        - car -> truck
        - bird -> plane
        - cat -> dog
        - deer -> horse
        - dog -> cat
        - frog -> (random)
        - horse -> deer
        - ship -> (random)
        - truck -> car
        """
        # Define similar class mapping
        similar_classes = {
            0: 2,  # plane -> bird
            1: 9,  # car -> truck
            2: 0,  # bird -> plane
            3: 5,  # cat -> dog
            4: 7,  # deer -> horse
            5: 3,  # dog -> cat
            6: np.random.randint(0, 10),  # frog -> random
            7: 4,  # horse -> deer
            8: np.random.randint(0, 10),  # ship -> random
            9: 1,  # truck -> car
        }
        
        # Get labels and convert to numpy array if needed
        if isinstance(self.dataset.targets, list):
            labels = np.array(self.dataset.targets)
        else:
            labels = self.dataset.targets.numpy() if isinstance(self.dataset.targets, torch.Tensor) else self.dataset.targets
        
        # Determine which labels to corrupt
        mask = np.random.rand(len(labels)) <= self.noise_level
        
        # Apply asymmetric changes
        for idx in np.where(mask)[0]:
            orig_label = labels[idx]
            labels[idx] = similar_classes[orig_label]
        
        # Update dataset labels
        if isinstance(self.dataset.targets, list):
            self.dataset.targets = labels.tolist()
        else:
            self.dataset.targets = torch.tensor(labels) if isinstance(self.dataset.targets, torch.Tensor) else labels
        
        return mask
    
    def __getitem__(self, index):
        """Get item with both noisy and clean labels."""
        img, noisy_label = self.dataset[index]
        clean_label = self.clean_labels[index]
        return img, noisy_label, clean_label
    
    def __len__(self):
        return len(self.dataset)

def get_data_transforms():
    """
    Define data transformations for CIFAR-10.
    """
    # Training data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Test data transforms - only normalization, no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    return train_transform, test_transform

def get_cifar10_loaders(args):
    """
    Create data loaders for CIFAR-10 with noise injection.
    
    Args:
        args: Configuration arguments
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, test_transform = get_data_transforms()
    
    # Load training data with noise
    train_dataset = CIFAR10_Noisy(
        root=args.data_path,
        train=True,
        transform=train_transform,
        download=args.download,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        random_state=args.seed
    )
    
    # Load test data (always clean)
    test_dataset = CIFAR10_Noisy(
        root=args.data_path,
        train=False,
        transform=test_transform,
        download=args.download,
        noise_type='none',
        noise_level=0
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Test loader (no noise)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader