"""
Configuration file for label noise experiments.
"""
import argparse
import os
import random
import numpy as np
import torch

def get_config():
    parser = argparse.ArgumentParser(description='Label Noise Experiments')
    
    # Data parameters
    parser.add_argument('--data-path', default='./data', type=str, 
                        help='Path to the CIFAR-10 dataset')
    parser.add_argument('--download', default=True, type=bool,
                        help='Download CIFAR-10 if not found')
    
    # Noise parameters
    parser.add_argument('--noise-type', default='symmetric', type=str, choices=['symmetric', 'asymmetric', 'none'],
                        help='Type of label noise: symmetric, asymmetric, or none')
    parser.add_argument('--noise-level', default=0.2, type=float, choices=[0.0, 0.1, 0.2, 0.4, 0.6],
                        help='Level of noise to inject (0.0, 0.1, 0.2, 0.4, 0.6)')
    
    # Training parameters
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help='Testing batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='Weight decay (L2 penalty)')
    
    # Model parameters
    parser.add_argument('--model', default='resnet18', type=str,
                        help='Model architecture')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='Use pre-trained model')
    
    # Output parameters
    parser.add_argument('--save-dir', default='./results', type=str,
                        help='Directory to save results')
    parser.add_argument('--checkpoint-dir', default='./results/checkpoints', type=str,
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='How many batches to wait before logging status')
    
    # Reproducibility
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    # GPU/Device parameters
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu-ids', default=None, type=str,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). Default: use all available GPUs')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--cuda-deterministic', action='store_true',
                        help='Set CUDA to deterministic mode (might reduce performance)')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark for faster training on consistent input sizes')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    
    # Configure GPU settings
    configure_gpu(args)
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    return args

def configure_gpu(args):
    """Configure GPU settings based on arguments."""
    if torch.cuda.is_available():
        # Display information about available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set specific GPU IDs if requested
        if args.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            print(f"Using GPUs: {args.gpu_ids}")
        else:
            print("Using all available GPUs")
        
        # Configure cuDNN
        if args.cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("CUDA running in deterministic mode")
        elif args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("Using cuDNN benchmark for faster training")
    else:
        print("No GPU available, using CPU.")

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False