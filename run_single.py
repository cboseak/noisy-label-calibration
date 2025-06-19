"""
Script for running a single label noise experiment with specific parameters.
"""
import os
import argparse
import torch
import numpy as np
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, set_seed
from utils.dataset import get_cifar10_loaders
from models.resnet import get_model
from utils.trainer import Trainer
from utils.metrics import compute_ece, compute_prediction_entropy

def parse_args():
    parser = argparse.ArgumentParser(description='Run a single label noise experiment')
    
    # Override config parameters
    parser.add_argument('--noise-type', default='symmetric', type=str, 
                        choices=['symmetric', 'asymmetric', 'none'],
                        help='Type of noise: symmetric, asymmetric, or none')
    parser.add_argument('--noise-level', default=0.2, type=float,
                        help='Level of noise to inject (0.0, 0.1, 0.2, 0.4, 0.6)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint if available')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints (defaults to args.save_dir/checkpoints)')
    
    # Parse args and get default config
    cli_args = parser.parse_args()
    config_args = get_config()
    
    # Override default config with CLI arguments
    for arg in vars(cli_args):
        setattr(config_args, arg, getattr(cli_args, arg))
    
    # Reset random seed
    set_seed(config_args.seed)
    
    return config_args

def run_experiment(args):
    """
    Run a single experiment with given config.
    """
    print("\nRunning experiment with the following configuration:")
    print(f"  - Noise type: {args.noise_type}")
    print(f"  - Noise level: {args.noise_level}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Optimizer: {args.optimizer}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Device: {args.device}")
    print(f"  - Seed: {args.seed}")
    
    # Setup checkpoint directory and path
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create experiment ID and checkpoint path
    experiment_id = f"{args.noise_type}_{args.noise_level}"
    checkpoint_path = os.path.join(args.checkpoint_dir, f"experiment_{experiment_id}.pth")
    
    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(args)
    
    # Get model
    model = get_model(args)
    print(f"Model: {args.model}")
    
    # Train model with checkpoint support
    trainer = Trainer(model, train_loader, test_loader, args, checkpoint_path=checkpoint_path)
    best_model, history = trainer.train()
    
    # Get final test metrics
    logits, targets = trainer.get_final_logits_and_targets()
    final_acc = history['test_acc'][-1]
    final_ece = history['test_ece'][-1]
    final_entropy = history['test_entropy'][-1]
    
    print("\nFinal Results:")
    print(f"  - Test accuracy: {final_acc:.4f}")
    print(f"  - Expected Calibration Error: {final_ece:.4f}")
    print(f"  - Prediction entropy: {final_entropy:.4f}")
    
    # Create a summary results file
    summary_path = os.path.join(args.save_dir, 'experiment_summary.json')
    summary = {
        'noise_type': args.noise_type,
        'noise_level': args.noise_level,
        'accuracy': float(final_acc),
        'ece': float(final_ece),
        'entropy': float(final_entropy),
        'epochs_completed': len(history['train_loss']),
        'best_epoch': history['test_acc'].index(max(history['test_acc'])),
        'training_time': sum(history['epoch_time'])
    }
    
    # Add to summary file or create new one
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            all_summaries = json.load(f)
    else:
        all_summaries = {}
    
    all_summaries[experiment_id] = summary
    
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    # Return best model and metrics
    metrics = {
        'accuracy': final_acc,
        'ece': final_ece, 
        'entropy': final_entropy,
        'history': history
    }
    
    return best_model, metrics

if __name__ == "__main__":
    args = parse_args()
    model, metrics = run_experiment(args)
    print("\nExperiment completed. Results saved to", args.save_dir)