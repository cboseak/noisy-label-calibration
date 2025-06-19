"""
Main script for running label noise experiments.
"""
import os
import torch
import numpy as np
import json
import pickle
from configs.config import get_config
from utils.dataset import get_cifar10_loaders
from models.resnet import get_model
from utils.trainer import Trainer
import matplotlib.pyplot as plt
from utils.visualization import plot_accuracy_vs_noise, plot_ece_vs_noise, plot_entropy_vs_noise

def run_single_experiment(args):
    """
    Run a single experiment with given noise type and level.
    
    Args:
        args: Configuration arguments
        
    Returns:
        best_model: Best model
        metrics: Dictionary with metrics
    """
    print(f"\nRunning experiment with {args.noise_type} noise at level {args.noise_level}")
    
    # Define checkpoint path for this specific experiment
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    experiment_id = f"{args.noise_type}_{args.noise_level}"
    checkpoint_path = os.path.join(checkpoint_dir, f"experiment_{experiment_id}.pth")
    results_path = os.path.join(checkpoint_dir, f"results_{experiment_id}.pkl")
    
    # Check if experiment results already exist
    if os.path.exists(results_path):
        print(f"Found existing results for {experiment_id}. Loading...")
        with open(results_path, 'rb') as f:
            metrics = pickle.load(f)
        # Load the best model if needed
        if os.path.exists(checkpoint_path):
            model = get_model(args)
            model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
            print(f"Loaded best model with accuracy: {metrics['accuracy']:.4f}")
            return model, metrics
    
    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(args)
    
    # Get model
    model = get_model(args)
    
    # Train model
    trainer = Trainer(model, train_loader, test_loader, args, checkpoint_path=checkpoint_path)
    best_model, history = trainer.train()
    
    # Get final logits and targets
    logits, targets = trainer.get_final_logits_and_targets()
    
    # Create metrics
    metrics = {
        'accuracy': history['test_acc'][-1],
        'ece': history['test_ece'][-1],
        'entropy': history['test_entropy'][-1],
        'history': history,
        'logits': logits.numpy(),
        'targets': targets.numpy()
    }
    
    # Save final results
    with open(results_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    return best_model, metrics

def save_experiment_progress(args, results, accuracy_results, ece_results, entropy_results):
    """Save experiment progress to allow resuming later."""
    progress_dir = os.path.join(args.save_dir, 'progress')
    os.makedirs(progress_dir, exist_ok=True)
    progress_path = os.path.join(progress_dir, 'experiment_progress.pkl')
    
    progress = {
        'results': results,
        'accuracy_results': accuracy_results,
        'ece_results': ece_results, 
        'entropy_results': entropy_results
    }
    
    with open(progress_path, 'wb') as f:
        pickle.dump(progress, f)
    
    # Also save a more readable JSON version with basic info
    json_progress = {noise_type: [] for noise_type in results.keys()}
    for noise_type, exps in results.items():
        for exp in exps:
            json_progress[noise_type].append({
                'noise_level': float(exp['noise_level']),
                'accuracy': float(exp['accuracy']),
                'ece': float(exp['ece']),
                'entropy': float(exp['entropy'])
            })
    
    with open(os.path.join(progress_dir, 'experiment_progress.json'), 'w') as f:
        json.dump(json_progress, f, indent=2)
        
    return progress_path

def load_experiment_progress(args):
    """Load previously saved experiment progress."""
    progress_path = os.path.join(args.save_dir, 'progress', 'experiment_progress.pkl')
    if os.path.exists(progress_path):
        print(f"Found existing experiment progress. Loading from {progress_path}...")
        with open(progress_path, 'rb') as f:
            progress = pickle.load(f)
        return progress['results'], progress['accuracy_results'], progress['ece_results'], progress['entropy_results']
    
    # Initialize empty data structures if no progress found
    noise_types = ['symmetric', 'asymmetric', 'none']
    results = {}
    accuracy_results = {noise_type: [] for noise_type in noise_types}
    ece_results = {noise_type: [] for noise_type in noise_types}
    entropy_results = {noise_type: [] for noise_type in noise_types}
    
    return results, accuracy_results, ece_results, entropy_results

def main():
    """
    Run experiments for different noise types and levels.
    """
    # Get default configuration
    args = get_config()
    
    # Define noise types and levels to test
    noise_types = ['symmetric', 'asymmetric', 'none']
    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6]
    
    # Create progress directory if it doesn't exist
    os.makedirs(os.path.join(args.save_dir, 'progress'), exist_ok=True)
    
    # Try to load previous experiment progress
    results, accuracy_results, ece_results, entropy_results = load_experiment_progress(args)
    
    # Create an experiment tracker file to monitor which experiments have been completed
    tracker_path = os.path.join(args.save_dir, 'progress', 'experiment_tracker.json')
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            completed_experiments = json.load(f)
    else:
        completed_experiments = {}
    
    try:
        # Run experiments for each noise type and level
        for noise_type in noise_types:
            if noise_type not in results:
                results[noise_type] = []
                
            for noise_level in noise_levels:
                # Skip noise_level 0.0 for none
                if noise_type == 'none' and noise_level > 0.0:
                    continue
                
                # Check if this experiment has already been completed
                experiment_id = f"{noise_type}_{noise_level}"
                if experiment_id in completed_experiments:
                    print(f"Experiment {experiment_id} already completed. Skipping...")
                    continue
                    
                # Update args
                args.noise_type = noise_type
                args.noise_level = noise_level
                
                # Run experiment
                _, metrics = run_single_experiment(args)
                
                # Store results
                results[noise_type].append({
                    'noise_level': noise_level,
                    'accuracy': metrics['accuracy'],
                    'ece': metrics['ece'],
                    'entropy': metrics['entropy']
                })
                
                # Append to plot data
                accuracy_results[noise_type].append((noise_level, metrics['accuracy']))
                ece_results[noise_type].append((noise_level, metrics['ece']))
                entropy_results[noise_type].append((noise_level, metrics['entropy']))
                
                # Mark experiment as completed
                completed_experiments[experiment_id] = True
                with open(tracker_path, 'w') as f:
                    json.dump(completed_experiments, f, indent=2)
                
                # Save progress after each experiment
                progress_path = save_experiment_progress(args, results, accuracy_results, ece_results, entropy_results)
                print(f"Experiment progress saved to {progress_path}")
                
                # Create interim comparison plots
                create_comparison_plots(args, accuracy_results, ece_results, entropy_results)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress has been saved and can be resumed later.")
    except Exception as e:
        print(f"\nError during experiment: {e}")
        print("Progress has been saved and can be resumed by running the script again.")
    
    # Create final comparison plots
    create_comparison_plots(args, accuracy_results, ece_results, entropy_results)
    
    # Print summary of results
    print("\nExperiment Results Summary:")
    print("--------------------------")
    
    for noise_type in results.keys():
        print(f"\n{noise_type.capitalize()} Noise:")
        for result in sorted(results[noise_type], key=lambda x: x['noise_level']):
            print(f"  Noise Level: {result['noise_level']}, "
                  f"Accuracy: {result['accuracy']:.4f}, "
                  f"ECE: {result['ece']:.4f}, "
                  f"Entropy: {result['entropy']:.4f}")
    
    print("\nExperiments completed. Results saved to", args.save_dir)

def create_comparison_plots(args, accuracy_results, ece_results, entropy_results):
    """Create comparison plots from current results."""
    os.makedirs(os.path.join(args.save_dir, 'comparison'), exist_ok=True)
    
    # Plot accuracy vs noise level
    acc_plot_path = os.path.join(args.save_dir, 'comparison', 'accuracy_vs_noise.png')
    plot_accuracy_vs_noise(accuracy_results, acc_plot_path)
    
    # Plot ECE vs noise level
    ece_plot_path = os.path.join(args.save_dir, 'comparison', 'ece_vs_noise.png')
    plot_ece_vs_noise(ece_results, ece_plot_path)
    
    # Plot entropy vs noise level
    entropy_plot_path = os.path.join(args.save_dir, 'comparison', 'entropy_vs_noise.png')
    plot_entropy_vs_noise(entropy_results, entropy_plot_path)

if __name__ == "__main__":
    main()