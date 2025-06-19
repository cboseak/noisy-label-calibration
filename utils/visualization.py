"""
Visualization utilities for analyzing experimental results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_accuracy_vs_noise(results, save_path=None):
    """
    Plot test accuracy vs noise level for different noise types.
    
    Args:
        results: Dictionary with noise types as keys and lists of (noise_level, accuracy) tuples as values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^']
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (noise_type, data) in enumerate(results.items()):
        noise_levels = [x[0] for x in data]
        accuracies = [x[1] for x in data]
        
        plt.plot(noise_levels, accuracies, marker=markers[i % len(markers)], 
                 linestyle='-', markersize=8, color=colors[i % len(colors)],
                 label=f"{noise_type.capitalize()} Noise")
    
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title('Test Accuracy vs Noise Level', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_ece_vs_noise(results, save_path=None):
    """
    Plot Expected Calibration Error (ECE) vs noise level for different noise types.
    
    Args:
        results: Dictionary with noise types as keys and lists of (noise_level, ece) tuples as values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^']
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (noise_type, data) in enumerate(results.items()):
        noise_levels = [x[0] for x in data]
        ece_values = [x[1] for x in data]
        
        plt.plot(noise_levels, ece_values, marker=markers[i % len(markers)], 
                 linestyle='-', markersize=8, color=colors[i % len(colors)],
                 label=f"{noise_type.capitalize()} Noise")
    
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Expected Calibration Error (ECE)', fontsize=14)
    plt.title('Model Calibration Error vs Noise Level', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_entropy_vs_noise(results, save_path=None):
    """
    Plot prediction entropy vs noise level for different noise types.
    
    Args:
        results: Dictionary with noise types as keys and lists of (noise_level, entropy) tuples as values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^']
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (noise_type, data) in enumerate(results.items()):
        noise_levels = [x[0] for x in data]
        entropy_values = [x[1] for x in data]
        
        plt.plot(noise_levels, entropy_values, marker=markers[i % len(markers)], 
                 linestyle='-', markersize=8, color=colors[i % len(colors)],
                 label=f"{noise_type.capitalize()} Noise")
    
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Prediction Entropy', fontsize=14)
    plt.title('Model Prediction Entropy vs Noise Level', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_reliability_diagram(accuracies, confidences, bin_counts, title, save_path=None):
    """
    Plot reliability diagram.
    
    Args:
        accuracies: Accuracy in each bin
        confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
        title: Plot title
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[4, 1])
    
    # Main plot - reliability diagram
    ax0 = plt.subplot(gs[0, 0])
    
    # Plot gap between accuracy and confidence
    gap = np.abs(accuracies - confidences)
    bar_positions = np.linspace(0.0, 1.0, len(accuracies), endpoint=False) + 1.0 / (2 * len(accuracies))
    bars = ax0.bar(bar_positions, gap, width=1.0 / len(accuracies), 
                   alpha=0.3, color='red', label='Gap')
    
    # Perfect calibration line
    ax0.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    # Accuracy vs confidence
    ax0.plot(confidences, accuracies, 'o-', color='blue', markersize=8, label='Model')
    
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel('Confidence', fontsize=14)
    ax0.set_ylabel('Accuracy', fontsize=14)
    ax0.set_title(title, fontsize=16)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=12)
    ax0.set_aspect('equal')
    
    # Histogram of sample counts in each bin
    ax1 = plt.subplot(gs[1, 0])
    ax1.bar(bar_positions, bin_counts, width=1.0 / len(accuracies), color='blue', alpha=0.6)
    ax1.set_xlabel('Confidence', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_per_class_accuracy(per_class_acc, class_names, title, save_path=None):
    """
    Plot per-class accuracy.
    
    Args:
        per_class_acc: Accuracy for each class
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(per_class_acc)), per_class_acc * 100, color='skyblue')
    
    # Add values on top of each bar
    for i, v in enumerate(per_class_acc):
        plt.text(i, v * 100 + 2, f"{v*100:.1f}%", ha='center', fontsize=10)
    
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(range(len(class_names)), class_names, rotation=45, fontsize=10)
    plt.ylim(0, 110)  # Leave room for the percentage text
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_training_progress(history, save_path=None):
    """
    Plot training progress metrics.
    
    Args:
        history: Dictionary with lists of training metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplot for accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create subplot for loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create subplot for ECE
    plt.subplot(2, 2, 3)
    plt.plot(history['test_ece'], label='ECE')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ECE', fontsize=12)
    plt.title('Expected Calibration Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create subplot for entropy
    plt.subplot(2, 2, 4)
    plt.plot(history['test_entropy'], label='Entropy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.title('Prediction Entropy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()