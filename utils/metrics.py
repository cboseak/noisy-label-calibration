"""
Metrics for evaluating model calibration and performance.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import calibration_error

def compute_accuracy(outputs, targets):
    """
    Compute classification accuracy.
    
    Args:
        outputs: Model output logits of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size)
        
    Returns:
        accuracy: Classification accuracy
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)

def compute_ece(logits, targets, n_bins=15):
    """
    Compute Expected Calibration Error using torchmetrics.
    
    Args:
        logits: Model output logits of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size)
        n_bins: Number of bins for histogram
        
    Returns:
        ece: Expected Calibration Error
    """
    probabilities = F.softmax(logits, dim=1)
    return calibration_error(probabilities, targets, n_bins=n_bins, task="multiclass", num_classes=10).item()

def compute_prediction_entropy(logits):
    """
    Compute Shannon entropy of predicted probability distribution.
    Higher entropy indicates higher uncertainty.
    
    Args:
        logits: Model output logits of shape (batch_size, num_classes)
        
    Returns:
        entropy: Average entropy of predictions
    """
    probabilities = F.softmax(logits, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy.mean().item()

def get_confidence_and_correctness(logits, targets):
    """
    Get the confidence scores and correctness for each prediction.
    
    Args:
        logits: Model output logits of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size)
        
    Returns:
        confidences: Confidence scores for each prediction
        correctness: Binary indicator if prediction was correct
    """
    probabilities = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probabilities, dim=1)
    correctness = predictions.eq(targets)
    
    return confidences.cpu().numpy(), correctness.cpu().numpy()

def get_calibration_data(logits, targets, n_bins=15):
    """
    Get data for reliability diagrams.
    
    Args:
        logits: Model output logits
        targets: Ground truth labels
        n_bins: Number of bins for histogram
        
    Returns:
        accuracies: Accuracy in each bin
        confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
    """
    confidences, correctness = get_confidence_and_correctness(logits, targets)
    
    # Create bins and compute statistics
    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[1:-1], right=True)
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Compute accuracy and average confidence for each bin
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        if np.sum(bin_mask) > 0:
            bin_accuracies[i] = np.mean(correctness[bin_mask])
            bin_confidences[i] = np.mean(confidences[bin_mask])
            bin_counts[i] = np.sum(bin_mask)
    
    return bin_accuracies, bin_confidences, bin_counts

def get_per_class_accuracy(logits, targets, num_classes=10):
    """
    Calculate per-class accuracy.
    
    Args:
        logits: Model output logits
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        per_class_acc: Accuracy for each class
    """
    _, predicted = torch.max(logits, 1)
    per_class_acc = torch.zeros(num_classes)
    
    for i in range(num_classes):
        class_mask = (targets == i)
        if class_mask.sum() > 0:
            per_class_acc[i] = (predicted[class_mask] == targets[class_mask]).float().mean()
    
    return per_class_acc.cpu().numpy()