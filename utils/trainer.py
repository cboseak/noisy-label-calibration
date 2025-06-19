"""
Training and evaluation utility functions.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import json
import pickle
from tqdm import tqdm

from utils.metrics import compute_accuracy, compute_ece, compute_prediction_entropy
from utils.metrics import get_calibration_data, get_per_class_accuracy
from utils.visualization import plot_reliability_diagram, plot_training_progress, plot_per_class_accuracy

class Trainer:
    def __init__(self, model, train_loader, test_loader, args, checkpoint_path=None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            args: Configuration arguments
            checkpoint_path: Path to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.device = torch.device(args.device)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(checkpoint_path) if checkpoint_path else args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create epoch checkpoint directory
        self.epoch_checkpoint_dir = os.path.join(
            self.checkpoint_dir,
            f'epochs_{args.noise_type}_{args.noise_level}'
        )
        os.makedirs(self.epoch_checkpoint_dir, exist_ok=True)
        
        # Set up loss function BEFORE setting up GPU
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up GPU usage and DataParallel if available
        self._setup_gpu()
        
        # Set up optimizer
        if args.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5, verbose=True
        )
        
        # Create history dictionary to track metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_ece': [],
            'test_entropy': [],
            'epoch_time': []
        }
        
        # Save best model state
        self.best_acc = 0.0
        self.best_model_state = None
        
        # Starting epoch (will be non-zero if resuming from checkpoint)
        self.start_epoch = 0
        
        # Try to load checkpoint if provided
        self.load_checkpoint()

    def _setup_gpu(self):
        """Set up GPU usage - always use only one GPU even if multiple are available."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            # Get first GPU index (0 by default, or first from gpu_ids if specified)
            gpu_idx = 0
            if self.args.gpu_ids:
                gpu_ids = [int(id) for id in self.args.gpu_ids.split(',')]
                if gpu_ids:
                    gpu_idx = gpu_ids[0]
            
            # Set CUDA visible devices to only show the selected GPU
            # This ensures only one GPU is used regardless of how many are available
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            # Get the device for the selected GPU
            self.device = torch.device(f'cuda:{0}')  # Always use cuda:0 as it's the only visible GPU now
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
            
            # Move model and criterion to GPU
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
        else:
            print("Using CPU for training")
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
    
    def load_checkpoint(self):
        """
        Try to load from checkpoint to resume training.
        """
        # Check if there's a checkpoint to resume from
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            
            # Load checkpoint with device specification
            map_location = self.device
            checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
            
            # Handle DataParallel models
            if isinstance(self.model, nn.DataParallel):
                # If saved model wasn't DataParallel, need to add 'module.' prefix
                if not list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                    new_state_dict = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
                    self.model.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If saved model was DataParallel but current isn't, need to remove 'module.' prefix
                if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                    self.model.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_acc = checkpoint.get('best_acc', 0.0)
            self.best_model_state = checkpoint['model_state_dict']
            
            # Try to load training history
            history_path = os.path.join(
                self.args.save_dir,
                f'noise_{self.args.noise_type}_{self.args.noise_level}',
                'history.npy'
            )
            if os.path.exists(history_path):
                try:
                    self.history = np.load(history_path, allow_pickle=True).item()
                    print(f"Loaded training history with {len(self.history['train_loss'])} epochs")
                    self.start_epoch = len(self.history['train_loss'])
                except Exception as e:
                    print(f"Error loading history: {e}")
            
            # Look for the latest epoch checkpoint
            if not self.start_epoch:
                epoch_checkpoints = [f for f in os.listdir(self.epoch_checkpoint_dir) 
                                    if f.startswith('epoch_') and f.endswith('.pth')]
                if epoch_checkpoints:
                    latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in epoch_checkpoints])
                    self.start_epoch = latest_epoch + 1
                    
                    # Load the latest epoch checkpoint
                    latest_path = os.path.join(self.epoch_checkpoint_dir, f'epoch_{latest_epoch}.pth')
                    latest_checkpoint = torch.load(latest_path, map_location=map_location)
                    
                    # Handle DataParallel models for the latest checkpoint
                    if isinstance(self.model, nn.DataParallel):
                        if not list(latest_checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                            new_state_dict = {'module.' + k: v for k, v in latest_checkpoint['model_state_dict'].items()}
                            self.model.load_state_dict(new_state_dict)
                        else:
                            self.model.load_state_dict(latest_checkpoint['model_state_dict'])
                    else:
                        if list(latest_checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                            new_state_dict = {k.replace('module.', ''): v for k, v in latest_checkpoint['model_state_dict'].items()}
                            self.model.load_state_dict(new_state_dict)
                        else:
                            self.model.load_state_dict(latest_checkpoint['model_state_dict'])
                    
                    self.optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
                    
                    # Load history from checkpoint
                    if 'history' in latest_checkpoint:
                        self.history = latest_checkpoint['history']
            
            print(f"Resuming training from epoch {self.start_epoch}")
            
            # Move optimizer state to the correct device after loading
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save checkpoint for the current epoch.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Extract model state dict, handling DataParallel
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_acc': self.best_acc,
        }
        
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(self.epoch_checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)
        
        # Save best checkpoint separately if this is the best model
        if is_best and self.checkpoint_path:
            torch.save(checkpoint, self.checkpoint_path)
            
        # Delete old checkpoints to save space (keep only the last 3)
        self._clean_old_checkpoints(epoch)
    
    def _clean_old_checkpoints(self, current_epoch, keep=3):
        """Clean up old checkpoints, keeping only the most recent ones."""
        epoch_checkpoints = [f for f in os.listdir(self.epoch_checkpoint_dir) 
                           if f.startswith('epoch_') and f.endswith('.pth')]
        
        if len(epoch_checkpoints) <= keep:
            return
            
        # Sort checkpoints by epoch number
        epochs = [int(f.split('_')[1].split('.')[0]) for f in epoch_checkpoints]
        epoch_files = sorted(zip(epochs, epoch_checkpoints), reverse=True)
        
        # Keep the current epoch and the most recent ones
        keep_epochs = set([current_epoch] + [e for e, _ in epoch_files[:keep]])
        
        # Remove old checkpoints
        for epoch, fname in epoch_files:
            if epoch not in keep_epochs:
                try:
                    os.remove(os.path.join(self.epoch_checkpoint_dir, fname))
                except:
                    pass
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for a progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (inputs, targets, _) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': train_loss/(batch_idx+1), 'Acc': 100.*correct/total})
            
            if batch_idx % self.args.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch+1} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}% | LR: {current_lr}')
                
            # Save batch-level checkpoint every 100 batches
            if (batch_idx + 1) % 100 == 0:
                batch_checkpoint = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                batch_checkpoint_path = os.path.join(
                    self.epoch_checkpoint_dir, 
                    f'epoch_{epoch}_batch_{batch_idx}.pth'
                )
                torch.save(batch_checkpoint, batch_checkpoint_path)
        
        return train_loss/len(self.train_loader), correct/total
    
    def test_epoch(self, epoch):
        """Evaluate on the test set."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store all outputs and targets for ECE and entropy calculation
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        test_acc = correct / total
        test_ece = compute_ece(all_outputs, all_targets)
        test_entropy = compute_prediction_entropy(all_outputs)
        
        # Get per-class accuracy
        per_class_acc = get_per_class_accuracy(all_outputs, all_targets)
        
        # Get calibration data for reliability diagrams
        bin_accuracies, bin_confidences, bin_counts = get_calibration_data(all_outputs, all_targets)
        
        # Print test metrics
        print(f'Test Epoch: {epoch+1} | Loss: {test_loss/len(self.test_loader):.4f} | '
              f'Acc: {100.*test_acc:.2f}% | ECE: {test_ece:.4f} | Entropy: {test_entropy:.4f}')
        
        return (test_loss/len(self.test_loader), test_acc, test_ece, test_entropy,
                bin_accuracies, bin_confidences, bin_counts, per_class_acc, all_outputs, all_targets)
    
    def train(self):
        """
        Run the full training loop.
        
        Returns:
            best_model: Best model based on validation accuracy
            history: Training history with metrics
        """
        print(f"Starting training with {self.args.noise_type} noise at level {self.args.noise_level}")
        
        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                start_time = time.time()
                
                # Train for one epoch
                train_loss, train_acc = self.train_epoch(epoch)
                
                # Test on validation set
                result = self.test_epoch(epoch)
                test_loss, test_acc = result[0], result[1]
                test_ece, test_entropy = result[2], result[3]
                bin_accuracies, bin_confidences, bin_counts = result[4], result[5], result[6]
                per_class_acc = result[7]
                
                # Update learning rate
                self.scheduler.step(test_loss)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
                self.history['test_ece'].append(test_ece)
                self.history['test_entropy'].append(test_entropy)
                self.history['epoch_time'].append(time.time() - start_time)
                
                # Save checkpoint after every epoch
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    self.best_model_state = self.model.state_dict()
                
                # Save epoch results and checkpoint
                self.save_checkpoint(epoch, is_best)
                self.save_epoch_results(epoch, bin_accuracies, bin_confidences, bin_counts, per_class_acc)
                self.save_training_progress()  # Save progress after each epoch
                
                # Print epoch stats
                print(f'Epoch {epoch+1} completed in {self.history["epoch_time"][-1]:.2f}s | '
                      f'Best test acc: {100.*self.best_acc:.2f}%')
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current progress...")
            # Save the current model and progress
            self.save_checkpoint(epoch, False)
            self.save_training_progress()
        except Exception as e:
            print(f"\nError during training: {e}")
            print("Saving current progress...")
            # Save the current model and progress
            self.save_checkpoint(epoch, False)
            self.save_training_progress()
            raise
            
        # Plot final training progress
        self.save_training_progress()
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model, self.history
    
    def save_epoch_results(self, epoch, bin_accuracies, bin_confidences, bin_counts, per_class_acc):
        """Save epoch results including reliability diagram and per-class accuracy."""
        # Create directory for this noise type and level if it doesn't exist
        result_dir = os.path.join(
            self.args.save_dir, 
            f'noise_{self.args.noise_type}_{self.args.noise_level}'
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Plot reliability diagram
        title = f'Reliability Diagram - {self.args.noise_type.capitalize()} Noise {self.args.noise_level}'
        save_path = os.path.join(result_dir, f'reliability_diagram_epoch_{epoch+1}.png')
        plot_reliability_diagram(bin_accuracies, bin_confidences, bin_counts, title, save_path)
        
        # Plot per-class accuracy
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        title = f'Per-class Accuracy - {self.args.noise_type.capitalize()} Noise {self.args.noise_level}'
        save_path = os.path.join(result_dir, f'per_class_acc_epoch_{epoch+1}.png')
        plot_per_class_accuracy(per_class_acc, class_names, title, save_path)
        
        # Save metrics for this epoch
        metrics = {
            'epoch': epoch,
            'train_loss': self.history['train_loss'][-1],
            'train_acc': self.history['train_acc'][-1],
            'test_loss': self.history['test_loss'][-1],
            'test_acc': self.history['test_acc'][-1],
            'test_ece': self.history['test_ece'][-1],
            'test_entropy': self.history['test_entropy'][-1],
            'bin_accuracies': bin_accuracies.tolist() if isinstance(bin_accuracies, np.ndarray) else bin_accuracies,
            'bin_confidences': bin_confidences.tolist() if isinstance(bin_confidences, np.ndarray) else bin_confidences,
            'bin_counts': bin_counts.tolist() if isinstance(bin_counts, np.ndarray) else bin_counts,
            'per_class_acc': per_class_acc.tolist() if isinstance(per_class_acc, np.ndarray) else per_class_acc
        }
        
        # Save as JSON
        metrics_path = os.path.join(result_dir, f'metrics_epoch_{epoch+1}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_training_progress(self):
        """Save training progress plots."""
        # Create directory for this noise type and level if it doesn't exist
        result_dir = os.path.join(
            self.args.save_dir, 
            f'noise_{self.args.noise_type}_{self.args.noise_level}'
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Plot training progress
        save_path = os.path.join(result_dir, 'training_progress.png')
        plot_training_progress(self.history, save_path)
        
        # Save history as numpy array
        np.save(os.path.join(result_dir, 'history.npy'), self.history)
        
        # Also save history as JSON for better readability
        history_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in self.history.items()}
        
        with open(os.path.join(result_dir, 'history.json'), 'w') as f:
            json.dump(history_json, f, indent=2)
    
    def get_final_logits_and_targets(self):
        """
        Get the logits and targets from the test set using the best model.
        
        Returns:
            logits: Model output logits
            targets: Ground truth labels
        """
        self.model.eval()
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all outputs and targets
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return all_logits, all_targets