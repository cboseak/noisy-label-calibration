"""
Model definition module.
"""
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(args):
    """
    Create a ResNet-18 model for CIFAR-10 classification.
    
    Args:
        args: Configuration arguments
        
    Returns:
        model: ResNet-18 model
    """
    # Get pre-trained ResNet-18 model
    model = models.resnet18(pretrained=args.pretrained)
    
    # Modify first layer for CIFAR-10 (32x32 images instead of ImageNet's 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool as it's not needed for CIFAR-10
    
    # Modify final fully connected layer for 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    return model

class ModelWithTemperature(nn.Module):
    """
    A thin wrapper around a model that can scale the logits with temperature
    for better confidence calibration.
    """
    def __init__(self, model, device='cuda'):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.device = device
        self.model.to(self.device)
        
    def forward(self, input):
        """Forward pass with temperature scaling."""
        logits = self.model(input)
        return self.temperature_scale(logits)
    
    def temperature_scale(self, logits):
        """
        Scales the logits by temperature for better calibration.
        
        Args:
            logits: Raw logits from the model
            
        Returns:
            Scaled logits (confidence scores)
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def get_logits(self, input):
        """Get raw logits without temperature scaling."""
        with torch.no_grad():
            return self.model(input)
            
    def get_confidence(self, input):
        """Get confidence scores (softmax of scaled logits)."""
        with torch.no_grad():
            logits = self.model(input)
            scaled_logits = self.temperature_scale(logits)
            return torch.softmax(scaled_logits, dim=1)
            
    def get_predictions(self, input):
        """Get class predictions."""
        with torch.no_grad():
            logits = self.model(input)
            scaled_logits = self.temperature_scale(logits)
            return torch.argmax(scaled_logits, dim=1)