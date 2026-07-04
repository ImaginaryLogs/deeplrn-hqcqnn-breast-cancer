  
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class MiddleLayerBase(nn.Module, ABC):
    """Common contract for all middle-layer variants."""
 
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
 
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
class BaselineCNN(MiddleLayerBase):
    def __init__(self, in_channels: int, out_channels: int, width: int = 16):
        super().__init__(in_channels, out_channels)
    
    def forward(self, x):
        return torch.Tensor([0])
    
class ResidualMiddleLayer(MiddleLayerBase):
    """Variant R: single residual block, ResNet motif."""
    def __init__(self, in_channels: int, out_channels: int, width: int = 16):
        super().__init__(in_channels, out_channels)
    
    def forward(self, x):
        return torch.Tensor([0])

class DenseMiddleLayer(MiddleLayerBase):
    """Variant D: 3 conv layers with concatenated feature reuse, DenseNet motif."""
    def __init__(self, in_channels: int, out_channels: int, width: int = 16):
        super().__init__(in_channels, out_channels)
    
    def forward(self, x):
        return torch.Tensor([0])
    
class InceptionMiddleLayer(MiddleLayerBase):
    def __init__(self, in_channels: int, out_channels: int, width: int = 16):
        super().__init__(in_channels, out_channels)
    
    def forward(self, x):
        return torch.Tensor([0])
    