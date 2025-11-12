import torch.nn as nn
import torch
import torch.nn.functional as F

class SNN_LRN_Wrapper(nn.Module):
    """
    Wrapper for SNN_LRN that handles temporal simulation.
    Processes spike tensors through SNN and outputs coarse images.
    
    This SNN model:
    - Takes spike tensors as input (voxelized: [B, 50, 224, 224])
    - Processes through spiking neurons (Leaky Integrate-and-Fire)
    - Outputs coarse images ([B, 1, 224, 224])
    """
    def __init__(self, inDim=50, outDim=1, num_steps=50, beta=0.9, threshold=1.0, norm='none'):
        super(SNN_LRN_Wrapper, self).__init__()
        self.num_steps = num_steps
        self.inDim = inDim
        self.outDim = outDim
        
        # Build SNN layers manually for proper temporal handling
        try:
            import snntorch as snn
        except ImportError:
            raise ImportError(
                "snnTorch is required for SNN model. Install with: pip install snnTorch"
            )
        
        # Layer 1
        self.conv1 = nn.Conv2d(inDim, 64, kernel_size=3, stride=1, padding=1)
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(64, affine=False, track_running_stats=False)
        else:
            self.norm1 = None
        # Use init_hidden=False to prevent graph reuse issues during training
        self.snn1 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=False)
        
        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        if norm in ('batch', 'instance'):
            if norm == 'batch':
                self.norm2 = nn.BatchNorm2d(128)
            else:
                self.norm2 = nn.InstanceNorm2d(128, affine=False, track_running_stats=False)
        else:
            self.norm2 = None
        self.snn2 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=False)
        
        # Layer 3
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        if norm in ('batch', 'instance'):
            if norm == 'batch':
                self.norm3 = nn.BatchNorm2d(64)
            else:
                self.norm3 = nn.InstanceNorm2d(64, affine=False, track_running_stats=False)
        else:
            self.norm3 = None
        self.snn3 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=False)
        
        # Layer 4
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        if norm in ('batch', 'instance'):
            if norm == 'batch':
                self.norm4 = nn.BatchNorm2d(16)
            else:
                self.norm4 = nn.InstanceNorm2d(16, affine=False, track_running_stats=False)
        else:
            self.norm4 = None
        self.snn4 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=False)
        
        # Final layer (no spiking, outputs analog values)
        self.conv_final = nn.Conv2d(16, outDim, kernel_size=3, stride=1, padding=1)
    
    def reset_hidden(self):
        """Reset hidden states of all SNN neurons. Call this before each forward pass during training."""
        self.snn1.reset()
        self.snn2.reset()
        self.snn3.reset()
        self.snn4.reset()
        
    def forward(self, x):
        """
        Forward pass through SNN with temporal simulation.
        
        Args:
            x: Input tensor [B, C, H, W] - voxelized spike data (C=50 channels)
               Each channel represents a temporal bin of spike data
        
        Returns:
            Output tensor [B, outDim, H, W] - reconstructed coarse image
        """
        B, C, H, W = x.shape
        
        # Process voxelized input: each channel is a temporal bin
        # For SNN, we can either:
        # 1. Process all channels at once (treat as spatial channels) - simpler
        # 2. Process sequentially over time (proper SNN simulation) - more accurate
        
        # Approach: Process voxelized input through SNN
        # The voxelized input [B, 50, H, W] is fed to conv1 which expects 50 channels
        # This is compatible with the original LRN architecture
        
        # Layer 1: Conv + Normalization + SNN neuron
        out = self.conv1(x)  # [B, 64, H, W]
        if self.norm1 is not None:
            out = self.norm1(out)
        # Create fresh hidden state tensor (zero-initialized) for each forward pass
        # This prevents graph reuse issues
        mem1 = torch.zeros_like(out, requires_grad=False)
        snn_out1 = self.snn1(out, mem1)  # SNN processes and outputs spikes
        # Handle different return formats from snnTorch
        if isinstance(snn_out1, tuple):
            spk1 = snn_out1[0]  # First element is always the spike output
        else:
            spk1 = snn_out1
        
        # Layer 2: Conv + Normalization + SNN neuron
        out = self.conv2(spk1)  # [B, 128, H, W]
        if self.norm2 is not None:
            out = self.norm2(out)
        mem2 = torch.zeros_like(out, requires_grad=False)
        snn_out2 = self.snn2(out, mem2)
        if isinstance(snn_out2, tuple):
            spk2 = snn_out2[0]
        else:
            spk2 = snn_out2
        
        # Layer 3: Conv + Normalization + SNN neuron
        out = self.conv3(spk2)  # [B, 64, H, W]
        if self.norm3 is not None:
            out = self.norm3(out)
        mem3 = torch.zeros_like(out, requires_grad=False)
        snn_out3 = self.snn3(out, mem3)
        if isinstance(snn_out3, tuple):
            spk3 = snn_out3[0]
        else:
            spk3 = snn_out3
        
        # Layer 4: Conv + Normalization + SNN neuron
        out = self.conv4(spk3)  # [B, 16, H, W]
        if self.norm4 is not None:
            out = self.norm4(out)
        mem4 = torch.zeros_like(out, requires_grad=False)
        snn_out4 = self.snn4(out, mem4)
        if isinstance(snn_out4, tuple):
            spk4 = snn_out4[0]
        else:
            spk4 = snn_out4
        
        # Final layer: Conv (analog output, no spiking)
        # Convert spike output to analog for final reconstruction
        output = self.conv_final(spk4)  # [B, outDim, H, W]
        
        return output
