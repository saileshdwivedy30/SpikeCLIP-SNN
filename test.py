#!/usr/bin/env python3
"""
Test script to demonstrate SpikeCLIP reconstruction on sample data.
Shows input spikes, reconstructed images, and outputs visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import clip
from torchvision.transforms import Normalize

from models import SNN_LRN_Wrapper
from utils import normal_img, save_img, middleTFI
from dataset import SpikeData

def visualize_results(spike, recon, tfp, tfi, save_path):
    """Visualize input spike, reconstruction, and baselines"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Input spike (temporal average)
    spike_vis = normal_img(torch.mean(spike, dim=1).unsqueeze(0), RGB=False, nor=True)
    axes[0].imshow(spike_vis, cmap='gray')
    axes[0].set_title('Input Spike (TFP)', fontsize=12)
    axes[0].axis('off')
    
    # TFP baseline
    tfp_vis = normal_img(tfp, RGB=False, nor=True)
    axes[1].imshow(tfp_vis, cmap='gray')
    axes[1].set_title('TFP Baseline', fontsize=12)
    axes[1].axis('off')
    
    # TFI baseline
    tfi_vis = normal_img(tfi, RGB=False, nor=True)
    axes[2].imshow(tfi_vis, cmap='gray')
    axes[2].set_title('TFI Baseline', fontsize=12)
    axes[2].axis('off')
    
    # SpikeCLIP reconstruction
    recon_vis = normal_img(recon[0, 0:1], RGB=False, nor=True)
    axes[3].imshow(recon_vis, cmap='gray')
    axes[3].set_title('SpikeCLIP Reconstruction', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

def test_model(model, test_loader, device, clip_model, labels, output_dir, num_samples=5, prompt_learner=None, text_encoder=None):
    """Test the model on a few samples"""
    model.eval()
    if prompt_learner is not None:
        prompt_learner.eval()
    
    # Setup CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    weight = np.array((0.299, 0.587, 0.114))
    gray_mean = sum(mean * weight)
    gray_std = np.sqrt(np.sum(np.power(weight, 2) * np.power(std, 2)))
    normal_clip = Normalize((gray_mean, ), (gray_std, ))
    
    # Use learned prompts if available, otherwise use standard CLIP
    if prompt_learner is not None and text_encoder is not None:
        with torch.no_grad():
            prompts_hq = prompt_learner(hq=True)  # [n_cls, seq_len, embed_dim] - use high-quality prompts
            val_features = text_encoder(prompts_hq, None)  # [n_cls, embed_dim]
            val_features = val_features / val_features.norm(dim=-1, keepdim=True)
        print(f"Using learned HQ prompts for classification (shape: {val_features.shape})")
    else:
        # Fallback to standard CLIP
        val_prompts = ['image of a ' + prompt for prompt in labels]
        text = clip.tokenize(val_prompts).to(device)
        val_features = clip_model.encode_text(text)
        val_features = val_features / val_features.norm(dim=-1, keepdim=True)
        print(f"Using standard CLIP text features (learned prompts not available)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (spike, label, label_idx) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            
            spike = spike.float().to(device)
            label_idx = label_idx.to(device)
            
            # Reconstruction
            # Dynamic voxelization: handles any T value and resamples to 50 bins
            from utils import make_voxel_224
            voxel = make_voxel_224(spike, bin_size=4, target_bins=50).to(device)
            spike_recon = model(voxel).repeat((1, 3, 1, 1))
            
            # Baselines
            tfp = torch.mean(spike, dim=1, keepdim=True)
            tfi = middleTFI(spike, len(spike[0])//2, len(spike[0])//4-1)
            
            # Classification with CLIP
            spike_recon_norm = normal_clip(spike_recon)
            image_features = clip_model.encode_image(spike_recon_norm)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ val_features.t()
            probs = logits.softmax(dim=-1)
            pred_idx = torch.max(probs, dim=1).indices.detach().cpu()
            
            # Check accuracy
            correct += (pred_idx == label_idx.cpu()).sum().item()
            total += len(label_idx)
            
            # Visualization
            save_path = os.path.join(output_dir, f'sample_{batch_idx:04d}_comparison.png')
            visualize_results(spike[0], spike_recon[0:1], tfp[0], tfi[0], save_path)
            
            # Save individual images
            save_img(path=os.path.join(output_dir, f'sample_{batch_idx:04d}_recon.png'),
                    img=spike_recon[0, 0, 30:-30, 10:-10])
            save_img(path=os.path.join(output_dir, f'sample_{batch_idx:04d}_tfp.png'),
                    img=tfp[0, 0, 30:-30, 10:-10])
            save_img(path=os.path.join(output_dir, f'sample_{batch_idx:04d}_tfi.png'),
                    img=tfi[0, 0, 30:-30, 10:-10])
            
            print(f"Sample {batch_idx}: Predicted '{labels[pred_idx[0]]}', Actual '{label[0]}'")
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\nClassification Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Outputs saved to {output_dir}")

def benchmark_performance(model, test_loader, device, num_batches=50):
    """Benchmark model performance (latency, throughput, power)"""
    model.eval()
    
    import time
    import subprocess
    
    # Check if CUDA is available
    use_cuda = (isinstance(device, torch.device) and device.type == 'cuda') or (isinstance(device, str) and device == 'cuda')
    use_cuda = use_cuda and torch.cuda.is_available()
    
    times = []
    power_readings = []
    
    # Check if nvidia-smi is available for power measurement
    measure_power = False
    if use_cuda:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                measure_power = True
                print("Power measurement enabled (nvidia-smi available)")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            print("Power measurement disabled (nvidia-smi not available)")
    
    with torch.no_grad():
        for batch_idx, (spike, _, _) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            
            spike = spike.float().to(device)
            # Dynamic voxelization: handles any T value and resamples to 50 bins
            from utils import make_voxel_224
            voxel = make_voxel_224(spike, bin_size=4, target_bins=50).to(device)
            
            # Warmup
            if batch_idx == 0:
                _ = model(voxel)
                if use_cuda:
                    torch.cuda.synchronize()
            
            # Measure power before inference (baseline)
            if measure_power:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        power_before = float(result.stdout.strip())
                except:
                    power_before = None
            else:
                power_before = None
            
            # Time inference
            if use_cuda:
                torch.cuda.synchronize()
            start = time.time()
            _ = model(voxel)
            if use_cuda:
                torch.cuda.synchronize()
            end = time.time()
            
            # Measure power after inference
            if measure_power:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        power_after = float(result.stdout.strip())
                        if power_before is not None:
                            # Use average of before/after or just after
                            power_readings.append(power_after)
                except:
                    pass
            
            times.append(end - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\nPerformance Benchmark:")
    print(f"  Device: {device}")
    print(f"  Average latency: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {fps:.2f} FPS")
    
    if measure_power and len(power_readings) > 0:
        avg_power = np.mean(power_readings)
        max_power = np.max(power_readings)
        min_power = np.min(power_readings)
        print(f"  Average power: {avg_power:.2f} W")
        print(f"  Power range: {min_power:.2f} - {max_power:.2f} W")
        return avg_time, fps, avg_power
    else:
        print(f"  Power: Not available (requires nvidia-smi on GPU)")
        return avg_time, fps, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SpikeCLIP model')
    parser.add_argument('--checkpoint', type=str, default='models/LRN_CALTECH.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_type', type=str, default='CALTECH', choices=['CALTECH', 'CIFAR'])
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to test data (auto-set based on data_type if not provided)')
    parser.add_argument('--output_dir', type=str, default='test_outputs',
                       help='Directory to save test outputs')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for results tracking')
    
    args = parser.parse_args()
    
    # Labels and data paths
    if args.data_type == 'CALTECH':
        labels = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
        if args.data_path is None:
            args.data_path = 'data/U-CALTECH'
    else:
        labels = ["frog","horse","dog","truck","airplane","automobile","bird","ship","cat","deer"]
        if args.data_path is None:
            args.data_path = 'data/U-CIFAR'
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP
    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root="./clip_model/")
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    # Import PromptLearner and TextEncoder from train.py
    import sys
    import importlib.util
    train_path = 'train.py'
    spec = importlib.util.spec_from_file_location("train", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    PromptLearner = train_module.PromptLearner
    TextEncoder = train_module.TextEncoder
    
    # Initialize prompt learner and text encoder
    prompt_learner = PromptLearner(clip_model, n_cls=len(labels)).to(device)
    text_encoder = TextEncoder(clip_model)
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    text_encoder = text_encoder.to(device)
    
    # Load reconstruction model (SNN)
    print(f"Loading model from {args.checkpoint}...")
    model = SNN_LRN_Wrapper(inDim=50, outDim=1, num_steps=50).to(device)
    print("Using SNN (Spiking Neural Network) model")
    
    if os.path.exists(args.checkpoint):
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint_data, dict):
            # New format: dictionary with 'recon_net', 'prompt_learner', etc.
            if 'recon_net' in checkpoint_data:
                model.load_state_dict(checkpoint_data['recon_net'])
                if 'prompt_learner' in checkpoint_data:
                    prompt_learner.load_state_dict(checkpoint_data['prompt_learner'])
                    print(f"Model loaded successfully (from checkpoint dict, epoch {checkpoint_data.get('epoch', 'unknown')}) - including prompt_learner")
                else:
                    print(f"Model loaded successfully (from checkpoint dict, epoch {checkpoint_data.get('epoch', 'unknown')}) - prompt_learner not found, using random init")
            elif 'state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['state_dict'])
                print("Model loaded successfully (from state_dict)")
            else:
                # Assume it's the state dict itself
                model.load_state_dict(checkpoint_data)
                print("Model loaded successfully (direct state_dict)")
        else:
            # Old format: direct state dict
            model.load_state_dict(checkpoint_data)
            print("Model loaded successfully")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")
    
    # Load test data
    print(f"Loading test data from {args.data_path}...")
    test_dataset = SpikeData(args.data_path, labels, stage='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    print(f"Found {len(test_dataset)} test samples")
    
    # Get checkpoint info for results tracking
    checkpoint_epoch = None
    best_val_accuracy = None
    if os.path.exists(args.checkpoint):
        try:
            checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
            if isinstance(checkpoint_data, dict):
                checkpoint_epoch = checkpoint_data.get('epoch')
                best_val_accuracy = checkpoint_data.get('accuracy')
        except:
            pass
    
    # Run tests
    print("\n" + "="*60)
    print("Running Reconstruction Tests")
    print("="*60)
    test_accuracy = None
    # Note: test_model prints accuracy but doesn't return it
    # We'll capture it from benchmark or use evaluate.py for full test accuracy
    test_model(model, test_loader, device, clip_model, labels, args.output_dir, args.num_samples, prompt_learner, text_encoder)
    
    # Run benchmark if requested
    latency_ms = None
    throughput_fps = None
    power_watts = None
    if args.benchmark:
        print("\n" + "="*60)
        print("Running Performance Benchmark")
        print("="*60)
        result = benchmark_performance(model, test_loader, device)
        if len(result) >= 2:
            latency_ms = result[0] * 1000  # Convert to ms
            throughput_fps = result[1]
            if len(result) >= 3:
                power_watts = result[2]
    
    # Save results to JSON
    try:
        from results_tracker import add_run, print_comparison
        
        # Extract exp_name from checkpoint path if available
        exp_name = args.exp_name if args.exp_name else args.checkpoint.split('/')[-2] if '/' in args.checkpoint else 'unknown'
        
        add_run(
            exp_name=exp_name,
            checkpoint_path=args.checkpoint,
            best_val_accuracy=best_val_accuracy,
            best_val_epoch=checkpoint_epoch,
            test_accuracy=test_accuracy,  # Will be None for test.py (use evaluate.py for full test accuracy)
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            power_watts=power_watts,
            data_type=args.data_type
        )
        
        # Print comparison
        print_comparison()
    except ImportError:
        print("Warning: results_tracker not available. Results not saved to JSON.")
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

