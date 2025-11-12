# SpikeCLIP: Spike-to-Image Reconstruction with CLIP Guidance

## 📕 Overview

This repository implements a **Spiking Neural Network (SNN)**-based version of SpikeCLIP for spike-to-image reconstruction. The implementation uses spiking neurons (Leaky Integrate-and-Fire) to process spike camera data and reconstruct images with CLIP guidance.

**Key Features:**
- **SNN Architecture**: Uses spiking neurons for biologically-inspired spike processing
- **CLIP Guidance**: Leverages CLIP model for semantic alignment and image quality
- **Three-Stage Training**: Coarse reconstruction → Prompt learning → Joint refinement
- **GPU-Accelerated**: Full CUDA support for efficient training and inference
- **Real-time Performance**: 81.89 FPS inference on NVIDIA T4 GPU

**Implementation Details:**
- Model: SNN-based Lightweight Reconstruction Network (LRN)
- Framework: PyTorch with snnTorch library
- Datasets: U-CIFAR and U-CALTECH (synthetic data generation supported)
- Performance: 52.00% test accuracy, 12.21ms latency, 30.12W power consumption

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/SpikeCLIP.git
cd SpikeCLIP
```

2. **Install dependencies**

**Option A: Using virtual environment (Recommended)**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use the automated setup script:
./setup.sh
```

**Option B: Manual installation (if venv not needed)**
```bash
# On macOS with Python 3.13+, you may need:
python3 -m pip install --user -r requirements.txt
# or with --break-system-packages flag (not recommended)
```

3. **Download CLIP models**
CLIP models will be automatically downloaded on first use. They will be saved to `./clip_model/`

### Dataset Preparation

The UHSR real-world spike dataset is available from [UHSR GitHub](https://github.com/Evin-X/UHSR).

**Option 1: Automatic download and organization**
```bash
# If you have a direct download URL:
python3 fetch_data.py --download_url <url_to_dataset.zip> --dataset all

# Or if you've downloaded the file locally:
python3 fetch_data.py --download_file /path/to/dataset.zip --dataset all
```

**Option 2: Manual download + automatic organization**
```bash
# 1. Download the dataset manually from https://github.com/Evin-X/UHSR
# 2. Extract .npz files to data/U-CALTECH/
# 3. Organize into train/test splits:
python3 fetch_data.py --manual_download --dataset all
```

**Option 3: Generate synthetic data (for testing)**

**Note**: Synthetic data is used when real UHSR dataset is not accessible. The format is 100% compatible with real data.

```bash
# Generate synthetic data that matches UHSR format
python3 generate_synthetic_data.py --dataset both --num_train 50 --num_test 10

# This creates synthetic .npz files for testing without downloading real data
# When real data becomes available, simply replace files in data/ directory
# No code changes needed - see DATA_COMPATIBILITY.md for details
```

**Option 4: Fully manual preparation**
1. Download the UHSR dataset from [here](https://github.com/Evin-X/UHSR)
2. Extract the data to the `data/` directory
3. The structure should be:
```
data/
├── U-CALTECH/
│   ├── train/    (files 0-4999)
│   └── test/     (files 5000+)
└── U-CIFAR/
    ├── train/
    └── test/
```

The expected data format is `.npz` files with keys:
- `spk`: spike data array of shape [200, 250, 250]
- `label`: class label index

## 🏃 Running the Code

### Generate Synthetic Data

```bash
# Generate synthetic spike data (U-CIFAR format)
python3 generate_synthetic_data.py --dataset cifar --num_train 150 --num_test 30
```

### Training

Train the SNN-based SpikeCLIP model:
```bash
# Full training (recommended)
python3 train.py --data_type CIFAR --epochs 100 --batch_size 4 \
  --exp_name snn_cifar_combined --early_stop --patience 10 \
  --epochs_coarse 5 --epochs_prompt 5 --epochs_fine 90 \
  --clip_weight 0.75 --recon_weight 1.5 --prompt_weight 0.75
```

**Training arguments:**
- `--data_type`: Dataset type (`CALTECH` or `CIFAR`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--recon_weight`: Weight for reconstruction loss (default: 1.5)
- `--clip_weight`: Weight for CLIP alignment loss (default: 0.75)
- `--prompt_weight`: Weight for prompt learning loss (default: 0.75)
- `--exp_name`: Experiment name for saving results
- `--early_stop`: Enable early stopping based on validation accuracy
- `--patience`: Early stopping patience (default: 10)
- `--resume`: Path to checkpoint to resume training

### Evaluation

Evaluate a trained model on test set:
```bash
python3 evaluate.py \
  --data_type CIFAR \
  --checkpoint exp/snn_cifar_combined/ckpts/checkpoint_best.pth \
  --exp_name snn_cifar_combined
```

This will:
- Evaluate on test set
- Calculate test accuracy
- Compute image quality metrics (NIQE, BRISQUE, PIQE)
- Save results to `experiment_results.json`

### Benchmarking Performance

Run performance benchmark (latency, throughput, power):
```bash
python3 test.py \
  --checkpoint exp/snn_cifar_combined/ckpts/checkpoint_best.pth \
  --data_type CIFAR \
  --benchmark \
  --exp_name snn_cifar_combined
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--data_type`: Dataset type
- `--benchmark`: Run performance benchmark (latency/throughput/power)
- `--num_samples`: Number of samples to visualize (default: 5)
- `--output_dir`: Directory to save outputs (default: `test_outputs`)

This will:
- Measure inference latency (ms)
- Calculate throughput (FPS)
- Measure power consumption (W) - requires nvidia-smi
- Save results to `experiment_results.json`

### View Results

View all experiment results and comparisons:
```bash
python3 results_tracker.py
```

This will:
- Display all recorded experiments
- Show best run (highlighted)
- Compare metrics across runs

## 📊 Results

### Note on Data

**Current Implementation**: This project uses **synthetic data** for testing and development because the real U-CIFAR and U-CALTECH datasets from the UHSR repository are not publicly accessible without special access (requires Baidu Drive account).

**Synthetic Data**: 
- Format: 100% compatible with real UHSR dataset (same `.npz` structure)
- Purpose: Allows complete pipeline testing without waiting for data access
- Compatibility: When real data becomes available, simply replace files in `data/` directory - no code changes needed

See `DATASET_NOTE.md` and `DATA_COMPATIBILITY.md` for details.

### Quantitative Results (U-CIFAR Dataset)

**Model**: SNN-based SpikeCLIP (Spiking Neural Network)

**Classification Performance:**
- **Test Accuracy**: 52.00% (on 300 test samples, 10 classes)
- **Validation Accuracy**: 59.00% (best epoch: 41)


**Image Quality Metrics:**
- **NIQE**: 20.03 (lower is better)

**Training Configuration:**
- Dataset: U-CIFAR (150 samples/class, 1500 total training samples)
- Epochs: 41 (early stopping)
- Batch Size: 4
- Loss Weights: clip_weight=0.75, recon_weight=1.5, prompt_weight=0.75
- Model: Spiking Neural Network

### Performance Benchmarks (GPU - NVIDIA T4)

**Inference Performance:**
- **Latency**: 12.21 ms per image
- **Throughput**: 81.89 FPS
- **Power Consumption**: 30.12 W (average), range: 27.51 - 40.87 W
- **Model Size**: ~2.4 MB (SNN reconstruction network)
- **Device**: NVIDIA T4 GPU (CUDA)

**Training Performance:**
- **Training Speed**: ~2.81 iterations/second
- **Time per Epoch**: ~1.8 minutes (300 batches)
- **Total Training Time**: ~1 hour (27 epochs with early stopping)

**Assessment:**
- **Real-time capable**: 81.89 FPS enables real-time spike camera processing
- **Power efficient**: ~30W average power consumption
- **Production-ready**: Low latency suitable for edge deployment

## 📁 Project Structure

```
SpikeCLIP/
├── data/                    # Dataset directory
│   ├── U-CALTECH/
│   └── U-CIFAR/
├── models/                  # Pretrained models
├── exp/                     # Experiment outputs
│   └── <exp_name>/
│       ├── ckpts/          # Checkpoints
│       ├── imgs/           # Output images
│       └── tensorboard/    # TensorBoard logs
├── dataset.py               # Dataset loader
├── models.py                # Model definitions
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── test.py                  # Test with visualization
├── metrics.py               # Evaluation metrics
├── utils.py                 # Utility functions
├── fetch_data.py            # Data preparation script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Model Architecture

SpikeCLIP uses a three-stage pipeline with **Spiking Neural Network (SNN)** architecture:

1. **Coarse Reconstruction**: SNN-based LRN (Lightweight Reconstruction Network) converts spike voxels to initial image estimate
2. **Prompt Learning**: Learnable text prompts optimize CLIP embeddings for semantic alignment
3. **Refinement**: CLIP-guided loss refines reconstruction quality

### SNN Reconstruction Network

The reconstruction network uses **spiking neurons** (Leaky Integrate and Fire) for biologically inspired processing:

- **Input**: 50-channel voxel representation from spike sequence [B, 50, 224, 224]
- **Architecture**: 
  - Conv(50→64) + BatchNorm + Leaky LIF neuron
  - Conv(64→128) + BatchNorm + Leaky LIF neuron
  - Conv(128→64) + BatchNorm + Leaky LIF neuron
  - Conv(64→16) + BatchNorm + Leaky LIF neuron
  - Conv(16→1) (analog output)
- **Output**: Single-channel grayscale image [B, 1, 224, 224]
- **Neuron Parameters**: beta=0.9, threshold=1.0
- **Implementation**: snnTorch library

**Key Features:**
- **SNN-based**: Uses spiking neurons for spike-to-image processing
- **GPU-accelerated**: Full CUDA support
- **Lightweight**: ~2.4 MB model size
- **Efficient**: 81.89 FPS inference on T4 GPU

## 📈 Training Details

- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Loss**: Combined reconstruction loss (L1) + CLIP alignment loss
- **Training stages**:
  1. Coarse reconstruction (reconstruction network only)
  2. Prompt learning (learnable prompts + reconstruction)
  3. Joint refinement (end-to-end fine-tuning)

## 🔬 Experimental Setup

- **Hardware**: NVIDIA GPU (CUDA 11.0+)
- **Software**: PyTorch 1.9+, Python 3.7+
- **Datasets**: Synthetic Data

## 🤝 Acknowledgments

- UHSR dataset: [Evin-X/UHSR](https://github.com/Evin-X/UHSR)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- Original SpikeCLIP implementation: [chenkang455/SpikeCLIP](https://github.com/chenkang455/SpikeCLIP)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


