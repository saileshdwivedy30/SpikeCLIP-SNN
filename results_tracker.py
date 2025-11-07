#!/usr/bin/env python3
"""
Results tracking utility for saving and comparing experiment results.
Saves metrics to JSON and automatically identifies the best run.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Any

RESULTS_FILE = "experiment_results.json"

def load_results() -> Dict[str, Any]:
    """Load existing results from JSON file."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"runs": [], "best_run": None}
    return {"runs": [], "best_run": None}

def save_results(results: Dict[str, Any]):
    """Save results to JSON file."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def add_run(
    exp_name: str,
    checkpoint_path: str,
    best_val_accuracy: Optional[float] = None,
    best_val_epoch: Optional[int] = None,
    test_accuracy: Optional[float] = None,
    latency_ms: Optional[float] = None,
    throughput_fps: Optional[float] = None,
    power_watts: Optional[float] = None,
    niqe: Optional[float] = None,
    brisque: Optional[float] = None,
    piqe: Optional[float] = None,
    data_type: Optional[str] = None,
    num_train_samples: Optional[int] = None,
    num_test_samples: Optional[int] = None,
    hyperparameters: Optional[Dict[str, Any]] = None
):
    """
    Add a new run to the results and update best run if needed.
    
    Args:
        exp_name: Experiment name
        checkpoint_path: Path to checkpoint file
        best_val_accuracy: Best validation accuracy during training
        best_val_epoch: Epoch number with best validation accuracy
        test_accuracy: Test set accuracy
        latency_ms: Average inference latency in milliseconds
        throughput_fps: Throughput in frames per second
        power_watts: Average power consumption in watts
        niqe: NIQE image quality metric
        brisque: BRISQUE image quality metric
        piqe: PIQE image quality metric
        data_type: Dataset type (CIFAR, CALTECH)
        num_train_samples: Number of training samples
        num_test_samples: Number of test samples
        hyperparameters: Dictionary of hyperparameters used
    """
    results = load_results()
    
    # Helper function to convert values to JSON-serializable types
    def to_serializable(val):
        if val is None:
            return None
        if hasattr(val, 'item'):  # Tensor
            return float(val.item())
        if isinstance(val, (int, float)):
            return float(val)
        return val
    
    # Create run entry
    run_entry = {
        "exp_name": exp_name,
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "best_val_accuracy": to_serializable(best_val_accuracy),
            "best_val_epoch": to_serializable(best_val_epoch),
            "test_accuracy": to_serializable(test_accuracy),
            "latency_ms": to_serializable(latency_ms),
            "throughput_fps": to_serializable(throughput_fps),
            "power_watts": to_serializable(power_watts),
            "niqe": to_serializable(niqe),
            "brisque": to_serializable(brisque),
            "piqe": to_serializable(piqe)
        },
        "config": {
            "data_type": data_type,
            "num_train_samples": num_train_samples,
            "num_test_samples": num_test_samples,
            "hyperparameters": hyperparameters or {}
        }
    }
    
    # Add to runs list
    results["runs"].append(run_entry)
    
    # Determine best run based on test accuracy (primary) and validation accuracy (secondary)
    best_run = results.get("best_run")
    best_test_acc = None
    best_val_acc = None
    
    if best_run is not None:
        best_test_acc = best_run.get("metrics", {}).get("test_accuracy")
        best_val_acc = best_run.get("metrics", {}).get("best_val_accuracy")
    
    # Check if current run is better
    current_test_acc = test_accuracy
    current_val_acc = best_val_accuracy
    
    is_better = False
    if current_test_acc is not None:
        if best_test_acc is None or current_test_acc > best_test_acc:
            is_better = True
        elif best_test_acc is not None and current_test_acc == best_test_acc:
            # Tie-breaker: use validation accuracy
            if current_val_acc is not None:
                if best_val_acc is None or current_val_acc > best_val_acc:
                    is_better = True
    
    if is_better:
        results["best_run"] = run_entry
        print(f"\n🏆 NEW BEST RUN! Test Accuracy: {current_test_acc:.2f}%")
        if best_test_acc is not None:
            print(f"   Previous best: {best_test_acc:.2f}%")
    else:
        if best_test_acc is not None:
            print(f"\n📊 Current run: Test Accuracy: {current_test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
    
    # Save results
    save_results(results)
    
    return run_entry, is_better

def get_best_run() -> Optional[Dict[str, Any]]:
    """Get the best run from saved results."""
    results = load_results()
    return results.get("best_run")

def print_comparison():
    """Print comparison of all runs."""
    results = load_results()
    runs = results.get("runs", [])
    best_run = results.get("best_run")
    
    if not runs:
        print("No runs recorded yet.")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*80)
    
    for i, run in enumerate(runs, 1):
        metrics = run.get("metrics", {})
        config = run.get("config", {})
        is_best = (best_run is not None and run.get("exp_name") == best_run.get("exp_name") and 
                  run.get("timestamp") == best_run.get("timestamp"))
        
        marker = "🏆 BEST" if is_best else f"  #{i}"
        print(f"\n{marker} - {run.get('exp_name', 'Unknown')}")
        print(f"  Timestamp: {run.get('timestamp', 'Unknown')}")
        print(f"  Checkpoint: {run.get('checkpoint_path', 'Unknown')}")
        
        if metrics.get("test_accuracy") is not None:
            print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%")
        if metrics.get("best_val_accuracy") is not None:
            print(f"  Val Accuracy: {metrics['best_val_accuracy']:.2f}% (Epoch {metrics.get('best_val_epoch', '?')})")
        if metrics.get("latency_ms") is not None:
            print(f"  Latency: {metrics['latency_ms']:.2f} ms")
        if metrics.get("throughput_fps") is not None:
            print(f"  Throughput: {metrics['throughput_fps']:.2f} FPS")
        if metrics.get("power_watts") is not None:
            print(f"  Power: {metrics['power_watts']:.2f} W")
        if metrics.get("brisque") is not None:
            print(f"  BRISQUE: {metrics['brisque']:.4f}")
        if metrics.get("niqe") is not None:
            print(f"  NIQE: {metrics['niqe']:.4f}")
        if metrics.get("piqe") is not None:
            print(f"  PIQE: {metrics['piqe']:.4f}")
        
        if config.get("data_type"):
            print(f"  Dataset: {config['data_type']}")
        if config.get("num_train_samples"):
            print(f"  Train Samples: {config['num_train_samples']}")
    
    print("\n" + "="*80)
    
    if best_run:
        print(f"\n🏆 BEST RUN: {best_run.get('exp_name', 'Unknown')}")
        best_metrics = best_run.get("metrics", {})
        if best_metrics.get("test_accuracy") is not None:
            print(f"   Test Accuracy: {best_metrics['test_accuracy']:.2f}%")
        if best_metrics.get("best_val_accuracy") is not None:
            print(f"   Val Accuracy: {best_metrics['best_val_accuracy']:.2f}% (Epoch {best_metrics.get('best_val_epoch', '?')})")
        if best_metrics.get("latency_ms") is not None:
            print(f"   Latency: {best_metrics['latency_ms']:.2f} ms")
        if best_metrics.get("throughput_fps") is not None:
            print(f"   Throughput: {best_metrics['throughput_fps']:.2f} FPS")
        if best_metrics.get("power_watts") is not None:
            print(f"   Power: {best_metrics['power_watts']:.2f} W")

if __name__ == "__main__":
    # Print comparison when run directly
    print_comparison()

