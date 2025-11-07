#!/usr/bin/env python3
"""
Generate synthetic spike camera data for testing SpikeCLIP.
- Matches UHSR-style .npz files with keys: 'spk' (T,H,W float32 in {0,1}) and 'label' (int)
- Folder structure:
    data/
      U-CALTECH/{train,test}/*.npz
      U-CIFAR/{train,test}/*.npz
- Incorporates:
    * Integrate-and-fire spiking with per-pixel thresholds (heterogeneity)
    * Motion via per-frame affine warp (speed regimes: slow/fast)
    * Low-light regimes (gain, dark current, shot noise)
    * Optional hot pixels & temporal leak
- Defaults are conservative and fast to generate; tune via CLI for heavier stress tests.
"""

import os
import cv2
import math
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict

# ----------------------------
# Utility
# ----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def zero_pad(n: int, width: int = 5) -> str:
    return f"{n:0{width}d}"

# ----------------------------
# Latent image generator (class-conditional patterns)
# ----------------------------
def generate_synthetic_image(class_idx: int, num_classes: int,
                             image_size: Tuple[int, int]=(224, 224)) -> np.ndarray:
    """
    Produces a simple grayscale latent image in [0,1] for a class index.
    Kept intentionally simple; the SpikeCLIP pipeline focuses on spiking & CLIP alignment.
    """
    H, W = image_size
    img = np.zeros((H, W), dtype=np.float32)
    third = num_classes // 3
    if class_idx < third:
        # Circular blobs
        center_x = W // 2 + int(30 * np.sin(class_idx))
        center_y = H // 2 + int(30 * np.cos(class_idx))
        radius = max(10, 30 + class_idx % 15)
        cv2.circle(img, (center_x, center_y), radius, 0.7, -1)
    elif class_idx < 2 * third:
        # Rectangles
        sx = 20 + (class_idx % third) * 2
        sy = 20 + (class_idx % third)
        x1, y1 = W//4 + sx, H//4 + sy
        x2, y2 = 3*W//4 - sx, 3*H//4 - sy
        cv2.rectangle(img, (x1, y1), (x2, y2), 0.8, -1)
    else:
        # Lines
        shift = (class_idx % third) * 3
        cv2.line(img, (shift, 0), (W - shift, H - 1), 0.6, thickness=3)
    
    # Smooth a bit and add noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.random.randn(H, W).astype(np.float32) * 0.05
    img = np.clip(img + noise, 0.0, 1.0)
    return img

# ----------------------------
# Motion model
# ----------------------------
def apply_motion(img: np.ndarray, t: int, T: int,
                 px_per_frame: float,
                 jitter: float = 0.25,
                 angle_deg: float = 0.0,
                 scale: float = 1.0,
                 border_mode=cv2.BORDER_REFLECT101) -> np.ndarray:
    """
    Apply a simple horizontal sweep with small jitter and optional rotation/scale.
    px_per_frame: average horizontal pixel shift per frame (speed proxy).
    """
    H, W = img.shape
    # Centered sweep over time: negative to positive displacement
    sweep = (2.0 * t / max(1, T - 1) - 1.0) * px_per_frame
    # Small random jitter per frame
    dx = sweep + np.random.randn() * jitter
    dy = np.random.randn() * (jitter * 0.25)
    # Compose rotation/scale and translation
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, scale)
    M[:, 2] += [dx, dy]
    warped = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border_mode)
    return warped

# ----------------------------
# Integrate-and-Fire spiking with low-light & heterogeneity
# ----------------------------
def image_to_spikes_integrate_and_fire_with_motion(
    base_img: np.ndarray,
    T: int = 200,
    params: Dict = None
) -> np.ndarray:
    """
    Convert a latent image to a (T,H,W) spike train using an integrate-and-fire model,
    with per-frame motion and low-light noise.
    params:
      - gain: nominal illumination gain (float)
      - gain_jitter: per-sample multiplicative jitter
      - dark_current: additive baseline
      - shot_noise: Poisson-like noise strength (0..)
      - theta_mean: mean threshold
      - theta_std: pixelwise threshold std
      - leak: integrator decay per step (0..1)
      - px_per_frame: motion speed proxy (pixels/frame)
      - hot_pixel_prob: rare hot pixels
      - hot_pixel_intensity: added intensity for hot pixels
    """
    H, W = base_img.shape
    p = {
        "gain": 0.5,
        "gain_jitter": 0.2,
        "dark_current": 0.002,
        "shot_noise": 0.5,
        "theta_mean": 1.0,
        "theta_std": 0.05,
        "leak": 0.99,
        "px_per_frame": 1.5,
        "hot_pixel_prob": 5e-4,
        "hot_pixel_intensity": 0.5,
        "angle_deg": 0.0,
        "scale": 1.0
    }
    if params:
        p.update(params)
    
    # Per-pixel thresholds (heterogeneity)
    theta_map = np.random.normal(p["theta_mean"], p["theta_std"], size=(H, W)).astype(np.float32)
    theta_map = np.clip(theta_map, 0.5 * p["theta_mean"], 1.5 * p["theta_mean"])
    
    # Integrator state
    v = np.zeros((H, W), dtype=np.float32)
    spk = np.zeros((T, H, W), dtype=np.float32)
    
    # Rare hot pixels
    hot = (np.random.rand(H, W) < p["hot_pixel_prob"]).astype(np.float32) * p["hot_pixel_intensity"]
    
    # Per-sample illumination
    gain = p["gain"] * float(np.random.lognormal(mean=0.0, sigma=p["gain_jitter"]))
    dark = p["dark_current"]
    shot = p["shot_noise"]
    pxpf = p["px_per_frame"]
    
    for t in range(T):
        # Motion on the latent
        img_t = apply_motion(
            base_img, t, T,
            px_per_frame=pxpf,
            jitter=0.25,
            angle_deg=p["angle_deg"],
            scale=p["scale"]
        )
        
        # Add hot pixels (static)
        img_t = np.clip(img_t + hot, 0.0, 1.0)
        
        # Illumination + low-light
        illum = img_t * gain + dark
        if shot > 0:
            # Poisson-like shot noise proportional to illum
            # Avoid very small rates by clamping illum
            lam = np.maximum(illum, 1e-6) * shot
            noise = np.random.poisson(lam).astype(np.float32) / max(1.0, shot)
            illum_noisy = illum + noise
        else:
            illum_noisy = illum
        
        # Integrate
        v = v * p["leak"] + illum_noisy
        
        # Spike when threshold crossed; allow multiple subtractions if very bright
        fired = v >= theta_map
        if np.any(fired):
            spk[t, fired] = 1.0
            v[fired] -= theta_map[fired]
            # (Optional) extra loop for rare multiple spikes per dt; skipped for speed
    
    return spk

# ----------------------------
# Sample & dataset generation
# ----------------------------
def generate_synthetic_sample(
    class_idx: int,
    num_classes: int,
    output_path: str,
    image_size: Tuple[int, int]=(224, 224),
    num_frames: int = 200,
    regime: str = "auto",
    lowlight: bool = False,
    seed_offset: int = 0
):
    """
    Generate one synthetic sample and save to .npz at output_path.
    - regime: 'slow', 'fast', or 'auto' (mix)
    - lowlight: True/False for dim scenes
    """
    # Latent image
    latent = generate_synthetic_image(class_idx, num_classes, image_size)
    
    # Pad to 250x250 (loader crops to 224x224 later)
    padded_size = (250, 250)
    padded = np.zeros(padded_size, dtype=np.float32)
    y0 = (padded_size[0] - image_size[0]) // 2
    x0 = (padded_size[1] - image_size[1]) // 2
    padded[y0:y0 + image_size[0], x0:x0 + image_size[1]] = latent
    
    # Choose motion & light params
    if regime == "auto":
        # 50/50 slow/fast by default; caller can randomize externally
        regime = "fast" if np.random.rand() < 0.5 else "slow"
    
    if regime == "slow":
        pxpf = np.random.uniform(0.2, 1.0)
    else:  # 'fast'
        pxpf = np.random.uniform(2.0, 5.0)
    
    if lowlight:
        gain = np.random.uniform(0.2, 0.5)
        dark_current = np.random.uniform(0.001, 0.004)
        shot_noise = np.random.uniform(0.6, 1.0)
    else:
        gain = np.random.uniform(0.6, 1.2)
        dark_current = np.random.uniform(0.0005, 0.002)
        shot_noise = np.random.uniform(0.2, 0.6)
    
    # Per-sample spiking params
    params = {
        "gain": float(gain),
        "gain_jitter": 0.25,
        "dark_current": float(dark_current),
        "shot_noise": float(shot_noise),
        "theta_mean": 1.0,
        "theta_std": 0.05,
        "leak": 0.99,
        "px_per_frame": float(pxpf),
        "hot_pixel_prob": 5e-4 if lowlight else 1e-4,
        "hot_pixel_intensity": 0.75 if lowlight else 0.5,
        "angle_deg": np.random.uniform(-5, 5),
        "scale": np.random.uniform(0.98, 1.02)
    }
    
    spk = image_to_spikes_integrate_and_fire_with_motion(
        padded, T=num_frames, params=params
    )
    
    # Save
    # Use compressed format to save disk space
    np.savez_compressed(output_path, spk=spk.astype(np.float32), label=int(class_idx))
    return spk, class_idx

def generate_dataset(
    output_dir: str,
    labels: list,
    num_train_per_class: int = 50,
    num_test_per_class: int = 10,
    image_size: Tuple[int, int]=(224, 224),
    num_frames: int = 200,
    lowlight_ratio: float = 0.5,
    fast_ratio: float = 0.5,
    seed: int = 0,
    start_index: int = 0
):
    """
    Generate a dataset with train/ and test/ subfolders.
    - lowlight_ratio: fraction of samples in low-light regime
    - fast_ratio: fraction of samples in 'fast' motion regime
    """
    ensure_dir(output_dir)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    ensure_dir(train_dir)
    ensure_dir(test_dir)
    
    num_classes = len(labels)
    file_idx = start_index
    
    # TRAIN
    print(f"\nGenerating TRAIN for {output_dir} (classes={num_classes})")
    for c in tqdm(range(num_classes), desc="Train classes"):
        for _ in range(num_train_per_class):
            regime = "fast" if np.random.rand() < fast_ratio else "slow"
            lowlight = bool(np.random.rand() < lowlight_ratio)
            fname = f"{zero_pad(file_idx)}.npz"
            outp = os.path.join(train_dir, fname)
            generate_synthetic_sample(
                class_idx=c,
                num_classes=num_classes,
                output_path=outp,
                image_size=image_size,
                num_frames=num_frames,
                regime=regime,
                lowlight=lowlight
            )
            file_idx += 1
    
    # TEST
    print(f"Generating TEST for {output_dir}")
    for c in tqdm(range(num_classes), desc="Test classes"):
        for _ in range(num_test_per_class):
            regime = "fast" if np.random.rand() < fast_ratio else "slow"
            lowlight = bool(np.random.rand() < lowlight_ratio)
            fname = f"{zero_pad(file_idx)}.npz"
            outp = os.path.join(test_dir, fname)
            generate_synthetic_sample(
                class_idx=c,
                num_classes=num_classes,
                output_path=outp,
                image_size=image_size,
                num_frames=num_frames,
                regime=regime,
                lowlight=lowlight
            )
            file_idx += 1
    
    total = num_classes * (num_train_per_class + num_test_per_class)
    print(f"✓ Done: {output_dir}")
    print(f"  Train: {num_classes * num_train_per_class} | Test: {num_classes * num_test_per_class} | Total files: {total}")

# ----------------------------
# Labels
# ----------------------------
CALTECH_LABELS = [
 'accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain',
 'brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier',
 'cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill',
 'dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo',
 'flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter',
 'ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus',
 'mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda',
 'panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors',
 'scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry',
 'sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background'
]

CIFAR_LABELS = ["frog","horse","dog","truck","airplane","automobile","bird","ship","cat","deer"]

# ----------------------------
# Main / CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate synthetic spike camera data for SpikeCLIP.")
    ap.add_argument('--dataset', type=str, choices=['caltech', 'cifar', 'both'], default='both',
                    help='Which dataset to generate')
    ap.add_argument('--data_root', type=str, default='data', help='Root directory for output')
    ap.add_argument('--num_train', type=int, default=50, help='Training samples per class')
    ap.add_argument('--num_test', type=int, default=10, help='Test samples per class')
    ap.add_argument('--num_frames', type=int, default=200, help='Frames (T) per spike sequence')
    ap.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Latent image size (H W)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--lowlight_ratio', type=float, default=0.5, help='Fraction of low-light samples [0..1]')
    ap.add_argument('--fast_ratio', type=float, default=0.5, help='Fraction of fast-motion samples [0..1]')
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    H, W = tuple(args.image_size)
    if H > 250 or W > 250:
        raise ValueError("image_size should be <= 250x250 due to padding/cropping scheme.")
    
    print("=" * 64)
    print("Synthetic Spike Dataset Generation")
    print("=" * 64)
    print(f" Seed              : {args.seed}")
    print(f" Datasets          : {args.dataset}")
    print(f" Data root         : {args.data_root}")
    print(f" Image size        : {H}x{W}")
    print(f" Frames per sample : {args.num_frames}")
    print(f" Train per class   : {args.num_train}")
    print(f" Test per class    : {args.num_test}")
    print(f" Low-light ratio   : {args.lowlight_ratio}")
    print(f" Fast-motion ratio : {args.fast_ratio}")
    print("=" * 64)
    
    start_idx = 0
    if args.dataset in ['caltech', 'both']:
        cal_dir = os.path.join(args.data_root, 'U-CALTECH')
        generate_dataset(
            output_dir=cal_dir,
            labels=CALTECH_LABELS,
            num_train_per_class=args.num_train,
            num_test_per_class=args.num_test,
            image_size=(H, W),
            num_frames=args.num_frames,
            lowlight_ratio=args.lowlight_ratio,
            fast_ratio=args.fast_ratio,
            seed=args.seed,
            start_index=start_idx
        )
        # keep continuous file numbering across datasets (not required, but tidy)
        start_idx += len(CALTECH_LABELS) * (args.num_train + args.num_test)
    
    if args.dataset in ['cifar', 'both']:
        cif_dir = os.path.join(args.data_root, 'U-CIFAR')
        generate_dataset(
            output_dir=cif_dir,
            labels=CIFAR_LABELS,
            num_train_per_class=args.num_train,
            num_test_per_class=args.num_test,
            image_size=(H, W),
            num_frames=args.num_frames,
            lowlight_ratio=args.lowlight_ratio,
            fast_ratio=args.fast_ratio,
            seed=args.seed,
            start_index=0  # reset numbering for CIFAR folder (consistent with typical repos)
        )
    
    print("\nAll done! You can now run SpikeCLIP code, e.g.:")
    print("  python3 evaluate.py --data_type CALTECH")
    print("or")
    print("  python3 evaluate.py --data_type CIFAR")

if __name__ == '__main__':
    main()
