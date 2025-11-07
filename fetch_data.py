#!/usr/bin/env python3
"""
Script to download and prepare U-CALTECH and U-CIFAR datasets.
The UHSR dataset is available at: https://github.com/Evin-X/UHSR
"""

import os
import argparse
import urllib.request
import urllib.error
import zipfile
import shutil
import numpy as np
from tqdm import tqdm
import hashlib
import re
import subprocess
import json

def download_file(url, dest_path, desc=None, chunk_size=8192):
    """Download a file with progress bar"""
    try:
        if desc:
            print(f"Downloading {desc}...")
        
        # Get file size
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))
            else:
                out_file.write(response.read())
        
        print(f"✓ Downloaded to {dest_path}")
        return True
    except urllib.error.HTTPError as e:
        print(f"✗ HTTP Error {e.code}: {e.reason}")
        print(f"  URL: {url}")
        return False
    except urllib.error.URLError as e:
        print(f"✗ URL Error: {e.reason}")
        print(f"  URL: {url}")
        return False
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip or tar file"""
    print(f"Extracting {zip_path}...")
    if zip_path.endswith('.zip'):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif zip_path.endswith('.tar') or zip_path.endswith('.tar.gz'):
        import tarfile
        mode = 'r:gz' if zip_path.endswith('.tar.gz') else 'r'
        with tarfile.open(zip_path, mode) as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"⚠️  Unknown archive format: {zip_path}")
        return False
    print(f"✓ Extracted to {extract_to}")
    return True

def prepare_caltech_dataset(data_dir, train_range=(0, 4999), test_start=5000):
    """
    Prepare U-CALTECH dataset by splitting data into train/test
    Expected structure: data_dir contains .npz files with 'spk' and 'label' keys
    """
    print("Preparing U-CALTECH dataset...")
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Find all .npz files
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    npz_files = sorted(npz_files)
    
    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        print("Please ensure you have downloaded and extracted the UHSR dataset.")
        return False
    
    print(f"Found {len(npz_files)} files. Splitting into train/test...")
    
    for i, filename in enumerate(tqdm(npz_files)):
        src_path = os.path.join(data_dir, filename)
        
        if train_range[0] <= i < train_range[1]:
            dest_path = os.path.join(train_dir, filename)
        elif i >= test_start:
            dest_path = os.path.join(test_dir, filename)
        else:
            continue
        
        shutil.copy2(src_path, dest_path)
    
    print(f"Created train set with files {train_range[0]}-{train_range[1]}")
    print(f"Created test set with files {test_start}-{len(npz_files)}")
    return True

def prepare_cifar_dataset(data_dir):
    """Prepare U-CIFAR dataset"""
    print("Preparing U-CIFAR dataset...")
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Similar structure to CALTECH
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    npz_files = sorted(npz_files)
    
    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return False
    
    # Split 80/20 for CIFAR
    split_idx = int(0.8 * len(npz_files))
    
    for i, filename in enumerate(tqdm(npz_files)):
        src_path = os.path.join(data_dir, filename)
        
        if i < split_idx:
            dest_path = os.path.join(train_dir, filename)
        else:
            dest_path = os.path.join(test_dir, filename)
        
        shutil.copy2(src_path, dest_path)
    
    print(f"Created train/test split: {split_idx}/{len(npz_files) - split_idx}")
    return True

def extract_baidu_drive_link(github_url):
    """
    Extract Baidu Drive link from GitHub README.
    This function attempts to fetch the README and find Baidu Drive links.
    """
    try:
        print(f"🔍 Fetching GitHub README from {github_url}...")
        req = urllib.request.Request(github_url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            
        # Pattern to match Baidu Drive links (pan.baidu.com)
        baidu_patterns = [
            r'https?://pan\.baidu\.com/[^\s\)]+',
            r'https?://yun\.baidu\.com/[^\s\)]+',
            r'pan\.baidu\.com/s/[a-zA-Z0-9_-]+',
        ]
        
        baidu_links = []
        for pattern in baidu_patterns:
            matches = re.findall(pattern, content)
            baidu_links.extend(matches)
        
        if baidu_links:
            # Remove duplicates
            baidu_links = list(set(baidu_links))
            print(f"✓ Found {len(baidu_links)} Baidu Drive link(s)")
            return baidu_links[0]  # Return first link
        else:
            print("⚠️  No Baidu Drive links found in README")
            return None
    except Exception as e:
        print(f"⚠️  Error fetching GitHub README: {e}")
        return None

def get_baidu_drive_download_info(baidu_url):
    """
    Get download information from Baidu Drive share link.
    Note: Baidu Drive requires special handling. This function provides guidance.
    """
    print("\n" + "=" * 60)
    print("Baidu Drive Download Guide")
    print("=" * 60)
    print(f"\n📥 Baidu Drive Link: {baidu_url}")
    print("\nTo download from Baidu Drive:")
    print("\nOption 1: Manual Download (Recommended)")
    print("  1. Open the link in your browser")
    print("  2. Download the zip files manually")
    print("  3. Use: python3 fetch_data.py --download_file <path_to_zip> --dataset all")
    print("\nOption 2: Extract Direct Download Link")
    print("  Baidu Drive share links can be converted to direct download links")
    print("  using tools like:")
    print("  - BaiduPanDownloadHelper")
    print("  - Or use --download_url with the direct link if available")
    print("\nOption 3: Use Baidu Pan API (if you have access)")
    print("=" * 60)
    return None

def download_from_github_repo(github_repo_url, dest_dir='data'):
    """
    Attempt to extract download links from GitHub repository README.
    Handles Baidu Drive links.
    """
    print(f"\n🔍 Analyzing GitHub repository: {github_repo_url}")
    
    # Try to get README
    readme_urls = [
        f"{github_repo_url}/blob/main/README.md",
        f"{github_repo_url}/blob/master/README.md",
        f"{github_repo_url}/raw/main/README.md",
        f"{github_repo_url}/raw/master/README.md",
    ]
    
    baidu_link = None
    for readme_url in readme_urls:
        baidu_link = extract_baidu_drive_link(readme_url)
        if baidu_link:
            break
    
    if baidu_link:
        get_baidu_drive_download_info(baidu_link)
        return baidu_link
    else:
        print("\n⚠️  Could not automatically extract download links")
        print("Please check the GitHub repository README manually")
        return None

def main():
    parser = argparse.ArgumentParser(description='Fetch and prepare SpikeCLIP datasets')
    parser.add_argument('--dataset', type=str, choices=['caltech', 'cifar', 'all'], 
                       default='all', help='Dataset to prepare')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for data')
    parser.add_argument('--manual_download', action='store_true',
                       help='Skip automatic download attempt, only organize existing data')
    parser.add_argument('--download_url', type=str, default=None,
                       help='Direct download URL for dataset (zip/tar file)')
    parser.add_argument('--download_file', type=str, default=None,
                       help='Local path to downloaded dataset file')
    parser.add_argument('--github_repo', type=str, default=None,
                       help='GitHub repository URL to extract download links from (e.g., https://github.com/Evin-X/UHSR)')
    parser.add_argument('--baidu_link', type=str, default=None,
                       help='Direct Baidu Drive share link')
    
    args = parser.parse_args()
    
    data_root = args.data_root
    os.makedirs(data_root, exist_ok=True)
    
    print("=" * 60)
    print("SpikeCLIP Dataset Fetching & Preparation")
    print("=" * 60)
    
    # Attempt to extract links from GitHub repo
    if args.github_repo:
        print("\n" + "=" * 60)
        print("GitHub Repository Analysis")
        print("=" * 60)
        baidu_link = download_from_github_repo(args.github_repo, data_root)
        if baidu_link:
            print(f"\n💡 Use this Baidu Drive link: {baidu_link}")
            print("   Follow the instructions above to download manually")
    
    # Handle Baidu Drive link
    if args.baidu_link:
        get_baidu_drive_download_info(args.baidu_link)
    
    # Attempt automatic download if URL or file provided
    downloaded_file = None
    if args.download_url:
        print(f"\n📥 Downloading from URL: {args.download_url}")
        filename = os.path.basename(args.download_url) or "dataset.zip"
        # Handle query parameters in URL
        if '?' in filename:
            filename = filename.split('?')[0]
        dest_path = os.path.join(data_root, filename)
        if download_file(args.download_url, dest_path, desc=filename):
            downloaded_file = dest_path
    elif args.download_file:
        if os.path.exists(args.download_file):
            print(f"\n✓ Using local file: {args.download_file}")
            downloaded_file = args.download_file
        else:
            print(f"\n✗ File not found: {args.download_file}")
    
    # Extract if downloaded file is an archive
    if downloaded_file and (downloaded_file.endswith('.zip') or downloaded_file.endswith('.tar') or downloaded_file.endswith('.tar.gz')):
        print(f"\n📦 Extracting archive: {downloaded_file}")
        extract_to = os.path.join(data_root, 'extracted')
        os.makedirs(extract_to, exist_ok=True)
        if extract_zip(downloaded_file, extract_to):
            print("✓ Extraction complete")
            print("\n📁 Next steps:")
            print("  1. Check the extracted files in:", extract_to)
            print("  2. Move .npz files to data/U-CALTECH/ or data/U-CIFAR/")
            print("  3. Run: python3 fetch_data.py --manual_download --dataset all")
    
    # Attempt GitHub download if no manual download flag
    if not args.manual_download and not downloaded_file and not args.github_repo and not args.baidu_link:
        print("\n" + "=" * 60)
        print("Download Options")
        print("=" * 60)
        print("\n📝 The UHSR dataset is available at:")
        print("   https://github.com/Evin-X/UHSR")
        print("\n💡 Quick Start (Baidu Drive):")
        print("   python3 fetch_data.py --github_repo https://github.com/Evin-X/UHSR")
        print("   This will extract Baidu Drive links from the README")
        print("\n💡 Other options:")
        print("   python3 fetch_data.py --download_url <direct_download_link>")
        print("   python3 fetch_data.py --download_file <path_to_local_zip>")
        print("   python3 fetch_data.py --manual_download  (organize existing files)")
        print("\n📖 For Baidu Drive help, see: baidu_drive_helper.md")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Dataset Organization")
    print("=" * 60)
    print("This script will organize data into train/test splits.")
    print("=" * 60)
    
    if args.manual_download or args.dataset in ['caltech', 'all']:
        caltech_dir = os.path.join(data_root, 'U-CALTECH')
        if os.path.exists(caltech_dir):
            print(f"\nFound U-CALTECH directory at {caltech_dir}")
            # Check if files exist at root level or subdirectories
            npz_files_root = [f for f in os.listdir(caltech_dir) if f.endswith('.npz')]
            npz_files_total = len(npz_files_root)
            
            # Check subdirectories
            for subdir in ['train', 'test']:
                subdir_path = os.path.join(caltech_dir, subdir)
                if os.path.exists(subdir_path):
                    npz_files_total += len([f for f in os.listdir(subdir_path) if f.endswith('.npz')])
            
            if npz_files_total == 0:
                print(f"⚠️  No .npz files found in {caltech_dir}")
                print("\n📥 To download the dataset:")
                print("   1. Visit: https://github.com/Evin-X/UHSR")
                print("   2. Download the U-CALTECH dataset files")
                print("   3. Extract .npz files to: data/U-CALTECH/")
                print("\n   OR place them directly in the directory structure:")
                print("      data/U-CALTECH/*.npz")
                print("\n   The script will organize them into train/ and test/ subdirectories.")
            elif prepare_caltech_dataset(caltech_dir):
                print("✓ U-CALTECH dataset prepared successfully")
            else:
                print("✗ Failed to prepare U-CALTECH dataset")
        else:
            print(f"\n📁 U-CALTECH directory not found at {caltech_dir}")
            print("Creating directory...")
            os.makedirs(caltech_dir, exist_ok=True)
            print("\n📥 Download the dataset:")
            print("   1. Visit: https://github.com/Evin-X/UHSR")
            print("   2. Download the U-CALTECH dataset")
            print(f"   3. Extract .npz files to: {caltech_dir}/")
            print("   4. Run this script again to organize into train/test splits")
    
    if args.manual_download or args.dataset in ['cifar', 'all']:
        cifar_dir = os.path.join(data_root, 'U-CIFAR')
        if os.path.exists(cifar_dir):
            print(f"\nFound U-CIFAR directory at {cifar_dir}")
            npz_files_root = [f for f in os.listdir(cifar_dir) if f.endswith('.npz')]
            npz_files_total = len(npz_files_root)
            
            for subdir in ['train', 'test']:
                subdir_path = os.path.join(cifar_dir, subdir)
                if os.path.exists(subdir_path):
                    npz_files_total += len([f for f in os.listdir(subdir_path) if f.endswith('.npz')])
            
            if npz_files_total == 0:
                print(f"⚠️  No .npz files found in {cifar_dir}")
                print("\n📥 To download the dataset:")
                print("   1. Visit: https://github.com/Evin-X/UHSR")
                print("   2. Download the U-CIFAR dataset files")
                print(f"   3. Extract .npz files to: {cifar_dir}/")
            elif prepare_cifar_dataset(cifar_dir):
                print("✓ U-CIFAR dataset prepared successfully")
            else:
                print("✗ Failed to prepare U-CIFAR dataset")
        else:
            print(f"\n📁 U-CIFAR directory not found at {cifar_dir}")
            print("Creating directory...")
            os.makedirs(cifar_dir, exist_ok=True)
            print("\n📥 Download the dataset:")
            print("   1. Visit: https://github.com/Evin-X/UHSR")
            print("   2. Download the U-CIFAR dataset")
            print(f"   3. Extract .npz files to: {cifar_dir}/")
            print("   4. Run this script again to organize into train/test splits")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print("\nExpected structure:")
    print(f"{data_root}/")
    print("├── U-CALTECH/")
    print("│   ├── train/")
    print("│   └── test/")
    print("└── U-CIFAR/")
    print("    ├── train/")
    print("    └── test/")
    print("\n" + "=" * 60)
    print("📝 Usage Examples:")
    print("=" * 60)
    print("1. Download from URL and organize:")
    print("   python3 fetch_data.py --download_url <url> --dataset all")
    print("\n2. Organize already downloaded files:")
    print("   python3 fetch_data.py --manual_download --dataset all")
    print("\n3. Generate synthetic data (for testing):")
    print("   python3 generate_synthetic_data.py --dataset both")
    print("\n4. Get download help:")
    print("   ./download_data.sh")
    print("=" * 60)

if __name__ == '__main__':
    main()

