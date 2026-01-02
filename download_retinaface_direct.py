#!/usr/bin/env python3
"""
Direct download script for InsightFace buffalo_l model pack (includes RetinaFace ONNX).
This script downloads the models directly without requiring onnxruntime.
"""

import os
import requests
import zipfile
import json

# InsightFace buffalo_l model pack download URL
# This is the official download location
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

def download_file(url, dest_path):
    """Download a file with progress bar."""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print("\nDownload complete!")

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the buffalo_l model pack
    zip_path = os.path.join(models_dir, "buffalo_l.zip")
    
    if not os.path.exists(zip_path):
        try:
            download_file(BUFFALO_L_URL, zip_path)
        except Exception as e:
            print(f"Error downloading: {e}")
            print("\nTrying alternative: You can manually download from:")
            print("https://github.com/deepinsight/insightface/releases")
            return
    else:
        print(f"Zip file already exists: {zip_path}")
    
    # Extract the zip file
    print("\nExtracting models...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error extracting: {e}")
        return
    
    # Find ONNX files
    print("\nLooking for ONNX files...")
    onnx_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.onnx'):
                full_path = os.path.join(root, file)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                onnx_files.append((file, full_path, size_mb))
                print(f"Found: {file}")
                print(f"  Path: {full_path}")
                print(f"  Size: {size_mb:.2f} MB")
                print()
    
    # Find the detection model (usually starts with 'det_' or contains 'detection')
    detection_model = None
    for file, path, size in onnx_files:
        if 'det' in file.lower() or 'detection' in file.lower():
            detection_model = (file, path, size)
            break
    
    if detection_model:
        file, path, size = detection_model
        print(f"\n✓ Detection model found: {file}")
        print(f"  To copy to your project:")
        print(f"  cp '{path}' /Users/Arya/FaceRecognition/models/retinaface.onnx")
    else:
        print("\n⚠ Detection model not automatically identified.")
        print("  Look for files with 'det' or 'detection' in the name.")
        print(f"  Models are in: {models_dir}")

if __name__ == "__main__":
    main()

