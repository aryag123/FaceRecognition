# RetinaFace ONNX Download Options

The Google Drive link you found contains MXNet files (`.params` and `.json`), not ONNX. Here are your options:

## Option 1: Use InsightFace Python Package (Recommended)

The InsightFace Python package can automatically download ONNX models:

### Step 1: Install InsightFace
```bash
pip install insightface
```

### Step 2: Run the download script
```bash
python3 download_retinaface_onnx.py
```

This will download the buffalo_l model pack which includes:
- RetinaFace detection model (ONNX format)
- Face recognition model
- Landmark detection model

### Step 3: Copy the detection model
After running the script, copy the detection model:
```bash
cp ~/.insightface/models/buffalo_l/det_*.onnx /Users/Arya/FaceRecognition/models/retinaface.onnx
```

## Option 2: Download from PyTorch RetinaFace Repository

The original RetinaFace PyTorch implementation may have ONNX exports:

1. Visit: https://github.com/biubug6/Pytorch_RetinaFace
2. Check the "Pre-trained Models" section
3. Look for ONNX format models or conversion scripts

## Option 3: Convert MXNet to ONNX

If you have the `.params` and `.json` files, you can convert them:

### Using Python/MXNet:
```python
import mxnet as mx
import onnx
from mxnet.contrib import onnx as onnx_mxnet

# Load MXNet model
sym, arg_params, aux_params = mx.model.load_checkpoint('R50-0000', 0)

# Convert to ONNX (this is simplified - actual conversion may be more complex)
onnx_model = onnx_mxnet.export_model(sym, arg_params, aux_params, 
                                     [1, 3, 640, 640], 
                                     'retinaface.onnx')
```

## Option 4: Use Alternative ONNX Face Detectors

If RetinaFace ONNX is hard to find, consider:

### SCRFD (InsightFace's newer detector)
- Part of buffalo model packs
- More efficient than RetinaFace
- Available as ONNX

### MTCNN ONNX
- Classic face detector
- Available as ONNX from various sources
- Less accurate than RetinaFace but easier to find

## Option 5: Keep Using XML Cascade (For Now)

If finding/downloading ONNX RetinaFace is taking too long:
- Your current XML cascade works fine
- You can switch to RetinaFace later when you have the model
- The code is already set up to support both

## Recommendation

**Use Option 1** (InsightFace Python package) - it's the easiest and most reliable way to get a proper ONNX RetinaFace model.

