# Setting Up InsightFace Model

## Recommended Model: InsightFace ArcFace R100 or R50

### Step 1: Download the Model

**Option A: From InsightFace Model Zoo (Recommended)**
1. Go to: https://github.com/deepinsight/insightface/wiki/Model-Zoo
2. Look for models like:
   - `glint360k_r100.onnx` (512-dim embeddings, best accuracy)
   - `w600k_r50.onnx` (512-dim embeddings, good balance)
   - `glint360k_r50.onnx` (512-dim embeddings)

**Option B: From Hugging Face**
1. Search: https://huggingface.co/models?search=insightface
2. Download a suitable ONNX model

**Option C: Direct download (example)**
```bash
# Example - you'll need to find the actual URL for the model you want
cd /Users/Arya/FaceRecognition/models
# Download model (replace URL with actual download link)
# wget <model_url> -O insightface_r100.onnx
```

### Step 2: Verified Preprocessing for InsightFace

InsightFace models (confirmed from documentation):
- **Input size**: 112x112 pixels
- **Color format**: BGR (OpenCV default - no conversion needed!)
- **Normalization**: (pixel - 127.5) / 128.0 → range approximately [-1, 1]
- **Tensor format**: CHW (Channel-Height-Width)
- **Input tensor name**: Usually "data" or "input" (needs verification)
- **Output dimension**: 512 (for R100/R50 models)

### Step 3: Code Changes Needed

The code should work with minimal changes:
1. Update model path in `FaceRecognitionService.java`
2. Verify input tensor name (might need to check)
3. Verify output dimension matches (512 vs current)

### Current Code Status

Your current preprocessing is:
- ✅ Size: 112x112 (correct)
- ⚠️ Color: Currently RGB, but InsightFace needs BGR
- ⚠️ Normalization: (pixel - 127.5) / 128.0 (correct)
- ✅ Format: CHW (correct)

**Main change needed**: Switch from RGB to BGR (remove RGB conversion)

