# Face Recognition Model Recommendations

## Current Situation
You're using an ArcFace model (`arcface.onnx`) that's not giving good discrimination (scores too close: 0.9744 vs 0.9709).

## Recommended Models (Free, ONNX Format)

### 1. **InsightFace ArcFace R100** ⭐ RECOMMENDED
- **Why**: Well-documented, proven preprocessing, widely used
- **Preprocessing**: BGR, (pixel - 127.5) / 128.0, 112x112
- **Where to get**: 
  - Hugging Face: https://huggingface.co/models?search=insightface
  - Official: https://github.com/deepinsight/insightface
  - Model Zoo: https://github.com/deepinsight/insightface/wiki/Model-Zoo
- **Input**: 112x112, BGR, normalized
- **Output**: 512-dim embedding (L2 normalized)

### 2. **FaceNet (Inception ResNet v1)**
- **Why**: Classic model, well-documented
- **Preprocessing**: RGB, normalized [0,1], 160x160
- **Where to get**: ONNX Model Zoo
- **Input**: 160x160, RGB
- **Output**: 512-dim embedding

### 3. **VGGFace2**
- **Why**: Good alternative, well-tested
- **Preprocessing**: BGR, mean subtraction
- **Where to get**: ONNX Model Zoo

### 4. **MobileFaceNet** (if you need smaller/faster)
- **Why**: Lightweight, good accuracy
- **Preprocessing**: BGR, (pixel - 127.5) / 128.0
- **Input**: 112x112

## Quick Start with InsightFace R100

1. **Download the model:**
   ```bash
   # From Hugging Face or InsightFace Model Zoo
   # Look for: "w600k_r50.onnx" or "glint360k_r50.onnx" or similar
   ```

2. **Preprocessing for InsightFace (confirmed):**
   - Input size: 112x112
   - Color: BGR (OpenCV default)
   - Normalization: (pixel - 127.5) / 128.0
   - Format: CHW (Channel-Height-Width)

3. **What needs to change in code:**
   - Update model path
   - Verify input tensor name (might be "data" or "input")
   - Verify output dimension (usually 512)
   - Preprocessing should already be mostly correct

## Recommendation

**Try InsightFace R100 or R50 model** because:
- ✅ Clear, documented preprocessing requirements
- ✅ Proven to work well
- ✅ Available as ONNX
- ✅ Good discrimination between faces
- ✅ Widely used in production

Would you like me to:
1. Help you find and download an InsightFace model?
2. Modify the code to work with InsightFace (minimal changes needed)?
3. Test with the new model once you download it?

