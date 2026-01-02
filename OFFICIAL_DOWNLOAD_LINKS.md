# Official InsightFace Model Download Links

## ⚠️ Important Note

The official InsightFace releases provide **MXNet models** (not ONNX), but they can be converted to ONNX. However, since you need ONNX format, I've found some options:

## Option 1: Official InsightFace Buffalo Models (Recommended) ⭐

**Official GitHub Release (v0.7):**
- **Buffalo_L** (Large - Best accuracy): https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
- **Buffalo_S** (Small - Faster): https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip

⚠️ These are ZIP files containing MXNet models. You would need to:
1. Extract the ZIP
2. Convert from MXNet to ONNX (requires Python + mxnet + onnxruntime)
3. Use the ONNX model

## Option 2: Pre-converted ONNX Models (Easier)

Since you need ONNX directly, here are trusted sources:

### Hugging Face (Community Maintained)
- Search: https://huggingface.co/models?search=insightface+onnx
- Many community members have converted and uploaded ONNX versions
- Look for models with many downloads/stars

### ONNX Model Zoo
- Check: https://github.com/onnx/models (may have face recognition models)

## Option 3: Use Your Current ArcFace Model with Better Preprocessing

Actually, your current `arcface.onnx` model might work fine - the issue might just be preprocessing. Since we've already fixed the preprocessing (BGR, correct normalization), you might want to test if a different model architecture helps.

## My Recommendation

**For simplicity and trustworthiness:**
1. Try using a pre-converted ONNX model from Hugging Face (look for one with many stars/downloads)
2. OR keep your current model and see if better preprocessing helps

Would you like me to:
- Help you find a specific Hugging Face model link?
- Help you convert the official Buffalo model to ONNX (requires Python setup)?
- Continue debugging your current model?

