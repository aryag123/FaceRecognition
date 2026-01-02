# Trustworthy InsightFace ONNX Model Download

## ⚠️ Important Finding

**Official InsightFace models are in MXNet format, not ONNX.** They need to be converted.

## Best Options:

### Option 1: Official Models + Convert (Most Trustworthy) ⭐

1. **Download official model:**
   ```bash
   # Buffalo_L (Best accuracy)
   wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   unzip buffalo_l.zip
   ```

2. **Convert to ONNX using Python:**
   ```python
   # Requires: pip install insightface onnxruntime
   import insightface
   app = insightface.app.FaceAnalysis()
   app.prepare(ctx_id=-1)  # CPU mode
   # Model will be in ~/.insightface/models/buffalo_l/
   ```

### Option 2: Hugging Face ONNX Models (Community, but widely used)

Search Hugging Face for pre-converted ONNX models:
- https://huggingface.co/models?search=insightface+onnx
- Look for models with many downloads/stars
- **Recommendation**: Search for "buffalo_l_onnx" or similar

### Option 3: Keep Your Current Model

Your `arcface.onnx` model might actually be fine. The preprocessing is now correct (BGR, proper normalization). The issue might be:
- Poor model quality (unlikely if it's from a good source)
- Still need better alignment (landmark detection broken)
- Need more diverse training data in the model

## My Recommendation:

Since you're using Java and need ONNX directly, try **Option 2** (Hugging Face) for the easiest path, OR **stick with your current model** and continue debugging the alignment issue.

Would you like me to:
1. Help find a specific Hugging Face model link?
2. Help set up conversion from MXNet to ONNX?
3. Continue optimizing your current model setup?

