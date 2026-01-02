# Recommendations for Improving Similarity Scores

## Current Situation
- Alignment: ✅ Enabled with estimated landmarks
- Preprocessing: ✅ BGR, correct normalization
- Scores: Still too close (0.9780 vs 0.9721 = 0.006 difference)

## Most Likely Issue: Model Quality

The preprocessing and alignment seem correct, but scores remain close. This suggests **the model itself may not be discriminative enough**.

## Recommended Actions (in order):

### 1. **Try a Better Model** ⭐ HIGHEST PRIORITY
Your current `arcface.onnx` might be:
- Poorly trained
- Wrong variant
- Not suitable for your use case

**Best solution:** Download a proven InsightFace model:
- **Official Buffalo_L**: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
- Requires conversion from MXNet to ONNX (needs Python)

**OR** Find a pre-converted ONNX model on Hugging Face

### 2. **Improve Estimated Landmarks** (Minor improvement)
The current estimated landmarks use simple geometry. You could:
- Fine-tune the eye position percentages
- Use a better estimation algorithm
- But this likely won't fix the core issue

### 3. **Fix Landmark Detector** (If you want better alignment)
The landmark detector produces invalid coordinates. Options:
- Fix the coordinate parsing logic
- Try a different landmark detector model
- Use a 5-point model instead of 132-point

### 4. **Test Different Preprocessing** (Unlikely to help)
You could try:
- RGB instead of BGR
- Different normalization formula
- But current preprocessing seems correct for most ArcFace models

## My Strong Recommendation

**Get a better model.** The preprocessing and alignment are probably fine - the model quality is likely the bottleneck.

Would you like me to:
1. Help you find and download an InsightFace ONNX model?
2. Help convert the official InsightFace model from MXNet to ONNX?
3. Try other preprocessing variations as a last resort?

