# InsightFace R100 Model Setup Guide

## Recommended R100 Models

Based on the [InsightFace Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md), here are the best R100 models for face recognition:

### 1. **R100@WebFace600K_pfc** ⭐ RECOMMENDED (Best Accuracy)
- **Accuracy**: 99.817 LFW, 99.143 CFP-FP, 98.117 AgeDB-30, 97.010 IJB-C
- **MR-ALL**: 89.951
- **Download**: [Google Drive](https://drive.google.com/file/d/11TASXssTnwLY1ZqKlRjsJiV-1nWu9pDY/view?usp=sharing)
- **Why**: Highest accuracy across all benchmarks

### 2. **R100@Glint360K** (Part of antelopev2 model pack)
- **Accuracy**: Excellent performance
- **MR-ALL**: ~91+ (varies by variant)
- **Download**: Available in antelopev2 model pack
- **Why**: Well-tested, widely used

### 3. **R100@MS1MV3**
- **Accuracy**: 84.312 MR-ALL
- **Download**: [Google Drive](https://drive.google.com/file/d/1fZOfvfnavFYjzfFoKTh5j1YDcS8KCnio/view?usp=sharing)
- **Why**: Good general-purpose model

## Step-by-Step Setup

### Step 1: Download the Model

**Option A: R100@WebFace600K_pfc (Recommended)**
```bash
cd /Users/Arya/FaceRecognition/models
# Download from Google Drive link above
# After downloading, rename it to something like:
# mv <downloaded_file>.onnx r100_webface600k.onnx
```

**Option B: Use wget/curl (if you have the direct link)**
```bash
cd /Users/Arya/FaceRecognition/models
# You'll need to extract the actual download link from Google Drive
# Google Drive links require authentication, so you may need to:
# 1. Download manually from the browser
# 2. Or use gdown (Python tool) if you have it installed
```

### Step 2: Verify Preprocessing (Already Correct!)

Your current preprocessing is **already correct** for InsightFace R100 models:

✅ **Input size**: 112x112 pixels (correct)
✅ **Color format**: BGR (OpenCV default - correct)
✅ **Normalization**: `(pixel - 127.5) / 128.0` (correct)
✅ **Tensor format**: CHW (Channel-Height-Width) (correct)

**No preprocessing changes needed!**

### Step 3: Update Model Path

Update the model path in `FaceRecognitionService.java`:

```java
String modelPath = "/Users/Arya/FaceRecognition/models/r100_webface600k.onnx";
```

Or whatever you named the downloaded file.

### Step 4: Verify Model Input/Output

The code will automatically detect the input/output tensor names when you run it. The diagnostic output will show:
- Input tensor name (usually "data" or "input")
- Output tensor name
- Embedding dimension (should be 512 for R100 models)

### Step 5: Test the Model

Run the program and check:
1. Model information is printed correctly
2. Embedding dimension is 512
3. Similarity scores show better discrimination between different people

## Expected Improvements

With the R100@WebFace600K_pfc model, you should see:
- **Better discrimination**: Similarity scores between different people should be lower (e.g., 0.85-0.92 instead of 0.97+)
- **Higher accuracy**: Same-person similarities should remain high (0.95+)
- **More reliable matching**: Clearer distinction between matches and non-matches

## Troubleshooting

### If the model doesn't load:
- Check the file path is correct
- Verify the file is a valid ONNX model
- Check file permissions

### If embeddings are wrong dimension:
- R100 models should output 512-dimensional embeddings
- The code will print the actual dimension in the diagnostic output

### If similarity scores are still too high:
- Make sure you downloaded the correct model (WebFace600K_pfc is best)
- Verify preprocessing is correct (already verified above)
- Check that the model is actually being used (check diagnostic output)

## Model Comparison

| Model | LFW | CFP-FP | AgeDB-30 | IJB-C | MR-ALL |
|-------|-----|--------|----------|-------|--------|
| R100@WebFace600K_pfc | 99.817 | 99.143 | 98.117 | 97.010 | 89.951 |
| R100@Glint360K | ~99.8+ | ~99.0+ | ~98.0+ | ~97.0+ | ~91+ |
| R100@MS1MV3 | - | - | - | - | 84.312 |

**Recommendation**: Use R100@WebFace600K_pfc for the best accuracy.

