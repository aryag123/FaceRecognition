# Quick Guide: Switching to InsightFace Model

## After You Download the Model

Once you download an InsightFace model (e.g., `insightface_r100.onnx`), here's what to do:

### 1. Place the model file
```bash
# Copy your downloaded model to the models folder
cp ~/Downloads/insightface_r100.onnx /Users/Arya/FaceRecognition/models/
```

### 2. Update the code (I can help with this!)

**Changes needed:**
- Update model path in `FaceRecognitionService.java`
- Remove RGB conversion (InsightFace uses BGR)
- Verify input tensor name (might be "data" or "input")
- Check output dimension (should be 512)

### 3. Test it

Run the code and see if discrimination improves!

## Preprocessing for InsightFace

**Confirmed requirements:**
- Size: 112x112 ✅ (already correct)
- Color: **BGR** ⚠️ (currently using RGB - needs change)
- Normalization: (pixel - 127.5) / 128.0 ✅ (already correct)
- Format: CHW ✅ (already correct)

The main fix: **Remove RGB conversion** - use BGR directly from OpenCV.

