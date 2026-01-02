# RetinaFace ONNX Setup Guide

## Overview

RetinaFace is a more accurate face detection model than Haar Cascade. This guide shows you how to set it up.

## Step 1: Download ONNX RetinaFace Model

### Option A: InsightFace RetinaFace (Recommended)

From the [InsightFace Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md):

**RetinaFace-R50 ONNX:**
- **Google Drive**: https://drive.google.com/file/d/1wm-6K688HQEx_H90UdAIuKv-NAsKBu85/view?usp=sharing
- **Baidu Drive**: https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ

**Download Steps:**
1. Open the Google Drive link in your browser
2. Download the ONNX file
3. Save it to `/Users/Arya/FaceRecognition/models/` (e.g., `retinaface_r50.onnx`)

### Option B: Other RetinaFace ONNX Models

You can also find RetinaFace ONNX models from:
- Hugging Face: https://huggingface.co/models?search=retinaface
- ONNX Model Zoo
- Other model repositories

## Step 2: Update Your Code

### Option 1: Use ONNX RetinaFace (Recommended)

In `FaceRecognitionService.java`, replace the cascade initialization:

```java
// OLD (XML Cascade):
String cascadePath = "/Users/Arya/FaceRecognition/models/haarcascade_frontalface_default.xml";
ImagePreprocessor.initFaceDetector(cascadePath);

// NEW (ONNX RetinaFace):
String retinaFacePath = "/Users/Arya/FaceRecognition/models/retinaface_r50.onnx";
ImagePreprocessor.initOnnxFaceDetector(retinaFacePath);
```

### Option 2: Keep Using XML Cascade

If you want to keep using the XML cascade (simpler, but less accurate), no changes needed.

## Step 3: Cleanup (Important!)

If you use ONNX RetinaFace, make sure to close it in the finally block:

```java
try (FaceRecognitionService service = new FaceRecognitionService(modelPath)) {
    // ... your code ...
} finally {
    // Cleanup ONNX face detector if used
    ImagePreprocessor.closeOnnxFaceDetector();
    // Cleanup landmark detector if used
    ImagePreprocessor.closeLandmarkDetector();
}
```

## Benefits of RetinaFace

✅ **More Accurate**: Better at detecting faces in various conditions
✅ **Better with Multiple Faces**: Handles multiple faces more reliably
✅ **Better with Angles**: Works better with side profiles and angles
✅ **Better with Small Faces**: Can detect smaller faces more reliably

## Troubleshooting

### Model doesn't load
- Check the file path is correct
- Verify the file is a valid ONNX model
- Check file permissions

### Wrong output format
- The code tries to handle different RetinaFace ONNX export formats
- If detection doesn't work, check the console output for model input/output names
- You may need to adjust the parsing logic in `ONNXFaceDetector.parseRetinaFaceOutputs()`

### Coordinates are wrong
- RetinaFace outputs may be normalized (0-1) or absolute
- The code tries to detect and handle both formats
- If faces are detected but in wrong locations, check the coordinate scaling logic

## Example Usage

```java
public static void main(String[] args) throws Exception {
    String modelPath = "/Users/Arya/FaceRecognition/models/model.onnx";
    String retinaFacePath = "/Users/Arya/FaceRecognition/models/retinaface_r50.onnx";
    String referenceFolderPath = "/Users/Arya/FaceRecognition/ReferencePhotos";
    String inputPhotoPath = "/Users/Arya/Downloads/arya1.jpg";

    // Initialize ONNX RetinaFace detector
    System.out.println("Initializing RetinaFace detector...");
    ImagePreprocessor.initOnnxFaceDetector(retinaFacePath);

    // ... rest of your code ...
    
    try (FaceRecognitionService service = new FaceRecognitionService(modelPath)) {
        // Process images...
    } finally {
        ImagePreprocessor.closeOnnxFaceDetector();
        ImagePreprocessor.closeLandmarkDetector();
    }
}
```

