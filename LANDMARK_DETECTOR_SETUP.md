# Facial Landmark Detector Setup

The face recognition system now supports optional facial landmark detection for improved face alignment accuracy.

## Current Status

Currently, the system uses **estimated landmarks** based on face geometry. This works but may not be as accurate as real landmark detection.

## How to Use a Landmark Detector Model

1. **Download a landmark detector ONNX model**:
   - Look for 5-point or 68-point facial landmark detection models in ONNX format
   - Popular options include models from InsightFace, MTCNN, or other face recognition frameworks
   - Example: Search for "face landmark detector onnx 5 point" or "facial landmark detection onnx model"

2. **Place the model file** in your `models/` directory

3. **Update the model path** in `FaceRecognitionService.java`:
   ```java
   // Change from:
   String landmarkModelPath = null;
   
   // To:
   String landmarkModelPath = "/Users/Arya/FaceRecognition/models/your_landmark_model.onnx";
   ```

4. **Run the program** - it will automatically use the landmark detector if the path is set

## Model Requirements

- Format: ONNX (.onnx file)
- Input: RGB image, typically 112x112 or 128x128
- Output: Array of landmark coordinates (x, y pairs)
  - 5-point models: [left_eye, right_eye, nose, left_mouth, right_mouth]
  - 68-point models: Full facial feature points
- Input tensor name: "input" or "data" (automatically detected)

## Benefits

Using a real landmark detector instead of estimated landmarks will:
- Improve face alignment accuracy
- Better handle rotated or tilted faces
- Improve overall face recognition accuracy
- Reduce false matches between different people

## Note

The system gracefully falls back to estimated landmarks if:
- No landmark model path is specified
- The landmark model fails to load
- Landmark detection fails for a particular image

