#!/usr/bin/env python3
"""
Script to download InsightFace buffalo model pack which includes RetinaFace ONNX models.
Run this to get the ONNX face detection model.
"""

import insightface
import os

# Initialize InsightFace app with buffalo_l model pack
# This will automatically download the models
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])

# The models are downloaded to ~/.insightface/models/buffalo_l/
# Look for the detection model (usually named something like det_10g.onnx or similar)

model_dir = os.path.expanduser("~/.insightface/models/buffalo_l/")
print(f"Models downloaded to: {model_dir}")
print("\nLooking for ONNX files...")

if os.path.exists(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith('.onnx'):
            full_path = os.path.join(model_dir, file)
            print(f"Found: {file}")
            print(f"  Path: {full_path}")
            print(f"  Size: {os.path.getsize(full_path) / (1024*1024):.2f} MB")
            print()

print("\nTo copy the detection model to your project:")
print("cp ~/.insightface/models/buffalo_l/det_*.onnx /Users/Arya/FaceRecognition/models/retinaface.onnx")

