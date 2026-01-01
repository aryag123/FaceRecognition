// ===== ONNXFaceDetector.java =====
package com.example;

import ai.onnxruntime.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Face detector using ONNX models (e.g., RetinaFace).
 * This is more accurate than Haar Cascade but requires an ONNX model file.
 */
public class ONNXFaceDetector implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Object lock = new Object();
    private final int inputWidth;
    private final int inputHeight;
    private final double scoreThreshold;

    public ONNXFaceDetector(String onnxModelPath) throws OrtException {
        this(onnxModelPath, 640, 640, 0.5); // Default: 640x640 input, 0.5 score threshold
    }

    public ONNXFaceDetector(String onnxModelPath, int inputWidth, int inputHeight, double scoreThreshold) throws OrtException {
        if (onnxModelPath == null || onnxModelPath.isEmpty()) {
            throw new IllegalArgumentException("ONNX model path cannot be null or empty");
        }

        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(onnxModelPath, new OrtSession.SessionOptions());
        
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.scoreThreshold = scoreThreshold;
        
        System.out.println("ONNX Face Detector initialized:");
        System.out.println("  Model: " + onnxModelPath);
        System.out.println("  Input names: " + session.getInputNames());
        System.out.println("  Output names: " + session.getOutputNames());
        System.out.println("  Input size: " + inputWidth + "x" + inputHeight);
        System.out.println("  Score threshold: " + scoreThreshold);
    }

    /**
     * Detects faces in an image using the ONNX model.
     * @param image The input image (BGR format)
     * @return List of detected face rectangles, or empty list if none found
     */
    public List<Rect> detectFaces(Mat image) throws OrtException {
        if (image == null || image.empty()) {
            throw new IllegalArgumentException("Image cannot be null or empty");
        }

        int originalWidth = image.cols();
        int originalHeight = image.rows();
        double scaleX = (double) inputWidth / originalWidth;
        double scaleY = (double) inputHeight / originalHeight;

        synchronized (lock) {
            // Resize image to model input size
            Mat resized = new Mat();
            opencv_imgproc.resize(image, resized, new Size(inputWidth, inputHeight));
            
            // Convert BGR to RGB and normalize
            Mat rgb = new Mat();
            opencv_imgproc.cvtColor(resized, rgb, opencv_imgproc.COLOR_BGR2RGB);
            
            // Normalize: (pixel / 255.0) - typical for RetinaFace
            Mat normalized = new Mat();
            rgb.convertTo(normalized, opencv_core.CV_32FC3, 1.0 / 255.0, 0.0);
            
            try {
                // Convert to CHW format
                int h = normalized.rows();
                int w = normalized.cols();
                int c = normalized.channels();
                
                float[] imageData = new float[c * h * w];
                FloatIndexer indexer = normalized.createIndexer();
                
                int idx = 0;
                for (int ch = 0; ch < c; ch++) {
                    for (int row = 0; row < h; row++) {
                        for (int col = 0; col < w; col++) {
                            imageData[idx++] = indexer.get(row, col, ch);
                        }
                    }
                }
                indexer.close();
                
                // Create input tensor
                long[] shape = new long[]{1, c, h, w};
                try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), shape)) {
                    // Get input name
                    String inputName = session.getInputNames().iterator().next();
                    
                    // Run inference
                    try (OrtSession.Result result = session.run(
                            java.util.Collections.singletonMap(inputName, tensor))) {
                        
                        return parseRetinaFaceOutputs(result, originalWidth, originalHeight, scaleX, scaleY);
                    }
                }
            } finally {
                resized.release();
                rgb.release();
                normalized.release();
            }
        }
    }

    /**
     * Parses RetinaFace model outputs and converts to face rectangles.
     * Handles different output formats that various ONNX exports might use.
     */
    private List<Rect> parseRetinaFaceOutputs(OrtSession.Result result, 
                                             int originalWidth, int originalHeight,
                                             double scaleX, double scaleY) throws OrtException {
        List<Rect> faces = new ArrayList<>();
        
        // RetinaFace outputs vary by export, but typically include:
        // - boxes: [batch, num_detections, 4] or [num_detections, 4]
        // - scores: [batch, num_detections] or [num_detections]
        // - landmarks: [batch, num_detections, 10] or [num_detections, 10] (optional)
        
        // Get output names
        java.util.Set<String> outputNames = session.getOutputNames();
        java.util.List<String> outputNameList = new java.util.ArrayList<>(outputNames);
        
        if (outputNameList.isEmpty()) {
            System.err.println("Warning: No outputs found in RetinaFace model");
            return faces;
        }
        
        // Debug: Print output info
        System.out.println("  RetinaFace outputs: " + outputNameList.size() + " outputs");
        for (int i = 0; i < Math.min(3, outputNameList.size()); i++) {
            String name = outputNameList.get(i);
            OnnxValue val = result.get(name).orElse(null);
            if (val != null) {
                Object obj = val.getValue();
                if (obj instanceof float[]) {
                    System.out.println("    Output " + i + " (" + name + "): float[] length=" + ((float[])obj).length);
                } else if (obj instanceof float[][]) {
                    float[][] arr = (float[][])obj;
                    System.out.println("    Output " + i + " (" + name + "): float[][] shape=[" + arr.length + "," + (arr.length > 0 ? arr[0].length : 0) + "]");
                } else if (obj instanceof float[][][]) {
                    float[][][] arr = (float[][][])obj;
                    System.out.println("    Output " + i + " (" + name + "): float[][][] shape=[" + arr.length + "," + (arr.length > 0 ? arr[0].length : 0) + "," + (arr.length > 0 && arr[0].length > 0 ? arr[0][0].length : 0) + "]");
                }
            }
        }
        
        // RetinaFace typically has multiple outputs for different scales
        // The buffalo_l det_10g model has 9 outputs representing feature maps
        // We need to decode these feature maps into boxes and scores
        // For now, let's try a simpler approach: use the InsightFace Python library's approach
        // or check if there's a combined output
        
        // Try to find a combined output or use the first few outputs
        // RetinaFace outputs are typically: [loc_branch, conf_branch, landm_branch] for each scale
        // So outputs 0,3,6 might be location/boxes, outputs 1,4,7 might be confidence/scores
        
        OnnxValue boxesValue = null;
        OnnxValue scoresValue = null;
        
        // Try to find by name first
        for (String outputName : outputNameList) {
            String lowerName = outputName.toLowerCase();
            if (lowerName.contains("box") || lowerName.contains("loc")) {
                boxesValue = result.get(outputName).orElse(null);
            } else if (lowerName.contains("score") || lowerName.contains("conf")) {
                scoresValue = result.get(outputName).orElse(null);
            }
        }
        
        // If not found by name and we have multiple outputs, try pattern matching
        // For 9 outputs: typically [loc0, conf0, landm0, loc1, conf1, landm1, loc2, conf2, landm2]
        if (boxesValue == null && outputNameList.size() >= 1) {
            boxesValue = result.get(outputNameList.get(0)).orElse(null);
        }
        if (scoresValue == null && outputNameList.size() >= 2) {
            scoresValue = result.get(outputNameList.get(1)).orElse(null);
        }
        
        if (boxesValue == null) {
            System.err.println("Warning: Could not find boxes output in RetinaFace model");
            return faces;
        }
        
        // Parse boxes
        Object boxesObj = boxesValue.getValue();
        float[][] boxes = null;
        
        if (boxesObj instanceof float[][][]) {
            // Shape: [batch, num_detections, 4]
            float[][][] boxes3d = (float[][][]) boxesObj;
            boxes = boxes3d[0]; // Get first batch
        } else if (boxesObj instanceof float[][]) {
            // Shape: [num_detections, 4]
            boxes = (float[][]) boxesObj;
        } else if (boxesObj instanceof float[]) {
            // Shape: [num_detections * 4] - reshape it
            float[] boxesFlat = (float[]) boxesObj;
            int numDetections = boxesFlat.length / 4;
            boxes = new float[numDetections][4];
            for (int i = 0; i < numDetections; i++) {
                boxes[i][0] = boxesFlat[i * 4];
                boxes[i][1] = boxesFlat[i * 4 + 1];
                boxes[i][2] = boxesFlat[i * 4 + 2];
                boxes[i][3] = boxesFlat[i * 4 + 3];
            }
        }
        
        if (boxes == null) {
            return faces;
        }
        
        // Parse scores if available
        float[] scores = null;
        if (scoresValue != null) {
            Object scoresObj = scoresValue.getValue();
            if (scoresObj instanceof float[][]) {
                float[][] scores2d = (float[][]) scoresObj;
                scores = scores2d.length > 0 ? scores2d[0] : null;
            } else if (scoresObj instanceof float[]) {
                scores = (float[]) scoresObj;
            }
        }
        
        // Convert boxes to rectangles
        for (int i = 0; i < boxes.length; i++) {
            // Check score threshold if scores are available
            if (scores != null && i < scores.length && scores[i] < scoreThreshold) {
                continue;
            }
            
            // Box format: [x1, y1, x2, y2] (normalized or absolute)
            float x1 = boxes[i][0];
            float y1 = boxes[i][1];
            float x2 = boxes[i][2];
            float y2 = boxes[i][3];
            
            // Check if coordinates are normalized (0-1) or absolute
            // If normalized, scale them; if absolute, they might be in input image coordinates
            int rectX, rectY, rectWidth, rectHeight;
            
            if (x1 <= 1.0 && y1 <= 1.0 && x2 <= 1.0 && y2 <= 1.0) {
                // Normalized coordinates - scale to original image
                rectX = (int) (x1 * originalWidth);
                rectY = (int) (y1 * originalHeight);
                rectWidth = (int) ((x2 - x1) * originalWidth);
                rectHeight = (int) ((y2 - y1) * originalHeight);
            } else {
                // Absolute coordinates - might be in input image size, scale back
                rectX = (int) (x1 / scaleX);
                rectY = (int) (y1 / scaleY);
                rectWidth = (int) ((x2 - x1) / scaleX);
                rectHeight = (int) ((y2 - y1) / scaleY);
            }
            
            // Ensure valid rectangle
            if (rectWidth > 0 && rectHeight > 0 && 
                rectX >= 0 && rectY >= 0 &&
                rectX + rectWidth <= originalWidth &&
                rectY + rectHeight <= originalHeight) {
                faces.add(new Rect(rectX, rectY, rectWidth, rectHeight));
            }
        }
        
        return faces;
    }

    @Override
    public void close() throws OrtException {
        synchronized (lock) {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        }
    }
}

