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
        // Calculate scale factors for resizing
        double scaleX = (double) inputWidth / originalWidth;
        double scaleY = (double) inputHeight / originalHeight;
        // Use the smaller scale to maintain aspect ratio (letterbox style)
        double scale = Math.min(scaleX, scaleY);

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
     * Parses RetinaFace model outputs using anchor decoding.
     * RetinaFace outputs feature maps that need to be decoded using anchor boxes.
     */
    private List<Rect> parseRetinaFaceOutputs(OrtSession.Result result, 
                                             int originalWidth, int originalHeight,
                                             double scaleX, double scaleY) throws OrtException {
        List<Rect> faces = new ArrayList<>();
        
        // Get output names
        java.util.Set<String> outputNames = session.getOutputNames();
        java.util.List<String> outputNameList = new java.util.ArrayList<>(outputNames);
        
        if (outputNameList.size() < 6) {
            System.err.println("Warning: RetinaFace model should have at least 6 outputs (got " + outputNameList.size() + ")");
            return faces;
        }
        
        // RetinaFace typically has 9 outputs: [loc0, conf0, landm0, loc1, conf1, landm1, loc2, conf2, landm2]
        // For 3 scales, each with location, confidence, and landmarks
        // We'll decode all scales and combine results
        
        // Inspect all outputs to understand their shapes and map them correctly
        java.util.List<OutputInfo> outputInfos = new java.util.ArrayList<>();
        for (int i = 0; i < outputNameList.size(); i++) {
            String name = outputNameList.get(i);
            OnnxValue val = result.get(name).orElse(null);
            if (val != null) {
                Object obj = val.getValue();
                int size = 0;
                if (obj instanceof float[][]) {
                    size = ((float[][])obj).length;
                } else if (obj instanceof float[]) {
                    size = ((float[])obj).length;
                }
                outputInfos.add(new OutputInfo(name, i, size));
            }
        }
        
        // Based on the debug output, the pattern appears to be:
        // Output 0 (448): 12800 - scale 0 locations
        // Output 1 (471): 3200 - scale 1 locations OR scale 0 confidences
        // Output 2 (494): 800 - scale 2 locations
        // Output 3 (451): 12800 - scale 0 confidences OR scale 1 locations
        // Output 4 (474): 3200 - scale 1 confidences
        // Output 5 (497): 800 - scale 2 confidences
        
        // Try to identify the pattern: locations should be 4x anchors, confidences should be 2x anchors
        // For 3 scales: anchors are [3200, 800, 200] (80x80x2, 40x40x2, 20x20x2)
        // So locations should be: [12800, 3200, 800] (3200*4, 800*4, 200*4)
        // And confidences should be: [6400, 1600, 400] (3200*2, 800*2, 200*2)
        // But we're seeing [12800, 3200, 800] for both, which suggests the data format is different
        
        List<DecodedDetection> allDetections = new ArrayList<>();
        
        // Map outputs correctly based on sizes
        // Locations: [12800, 3200, 800] = [3200*4, 800*4, 200*4] for scales 0,1,2
        // Confidences: Need to find outputs with matching anchor counts
        // Pattern from debug: outputs 0,1,2 have sizes [12800, 3200, 800]
        // These are likely locations. Confidences might be in outputs 3,4,5 or a different format.
        
        if (outputNameList.size() >= 6) {
            // InsightFace det_10g.onnx has 9 outputs
            // Pattern: [loc0, conf0, landm0, loc1, conf1, landm1, loc2, conf2, landm2]
            // Sizes observed: [12800, 3200, 800] for locations (3200*4, 800*4, 200*4)
            // Confidences might be: [3200, 800, 200] (1 per anchor) or [6400, 1600, 400] (2 per anchor)
            
            // Map outputs by size to scales
            // Scale 0: 3200 anchors -> 12800 location values, 3200 or 6400 confidence values
            // Scale 1: 800 anchors -> 3200 location values, 800 or 1600 confidence values  
            // Scale 2: 200 anchors -> 800 location values, 200 or 400 confidence values
            
            java.util.Map<Integer, String[]> scaleOutputs = new java.util.HashMap<>();
            
            // First pass: identify outputs by size
            for (int i = 0; i < outputNameList.size(); i++) {
                String name = outputNameList.get(i);
                OnnxValue val = result.get(name).orElse(null);
                if (val != null) {
                    Object obj = val.getValue();
                    int size = 0;
                    if (obj instanceof float[][]) {
                        size = ((float[][])obj).length;
                    } else if (obj instanceof float[]) {
                        size = ((float[])obj).length;
                    }
                    
                    // Identify scale by location size (locations are 4x anchors)
                    if (size == 12800) {
                        // Scale 0 locations (3200 anchors * 4)
                        scaleOutputs.putIfAbsent(0, new String[3]);
                        if (scaleOutputs.get(0)[0] == null) scaleOutputs.get(0)[0] = name;
                    } else if (size == 3200) {
                        // Could be scale 1 locations (800*4) OR scale 0 confidences (3200*1)
                        // Check if we already have scale 0 location
                        if (scaleOutputs.containsKey(0) && scaleOutputs.get(0)[1] == null) {
                            // Likely scale 0 confidence
                            scaleOutputs.get(0)[1] = name;
                        } else {
                            // Likely scale 1 location
                            scaleOutputs.putIfAbsent(1, new String[3]);
                            if (scaleOutputs.get(1)[0] == null) scaleOutputs.get(1)[0] = name;
                        }
                    } else if (size == 800) {
                        // Could be scale 2 locations (200*4) OR scale 1 confidences (800*1)
                        if (scaleOutputs.containsKey(1) && scaleOutputs.get(1)[1] == null) {
                            scaleOutputs.get(1)[1] = name;
                        } else {
                            scaleOutputs.putIfAbsent(2, new String[3]);
                            if (scaleOutputs.get(2)[0] == null) scaleOutputs.get(2)[0] = name;
                        }
                    } else if (size == 200) {
                        // Likely scale 2 confidence
                        if (scaleOutputs.containsKey(2)) {
                            scaleOutputs.get(2)[1] = name;
                        }
                    }
                }
            }
            
            // Decode each scale
            for (int scale = 0; scale < 3; scale++) {
                String[] outputs = scaleOutputs.get(scale);
                if (outputs != null && outputs[0] != null) {
                    String locName = outputs[0];
                    String confName = outputs[1];
                    
                    OnnxValue locValue = result.get(locName).orElse(null);
                    OnnxValue confValue = confName != null ? result.get(confName).orElse(null) : null;
                    
                    if (locValue != null) {
                        float[][] locations = parseFloat2D(locValue.getValue());
                        float[][] confidences = null;
                        
                        if (confValue != null) {
                            confidences = parseFloat2D(confValue.getValue());
                        }
                        
                        if (locations != null) {
                            // If no confidences found, create placeholder (will filter by score threshold)
                            if (confidences == null) {
                                confidences = new float[locations.length][1];
                                for (int i = 0; i < confidences.length; i++) {
                                    confidences[i] = new float[]{0.5f}; // Default score
                                }
                            }
                            
                            List<DecodedDetection> scaleDetections = decodeRetinaFaceScale(
                                locations, confidences, scale, 
                                originalWidth, originalHeight, scaleX, scaleY);
                            allDetections.addAll(scaleDetections);
                        }
                    }
                }
            }
        }
        
        // Apply Non-Maximum Suppression (NMS) to filter overlapping detections
        List<DecodedDetection> nmsDetections = applyNMS(allDetections, 0.4f); // IoU threshold 0.4
        
        // Convert to rectangles
        for (DecodedDetection det : nmsDetections) {
            if (det.score >= scoreThreshold) {
                int x = Math.max(0, (int)det.x1);
                int y = Math.max(0, (int)det.y1);
                int width = Math.min(originalWidth - x, (int)(det.x2 - det.x1));
                int height = Math.min(originalHeight - y, (int)(det.y2 - det.y1));
                
                if (width > 0 && height > 0 && width < originalWidth && height < originalHeight) {
                    faces.add(new Rect(x, y, width, height));
                }
            }
        }
        
        return faces;
    }
    
    /**
     * Helper class to store output information.
     */
    private static class OutputInfo {
        String name;
        int index;
        int size;
        
        OutputInfo(String name, int index, int size) {
            this.name = name;
            this.index = index;
            this.size = size;
        }
    }
    
    /**
     * Helper class to store decoded detections with scores.
     */
    private static class DecodedDetection {
        float x1, y1, x2, y2;
        float score;
        
        DecodedDetection(float x1, float y1, float x2, float y2, float score) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
        }
        
        float area() {
            return (x2 - x1) * (y2 - y1);
        }
        
        float iou(DecodedDetection other) {
            float interX1 = Math.max(x1, other.x1);
            float interY1 = Math.max(y1, other.y1);
            float interX2 = Math.min(x2, other.x2);
            float interY2 = Math.min(y2, other.y2);
            
            if (interX2 <= interX1 || interY2 <= interY1) {
                return 0.0f;
            }
            
            float interArea = (interX2 - interX1) * (interY2 - interY1);
            float unionArea = area() + other.area() - interArea;
            
            return unionArea > 0 ? interArea / unionArea : 0.0f;
        }
    }
    
    /**
     * Parses a value into float[][] format.
     * Handles [N, 1] format where data might be flattened.
     */
    private float[][] parseFloat2D(Object value) {
        if (value instanceof float[][]) {
            float[][] arr = (float[][]) value;
            // Check if it's [N, 1] format that needs reshaping
            if (arr.length > 0 && arr[0].length == 1) {
                // Data is flattened in [N, 1] format
                // For locations: reshape to [N/4, 4]
                // For confidences: reshape to [N/2, 2]
                int total = arr.length;
                float[] flat = new float[total];
                for (int i = 0; i < total; i++) {
                    flat[i] = arr[i][0];
                }
                
                // Try to determine if it's locations (divisible by 4) or confidences (divisible by 2)
                if (total % 4 == 0) {
                    // Likely locations: [dx, dy, dw, dh] per anchor
                    int rows = total / 4;
                    float[][] result = new float[rows][4];
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < 4; j++) {
                            result[i][j] = flat[i * 4 + j];
                        }
                    }
                    return result;
                } else if (total % 2 == 0) {
                    // Likely confidences: [bg_score, face_score] per anchor
                    int rows = total / 2;
                    float[][] result = new float[rows][2];
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < 2; j++) {
                            result[i][j] = flat[i * 2 + j];
                        }
                    }
                    return result;
                } else {
                    // Keep as [N, 1]
                    return arr;
                }
            }
            return arr;
        } else if (value instanceof float[]) {
            // Reshape: assume it's [N*4] for locations or [N*2] for confidences
            float[] flat = (float[]) value;
            int cols = flat.length > 0 ? (flat.length % 4 == 0 ? 4 : 2) : 1;
            int rows = flat.length / cols;
            float[][] result = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] = flat[i * cols + j];
                }
            }
            return result;
        }
        return null;
    }
    
    /**
     * Decodes RetinaFace outputs for a single scale using anchor boxes.
     * Based on InsightFace RetinaFace implementation.
     */
    private List<DecodedDetection> decodeRetinaFaceScale(float[][] locations, float[][] confidences,
                                                         int scaleIdx, int originalWidth, int originalHeight,
                                                         double scaleX, double scaleY) {
        List<DecodedDetection> detections = new ArrayList<>();
        
        // RetinaFace anchor parameters for InsightFace det_10g
        // Strides: [8, 16, 32] pixels for 3 scales
        // Anchor scales: [16, 32, 64] pixels
        // Typically 1 anchor per location (square)
        
        int[] strides = {8, 16, 32};
        float[] anchorScales = {16.0f, 32.0f, 64.0f};
        
        if (scaleIdx >= strides.length || scaleIdx < 0) {
            return detections;
        }
        
        int stride = strides[scaleIdx];
        float anchorScale = anchorScales[scaleIdx];
        
        // Calculate feature map size (input is 640x640)
        int featH = inputHeight / stride;
        int featW = inputWidth / stride;
        int numAnchors = featH * featW;
        
        // Verify location count matches (should be numAnchors * 4)
        if (locations.length != numAnchors) {
            // Adjust if mismatch (might be due to rounding)
            numAnchors = locations.length;
            // Recalculate feature map size
            featW = (int)Math.sqrt(numAnchors);
            featH = numAnchors / featW;
        }
        
        // Generate anchors and decode
        // InsightFace RetinaFace uses 1 anchor per location (square anchors)
        int anchorIdx = 0;
        for (int y = 0; y < featH; y++) {
            for (int x = 0; x < featW; x++) {
                if (anchorIdx >= locations.length) {
                    break;
                }
                
                // Anchor center in input image coordinates (640x640)
                // Anchors are centered at feature map grid points
                float anchorX = (x + 0.5f) * stride;
                float anchorY = (y + 0.5f) * stride;
                float anchorW = anchorScale;
                float anchorH = anchorScale;
                
                // Decode location offsets (typically [dx, dy, dw, dh])
                float[] loc = locations[anchorIdx];
                if (loc != null && loc.length >= 4) {
                        float dx = loc[0];
                        float dy = loc[1];
                        float dw = loc[2];
                        float dh = loc[3];
                        
                        // RetinaFace decoding formula (standard):
                        // center_x = anchor_x + dx * anchor_w
                        // center_y = anchor_y + dy * anchor_h  
                        // width = anchor_w * exp(dw)
                        // height = anchor_h * exp(dh)
                        float centerX = anchorX + dx * anchorW;
                        float centerY = anchorY + dy * anchorH;
                        float width = anchorW * (float)Math.exp((double)dw);
                        float height = anchorH * (float)Math.exp((double)dh);
                        
                        // Convert to corner format (x1, y1, x2, y2) in input image coordinates (640x640)
                        float x1_input = centerX - width / 2.0f;
                        float y1_input = centerY - height / 2.0f;
                        float x2_input = centerX + width / 2.0f;
                        float y2_input = centerY + height / 2.0f;
                        
                        // Clamp to input image bounds first
                        x1_input = Math.max(0, Math.min(inputWidth, x1_input));
                        y1_input = Math.max(0, Math.min(inputHeight, y1_input));
                        x2_input = Math.max(0, Math.min(inputWidth, x2_input));
                        y2_input = Math.max(0, Math.min(inputHeight, y2_input));
                        
                        // Scale back to original image coordinates
                        // scaleX = inputWidth / originalWidth, so to convert: coord_original = coord_input / scaleX
                        float x1 = x1_input / (float)scaleX;
                        float y1 = y1_input / (float)scaleY;
                        float x2 = x2_input / (float)scaleX;
                        float y2 = y2_input / (float)scaleY;
                        
                        // Final clamp to original image bounds
                        x1 = Math.max(0, Math.min(originalWidth, x1));
                        y1 = Math.max(0, Math.min(originalHeight, y1));
                        x2 = Math.max(0, Math.min(originalWidth, x2));
                        y2 = Math.max(0, Math.min(originalHeight, y2));
                        
                        // Get confidence score (face probability)
                        float faceScore = 0.0f;
                        if (anchorIdx < confidences.length) {
                            float[] conf = confidences[anchorIdx];
                            if (conf != null) {
                                if (conf.length >= 2) {
                                    // Binary classification: [background_score, face_score]
                                    faceScore = conf[1]; // Face score
                                } else if (conf.length >= 1) {
                                    // Single value: face score directly
                                    faceScore = conf[0];
                                }
                            }
                        }
                        
                    // Only add detections with reasonable confidence
                    // Lower threshold before NMS to catch more candidates
                    if (faceScore > 0.1f && x2 > x1 && y2 > y1) {
                        detections.add(new DecodedDetection(x1, y1, x2, y2, faceScore));
                    }
                }
                
                anchorIdx++;
            }
        }
        
        return detections;
    }
    
    /**
     * Applies Non-Maximum Suppression to filter overlapping detections.
     */
    private List<DecodedDetection> applyNMS(List<DecodedDetection> detections, float iouThreshold) {
        if (detections.isEmpty()) {
            return detections;
        }
        
        // Sort by score (descending)
        detections.sort((a, b) -> Float.compare(b.score, a.score));
        
        List<DecodedDetection> result = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];
        
        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) {
                continue;
            }
            
            DecodedDetection det = detections.get(i);
            result.add(det);
            
            // Suppress overlapping detections
            for (int j = i + 1; j < detections.size(); j++) {
                if (!suppressed[j]) {
                    float iou = det.iou(detections.get(j));
                    if (iou > iouThreshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
        
        return result;
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

