// ===== FaceLandmarkDetector.java =====
package com.example;

import ai.onnxruntime.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import java.nio.FloatBuffer;

/**
 * Detects facial landmarks using an ONNX model.
 * Supports models that output 5 or 68 landmarks.
 */
public class FaceLandmarkDetector implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Object lock = new Object();
    private final int landmarkCount;

    public FaceLandmarkDetector(String onnxModelPath) throws OrtException {
        if (onnxModelPath == null || onnxModelPath.isEmpty()) {
            throw new IllegalArgumentException("ONNX model path cannot be null or empty");
        }

        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(onnxModelPath, new OrtSession.SessionOptions());
        
        // Default to 5-point model (most common for face alignment)
        // The actual count will be determined at runtime from the output
        this.landmarkCount = 5;
    }

    /**
     * Detects landmarks in a face region of the full image.
     * @param fullImage The full image containing the face
     * @param faceRect The bounding rectangle of the detected face
     * @return Array of Point2f landmarks in original image coordinates, or null if detection fails
     */
    public Point2f[] detectLandmarks(Mat fullImage, Rect faceRect) throws OrtException {
        if (fullImage == null || fullImage.empty()) {
            throw new IllegalArgumentException("Image cannot be null or empty");
        }
        if (faceRect == null) {
            throw new IllegalArgumentException("Face rectangle cannot be null");
        }

        synchronized (lock) {
            // Crop face region with some padding
            int padding = (int)(faceRect.width() * 0.2);
            int x = Math.max(0, faceRect.x() - padding);
            int y = Math.max(0, faceRect.y() - padding);
            int width = Math.min(fullImage.cols() - x, faceRect.width() + 2 * padding);
            int height = Math.min(fullImage.rows() - y, faceRect.height() + 2 * padding);
            
            Rect paddedRect = new Rect(x, y, width, height);
            Mat faceCrop = new Mat(fullImage, paddedRect);
            
            try {
                // Preprocess face image for landmark detector
                // This model expects 128x128 input
                Mat resized = new Mat();
                try {
                    opencv_imgproc.resize(faceCrop, resized, new org.bytedeco.opencv.opencv_core.Size(128, 128));
                    
                    Mat normalized = new Mat();
                    try {
                        // Convert BGR to RGB
                        Mat rgb = new Mat();
                        try {
                            opencv_imgproc.cvtColor(resized, rgb, opencv_imgproc.COLOR_BGR2RGB);
                            
                            // Normalize to [0, 1] range
                            rgb.convertTo(normalized, opencv_core.CV_32FC3, 1.0 / 255.0, 0);
                            
                            // Convert to CHW format
                            float[] imageData = convertHWCtoCHW(normalized);
                            
                            // Run inference - model expects input name "image" and 128x128 size
                            long[] shape = new long[]{1, 3, 128, 128};
                            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), shape)) {
                                OrtSession.Result result = session.run(java.util.Collections.singletonMap("image", tensor));
                                try {
                                    
                            // Get landmarks output
                            OnnxValue output = result.get(0);
                            Object value = output.getValue();
                            
                            float[] flatLandmarks;
                            if (value instanceof float[][]) {
                                // Shape [1, num_landmarks*2] or [num_landmarks, 2]
                                float[][] landmarks2d = (float[][]) value;
                                if (landmarks2d.length == 1) {
                                    // Single row: [1, num_landmarks*2]
                                    flatLandmarks = landmarks2d[0];
                                } else {
                                    // Multiple rows: [num_landmarks, 2] - flatten it
                                    int numPoints = landmarks2d.length;
                                    flatLandmarks = new float[numPoints * 2];
                                    for (int i = 0; i < numPoints; i++) {
                                        flatLandmarks[i * 2] = landmarks2d[i][0];
                                        flatLandmarks[i * 2 + 1] = landmarks2d[i][1];
                                    }
                                }
                            } else if (value instanceof float[]) {
                                // Shape [num_landmarks*2]
                                flatLandmarks = (float[]) value;
                            } else {
                                throw new OrtException("Unexpected landmark output format: " + value.getClass().getName());
                            }
                                    
                                    // Parse landmarks in 128x128 coordinate space (the resized image)
                                    Point2f[] resizedPoints = parseLandmarks(flatLandmarks, 128, 128);
                                    
                                    // Convert from 128x128 space to padded crop space, then to original image coordinates
                                    Point2f[] imagePoints = new Point2f[resizedPoints.length];
                                    float scaleX = (float)width / 128.0f;
                                    float scaleY = (float)height / 128.0f;
                                    for (int i = 0; i < resizedPoints.length; i++) {
                                        imagePoints[i] = new Point2f(
                                            resizedPoints[i].x() * scaleX + x,
                                            resizedPoints[i].y() * scaleY + y
                                        );
                                    }
                                    
                                    return imagePoints;
                                } finally {
                                    result.close();
                                }
                            }
                        } finally {
                            rgb.release();
                        }
                    } finally {
                        normalized.release();
                    }
                } finally {
                    resized.release();
                }
            } finally {
                faceCrop.release();
            }
        }
    }

    /**
     * Parses flat landmark array into Point2f array.
     * Handles both absolute coordinates and normalized coordinates.
     */
    private Point2f[] parseLandmarks(float[] flatLandmarks, int imgWidth, int imgHeight) {
        int count = flatLandmarks.length / 2;
        Point2f[] points = new Point2f[count];
        
        // Check coordinate system: normalized [0,1], absolute, or centered [-1,1]
        boolean hasNegative = false;
        boolean hasGreaterThanOne = false;
        float minVal = Float.MAX_VALUE;
        float maxVal = Float.MIN_VALUE;
        
        for (float val : flatLandmarks) {
            if (val < 0.0f) hasNegative = true;
            if (val > 1.0f) hasGreaterThanOne = true;
            minVal = Math.min(minVal, val);
            maxVal = Math.max(maxVal, val);
        }
        
        // System.out.println("    Coordinate range: [" + minVal + ", " + maxVal + "], hasNegative=" + hasNegative + ", hasGreaterThanOne=" + hasGreaterThanOne);
        
        for (int i = 0; i < count; i++) {
            float x = flatLandmarks[i * 2];
            float y = flatLandmarks[i * 2 + 1];
            
            if (hasNegative && !hasGreaterThanOne) {
                // Centered coordinates [-1,1] - convert to [0,1] then to absolute
                x = (x + 1.0f) / 2.0f * imgWidth;
                y = (y + 1.0f) / 2.0f * imgHeight;
            } else if (!hasNegative && !hasGreaterThanOne) {
                // Normalized [0,1]
                x = x * imgWidth;
                y = y * imgHeight;
            } else if (hasNegative && hasGreaterThanOne) {
                // Has both negative and >1 values - likely mostly [-1,1] with outliers
                // Clamp outliers and convert from [-1,1] to absolute
                x = Math.max(-1.0f, Math.min(1.0f, x));
                y = Math.max(-1.0f, Math.min(1.0f, y));
                x = (x + 1.0f) / 2.0f * imgWidth;
                y = (y + 1.0f) / 2.0f * imgHeight;
            } else {
                // hasGreaterThanOne but no negatives - could be absolute, but check if reasonable
                if (maxVal < imgWidth && maxVal < imgHeight) {
                    // Values are within image bounds, treat as absolute
                    // x = x; y = y; (use as-is)
                } else {
                    // Values are out of bounds, likely normalized [0,1]
                    x = x * imgWidth;
                    y = y * imgHeight;
                }
            }
            
            points[i] = new Point2f(x, y);
        }
        
        return points;
    }

    /**
     * Converts image from HWC (Height-Width-Channel) to CHW (Channel-Height-Width) format.
     */
    private float[] convertHWCtoCHW(Mat image) {
        int h = image.rows();
        int w = image.cols();
        int c = image.channels();

        float[] chwData = new float[h * w * c];
        FloatIndexer indexer = image.createIndexer();

        int idx = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    chwData[idx++] = indexer.get(row, col, ch);
                }
            }
        }

        indexer.close();
        return chwData;
    }

    public int getLandmarkCount() {
        return landmarkCount;
    }

    @Override
    public void close() throws OrtException {
        synchronized (lock) {
            if (session != null) {
                session.close();
            }
        }
    }
}

