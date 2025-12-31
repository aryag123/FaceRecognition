// ===== ImagePreprocessor.java =====
package com.example;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import java.io.File;

public class ImagePreprocessor {

    public static final int TARGET_WIDTH = 112;
    public static final int TARGET_HEIGHT = 112;
    public static final int CHANNELS = 3;

    private static CascadeClassifier faceDetector;
    private static FaceLandmarkDetector landmarkDetector;

    // Initialize face detector (call this once at startup)
    public static void initFaceDetector(String cascadePath) {
        faceDetector = new CascadeClassifier(cascadePath);
        if (faceDetector.empty()) {
            throw new RuntimeException("Failed to load face detector from: " + cascadePath);
        }
    }

    // Initialize landmark detector (optional - if null, uses estimated landmarks)
    public static void initLandmarkDetector(String landmarkModelPath) throws Exception {
        if (landmarkModelPath != null && !landmarkModelPath.isEmpty()) {
            landmarkDetector = new FaceLandmarkDetector(landmarkModelPath);
        }
    }

    // Cleanup landmark detector
    public static void closeLandmarkDetector() throws Exception {
        if (landmarkDetector != null) {
            landmarkDetector.close();
            landmarkDetector = null;
        }
    }

    public static float[] preprocessImage(String imagePath) {
        if (imagePath == null || imagePath.isEmpty()) {
            throw new IllegalArgumentException("Image path cannot be null or empty");
        }

        File file = new File(imagePath);
        if (!file.exists() || !file.isFile()) {
            throw new IllegalArgumentException("Image file does not exist: " + imagePath);
        }

        Mat image = opencv_imgcodecs.imread(imagePath);
        if (image.empty()) {
            throw new IllegalArgumentException("Failed to load image: " + imagePath);
        }

        try {
            // Detect face and get bounding box
            Rect faceRect = detectFaceRect(image);
            if (faceRect == null) {
                throw new IllegalArgumentException("No face detected in image: " + imagePath);
            }

            // Align face using estimated landmarks (landmark detector is disabled due to issues)
            Mat alignedFace = alignFace(image, faceRect);
            if (alignedFace == null) {
                throw new IllegalArgumentException("Face alignment failed for image: " + imagePath);
            }

            try {
                // Use BGR directly (InsightFace and many ArcFace models expect BGR)
                // No color conversion needed - OpenCV loads images as BGR
                Mat normalized = new Mat();
                try {
                    // Standard preprocessing: (pixel - 127.5) / 128.0
                    // This converts [0, 255] to approximately [-1, 1] range
                    alignedFace.convertTo(normalized, opencv_core.CV_32FC3, 1.0 / 128.0, -127.5 / 128.0);
                    return convertHWCtoCHW(normalized);
                } finally {
                    normalized.release();
                }
            } finally {
                alignedFace.release();
            }
        } finally {
            image.release();
        }
    }

    /**
     * Detects face in image and returns face bounding rectangle.
     * Returns null if no face detected.
     */
    private static Rect detectFaceRect(Mat image) {
        if (faceDetector == null) {
            throw new IllegalStateException("Face detector not initialized. Call initFaceDetector() first.");
        }

        // Convert to grayscale for detection
        Mat gray = new Mat();
        try {
            opencv_imgproc.cvtColor(image, gray, opencv_imgproc.COLOR_BGR2GRAY);
            opencv_imgproc.equalizeHist(gray, gray);

            // Detect faces
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(gray, faces);

            if (faces.size() == 0) {
                return null; // No face found
            }

            // Get the largest face (in case multiple detected)
            Rect largestFace = faces.get(0);
            for (int i = 1; i < faces.size(); i++) {
                Rect face = faces.get(i);
                if (face.width() * face.height() > largestFace.width() * largestFace.height()) {
                    largestFace = face;
                }
            }

            return largestFace;
        } finally {
            gray.release();
        }
    }

    /**
     * Aligns face to standard ArcFace template using similarity transform.
     * Standard template landmarks (for 112x112 image):
     * - Left eye: (30.2946, 51.6963)
     * - Right eye: (65.5318, 51.5014)
     * - Nose: (48.0252, 71.7366)
     * - Left mouth: (33.5493, 92.3655)
     * - Right mouth: (62.7299, 92.2041)
     */
    private static Mat alignFace(Mat image, Rect faceRect) {
        Point2f leftEye, rightEye;
        
        // Try to detect real landmarks if landmark detector is available
        if (landmarkDetector != null) {
            try {
                Point2f[] landmarks = landmarkDetector.detectLandmarks(image, faceRect);
                if (landmarks != null && landmarks.length >= 5) {
                    System.out.println("    Detected " + landmarks.length + " landmarks");
                    // For 5-point models: [left_eye, right_eye, nose, left_mouth, right_mouth]
                    // For 68-point models: indices 36-47 are eyes, 30 is nose, 48-67 is mouth
                    if (landmarks.length == 5) {
                        // 5-point model
                        leftEye = landmarks[0];
                        rightEye = landmarks[1];
                        System.out.println("    Using 5-point landmarks: leftEye=(" + String.format("%.1f", leftEye.x()) + "," + String.format("%.1f", leftEye.y()) + 
                                         ") rightEye=(" + String.format("%.1f", rightEye.x()) + "," + String.format("%.1f", rightEye.y()) + ")");
                        // System.out.println("    Using 5-point landmarks: leftEye=(" + leftEye.x() + "," + leftEye.y() + 
                        //                  ") rightEye=(" + rightEye.x() + "," + rightEye.y() + ")");
                    } else if (landmarks.length == 66) {
                        // 66-point model (33 landmarks) - typical format
                        // For 66-point models, eyes are typically around indices 16-19 (left) and 20-23 (right)
                        // Use approximate centers
                        leftEye = landmarks[17];   // Approximate left eye center
                        rightEye = landmarks[21];  // Approximate right eye center
                    } else if (landmarks.length == 132) {
                        // 132-point model - try different approaches
                        // First, try to find eyes by looking for horizontally separated points in upper face region
                        // Eyes should be in top 1/3 of face and horizontally separated
                        float maxY = 0;
                        for (Point2f p : landmarks) {
                            maxY = Math.max(maxY, p.y());
                        }
                        float eyeRegionTop = maxY * 0.2f;  // Top 20% of face
                        float eyeRegionBottom = maxY * 0.5f; // Top 50% of face
                        
                        // Find leftmost and rightmost points in eye region
                        float leftMostX = Float.MAX_VALUE, rightMostX = Float.MIN_VALUE;
                        int leftIdx = -1, rightIdx = -1;
                        
                        for (int i = 0; i < landmarks.length; i++) {
                            Point2f p = landmarks[i];
                            if (p.y() >= eyeRegionTop && p.y() <= eyeRegionBottom) {
                                if (p.x() < leftMostX) {
                                    leftMostX = p.x();
                                    leftIdx = i;
                                }
                                if (p.x() > rightMostX) {
                                    rightMostX = p.x();
                                    rightIdx = i;
                                }
                            }
                        }
                        
                        if (leftIdx >= 0 && rightIdx >= 0 && leftIdx != rightIdx) {
                            leftEye = landmarks[leftIdx];
                            rightEye = landmarks[rightIdx];
                            System.out.println("    Using " + landmarks.length + "-point landmarks (found eyes at indices " + leftIdx + "," + rightIdx + "): leftEye=(" + 
                                             String.format("%.1f", leftEye.x()) + "," + String.format("%.1f", leftEye.y()) + 
                                             ") rightEye=(" + String.format("%.1f", rightEye.x()) + "," + String.format("%.1f", rightEye.y()) + ")");
                        } else {
                            // Fallback to fixed indices
                            leftEye = landmarks[Math.min(32, landmarks.length - 1)];
                            rightEye = landmarks[Math.min(36, landmarks.length - 1)];
                            // System.out.println("    Using 132-point landmarks (fallback indices): leftEye=(" + 
                            //                  String.format("%.1f", leftEye.x()) + "," + String.format("%.1f", leftEye.y()) + 
                            //                  ") rightEye=(" + String.format("%.1f", rightEye.x()) + "," + String.format("%.1f", rightEye.y()) + ")");
                        }
                    } else {
                        // 68-point or other - use standard indices
                        int leftIdx = Math.min(36, landmarks.length - 1);
                        int rightIdx = Math.min(45, landmarks.length - 1);
                        leftEye = landmarks[leftIdx];
                        rightEye = landmarks[rightIdx];
                        System.out.println("    Using " + landmarks.length + "-point landmarks (standard indices " + leftIdx + "," + rightIdx + "): leftEye=(" + 
                                         String.format("%.1f", leftEye.x()) + "," + String.format("%.1f", leftEye.y()) + 
                                         ") rightEye=(" + String.format("%.1f", rightEye.x()) + "," + String.format("%.1f", rightEye.y()) + ")");
                    }
                    } else {
                        // Fall back to estimated landmarks
                        leftEye = null;
                        rightEye = null;
                    }
                } catch (Exception e) {
                    // Fall back to estimated landmarks if detection fails
                    leftEye = null;
                    rightEye = null;
                }
        } else {
            leftEye = null;
            rightEye = null;
        }
        
        // Use estimated landmarks if real ones are not available
        if (leftEye == null || rightEye == null) {
            // Better estimation: use face geometry
            // Eyes are typically at about 30-40% from top of face
            // Eye separation is typically 30-40% of face width
            float eyeY = faceRect.y() + faceRect.height() * 0.35f;  // 35% from top
            float eyeDistance = faceRect.width() * 0.38f;  // 38% of face width
            float centerX = faceRect.x() + faceRect.width() / 2.0f;
            
            leftEye = new Point2f(centerX - eyeDistance / 2.0f, eyeY);
            rightEye = new Point2f(centerX + eyeDistance / 2.0f, eyeY);
        }
        
        // Standard template landmarks (normalized to 112x112)
        Point2f templateLeftEye = new Point2f(30.2946f, 51.6963f);
        Point2f templateRightEye = new Point2f(65.5318f, 51.5014f);
        
        // Calculate similarity transform (rotation, scale, translation)
        Mat transform = getSimilarityTransform(leftEye, rightEye, templateLeftEye, templateRightEye);
        
        // Apply transformation
        Mat aligned = new Mat();
        opencv_imgproc.warpAffine(image, aligned, transform, new Size(TARGET_WIDTH, TARGET_HEIGHT));
        transform.close();
        
        return aligned;
    }

    /**
     * Computes similarity transform matrix to align source landmarks to template landmarks.
     * Similarity transform: rotation + scale + translation
     */
    private static Mat getSimilarityTransform(Point2f srcLeftEye, Point2f srcRightEye,
                                              Point2f dstLeftEye, Point2f dstRightEye) {
        // Calculate scale
        double srcEyeDist = Math.sqrt(Math.pow(srcRightEye.x() - srcLeftEye.x(), 2) + 
                                      Math.pow(srcRightEye.y() - srcLeftEye.y(), 2));
        double dstEyeDist = Math.sqrt(Math.pow(dstRightEye.x() - dstLeftEye.x(), 2) + 
                                      Math.pow(dstRightEye.y() - dstLeftEye.y(), 2));
        double scale = dstEyeDist / srcEyeDist;
        
        // Calculate rotation angle
        double srcAngle = Math.atan2(srcRightEye.y() - srcLeftEye.y(), 
                                     srcRightEye.x() - srcLeftEye.x());
        double dstAngle = Math.atan2(dstRightEye.y() - dstLeftEye.y(), 
                                     dstRightEye.x() - dstLeftEye.x());
        double angle = dstAngle - srcAngle;
        
        // Calculate translation
        double srcCenterX = (srcLeftEye.x() + srcRightEye.x()) / 2.0;
        double srcCenterY = (srcLeftEye.y() + srcRightEye.y()) / 2.0;
        double dstCenterX = (dstLeftEye.x() + dstRightEye.x()) / 2.0;
        double dstCenterY = (dstLeftEye.y() + dstRightEye.y()) / 2.0;
        
        // Build similarity transform matrix [s*cos(θ) -s*sin(θ) tx]
        //                                   [s*sin(θ)  s*cos(θ) ty]
        double cosA = Math.cos(angle);
        double sinA = Math.sin(angle);
        
        double tx = dstCenterX - scale * (cosA * srcCenterX - sinA * srcCenterY);
        double ty = dstCenterY - scale * (sinA * srcCenterX + cosA * srcCenterY);
        
        // Create 2x3 affine transformation matrix
        Mat transform = new Mat(2, 3, opencv_core.CV_64FC1);
        DoubleIndexer idx = transform.createIndexer();
        idx.put(0, 0, scale * cosA);
        idx.put(0, 1, -scale * sinA);
        idx.put(0, 2, tx);
        idx.put(1, 0, scale * sinA);
        idx.put(1, 1, scale * cosA);
        idx.put(1, 2, ty);
        idx.close();
        
        return transform;
    }

    private static float[] convertHWCtoCHW(Mat image) {
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
}