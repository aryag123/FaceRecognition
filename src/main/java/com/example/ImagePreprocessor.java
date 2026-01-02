// ===== ImagePreprocessor.java =====
package com.example;

import java.io.File;

import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class ImagePreprocessor {

    public static final int TARGET_WIDTH = 112;
    public static final int TARGET_HEIGHT = 112;
    public static final int CHANNELS = 3;

    private static CascadeClassifier faceDetector;
    private static ONNXFaceDetector onnxFaceDetector;
    private static FaceLandmarkDetector landmarkDetector;
    private static boolean useOnnxDetector = false;

    // Initialize face detector using XML cascade (call this once at startup)
    public static void initFaceDetector(String cascadePath) {
        faceDetector = new CascadeClassifier(cascadePath);
        if (faceDetector.empty()) {
            throw new RuntimeException("Failed to load face detector from: " + cascadePath);
        }
        useOnnxDetector = false;
    }

    // Initialize face detector using ONNX model (e.g., RetinaFace)
    public static void initOnnxFaceDetector(String onnxModelPath) throws Exception {
        onnxFaceDetector = new ONNXFaceDetector(onnxModelPath);
        useOnnxDetector = true;
    }

    // Cleanup ONNX face detector
    public static void closeOnnxFaceDetector() throws Exception {
        if (onnxFaceDetector != null) {
            onnxFaceDetector.close();
            onnxFaceDetector = null;
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

    /**
     * Detects all faces in an image and returns their bounding rectangles.
     * @param imagePath Path to the image file
     * @return List of detected face rectangles, empty list if no faces found
     */
    public static java.util.List<Rect> detectAllFaces(String imagePath) {
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
            // Get all faces BEFORE filtering (raw detection results)
            java.util.List<Rect> allDetectedFaces = detectFaceRectsRaw(image);
            
            // Export ALL detected faces (before filtering)
            if (allDetectedFaces != null && !allDetectedFaces.isEmpty()) {
                System.out.println("    Exporting " + allDetectedFaces.size() + " detected face(s) to DetectedFaces folder...");
                for (int i = 0; i < allDetectedFaces.size(); i++) {
                    exportDetectedFace(imagePath, allDetectedFaces.get(i), i + 1, image);
                }
            } else {
                System.out.println("    No faces detected to export.");
            }
            
            // Now return filtered faces
            return detectFaceRects(image);
        } finally {
            image.release();
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
            // Detect face and get bounding box (uses largest face for backward compatibility)
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
                
                    float[] result = convertHWCtoCHW(normalized);
                    
                    return result;
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
     * Preprocesses a specific face rectangle from an image.
     * @param imagePath Path to the image file
     * @param faceRect The face rectangle to process
     * @return Preprocessed face embedding array
     */
    public static float[] preprocessFace(String imagePath, Rect faceRect) {
        if (imagePath == null || imagePath.isEmpty()) {
            throw new IllegalArgumentException("Image path cannot be null or empty");
        }
        if (faceRect == null) {
            throw new IllegalArgumentException("Face rectangle cannot be null");
        }

        Mat image = opencv_imgcodecs.imread(imagePath);
        if (image.empty()) {
            throw new IllegalArgumentException("Failed to load image: " + imagePath);
        }

        try {
            // Align face using estimated landmarks
            Mat alignedFace = alignFace(image, faceRect);
            if (alignedFace == null) {
                throw new IllegalArgumentException("Face alignment failed for image: " + imagePath);
            }

            try {
                // Use BGR directly (InsightFace and many ArcFace models expect BGR)
                Mat normalized = new Mat();
                try {
                    // Standard preprocessing: (pixel - 127.5) / 128.0
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
     * Detects faces in image and returns list of face bounding rectangles.
     * Returns empty list if no faces detected.
     */
    private static java.util.List<Rect> detectFaceRects(Mat image) {
        if (useOnnxDetector) {
            return detectFaceRectsOnnx(image);
        } else {
            return detectFaceRectsCascade(image);
        }
    }
    
    /**
     * Detects face in image and returns face bounding rectangle (for backward compatibility).
     * Returns null if no face detected. Returns the largest face if multiple detected.
     */
    private static Rect detectFaceRect(Mat image) {
        java.util.List<Rect> faces = detectFaceRects(image);
        if (faces == null || faces.isEmpty()) {
            return null;
        }
        // Return largest face for backward compatibility
        Rect largestFace = faces.get(0);
        for (int i = 1; i < faces.size(); i++) {
            Rect face = faces.get(i);
            if (face.width() * face.height() > largestFace.width() * largestFace.height()) {
                largestFace = face;
            }
        }
        return largestFace;
    }

    /**
     * Detects faces using ONNX model (e.g., RetinaFace) - raw detection, no filtering.
     * Returns list of all detected faces before any filtering.
     */
    private static java.util.List<Rect> detectFaceRectsOnnxRaw(Mat image) {
        if (onnxFaceDetector == null) {
            throw new IllegalStateException("ONNX face detector not initialized. Call initOnnxFaceDetector() first.");
        }

        try {
            java.util.List<Rect> faces = onnxFaceDetector.detectFaces(image);
            
            if (faces == null || faces.isEmpty()) {
                return new java.util.ArrayList<>(); // Return empty list if no faces found
            }

            return faces; // Return raw detected faces (no filtering)
        } catch (Exception e) {
            // If ONNX detection fails, fall back to cascade if available
            System.err.println("ONNX face detection failed: " + e.getMessage());
            System.err.println("Falling back to XML cascade detector...");
            if (faceDetector != null) {
                return detectFaceRectsCascadeRaw(image);
            }
            throw new RuntimeException("ONNX face detection failed and no cascade fallback available: " + e.getMessage(), e);
        }
    }
    
    /**
     * Detects faces using ONNX model (e.g., RetinaFace).
     * Returns list of all detected faces (after filtering).
     */
    private static java.util.List<Rect> detectFaceRectsOnnx(Mat image) {
        if (onnxFaceDetector == null) {
            throw new IllegalStateException("ONNX face detector not initialized. Call initOnnxFaceDetector() first.");
        }

        try {
            java.util.List<Rect> faces = onnxFaceDetector.detectFaces(image);
            
            if (faces == null || faces.isEmpty()) {
                return new java.util.ArrayList<>(); // Return empty list if no faces found
            }

            // Filter out very small faces (likely false positives)
            faces = filterSmallFaces(faces, image);
            
            // Filter false positives using landmark validation if available
            if (landmarkDetector != null) {
                faces = validateFacesWithLandmarks(image, faces);
            }

            return faces; // Return all detected faces
        } catch (Exception e) {
            // If ONNX detection fails, fall back to cascade if available
            System.err.println("ONNX face detection failed: " + e.getMessage());
            System.err.println("Falling back to XML cascade detector...");
            if (faceDetector != null) {
                return detectFaceRectsCascade(image);
            }
            throw new RuntimeException("ONNX face detection failed and no cascade fallback available: " + e.getMessage(), e);
        }
    }
    
    /**
     * Gets raw face detections (before filtering) for export purposes.
     */
    private static java.util.List<Rect> detectFaceRectsRaw(Mat image) {
        if (useOnnxDetector) {
            return detectFaceRectsOnnxRaw(image);
        } else {
            return detectFaceRectsCascadeRaw(image);
        }
    }
    
    /**
     * Exports a detected face using the Mat image directly (for use during detection).
     */
    private static void exportDetectedFace(String imagePath, Rect faceRect, int faceIndex, Mat image) {
        try {
            // Extract the face region from the image
            Mat faceRegion = new Mat(image, faceRect);
            
            // Use absolute path to ensure folder is created
            String exportFolder = "/Users/Arya/FaceRecognition/DetectedFaces";
            File exportDir = new File(exportFolder);
            if (!exportDir.exists()) {
                boolean created = exportDir.mkdirs();
                if (!created) {
                    System.err.println("Warning: Failed to create DetectedFaces folder: " + exportFolder);
                } else {
                    System.out.println("    Created DetectedFaces folder: " + exportFolder);
                }
            }
            
            // Generate filename: originalName_faceN.jpg
            File imageFile = new File(imagePath);
            String baseName = imageFile.getName();
            String nameWithoutExt = baseName.replaceFirst("\\.[^.]+$", "");
            String ext = baseName.substring(baseName.lastIndexOf('.'));
            String outputFileName = String.format("%s_face%d%s", nameWithoutExt, faceIndex, ext);
            String outputPath = exportFolder + File.separator + outputFileName;
            
            // Save the face region
            boolean success = opencv_imgcodecs.imwrite(outputPath, faceRegion);
            if (success) {
                System.out.println("    Exported detected face " + faceIndex + " to: " + outputPath);
            } else {
                System.err.println("    Failed to export detected face " + faceIndex + " to: " + outputPath);
            }
            
            faceRegion.release();
        } catch (Exception e) {
            System.err.println("Failed to export detected face " + faceIndex + ": " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Detects faces using XML cascade classifier (raw detection, no filtering).
     * Returns list of all detected faces before any filtering.
     */
    private static java.util.List<Rect> detectFaceRectsCascadeRaw(Mat image) {
        if (faceDetector == null) {
            throw new IllegalStateException("Face detector not initialized. Call initFaceDetector() first.");
        }

        // Convert to grayscale for detection
        Mat gray = new Mat();
        try {
            opencv_imgproc.cvtColor(image, gray, opencv_imgproc.COLOR_BGR2GRAY);
            opencv_imgproc.equalizeHist(gray, gray);

            // Detect faces with optimized parameters for better accuracy
            // Using stricter parameters to reduce false positives
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(
                gray,
                faces,
                1.1,         // scaleFactor: higher = fewer false positives (1.1 is more strict than 1.05)
                6,           // minNeighbors: higher = fewer false positives (6 is more strict than 5)
                0,           // flags
                new Size(40, 40), // minSize: larger minimum size reduces false positives
                new Size()        // maxSize (0,0 means no explicit max)
            );

            // Convert RectVector to List<Rect> (raw results, no filtering)
            java.util.List<Rect> faceList = new java.util.ArrayList<>();
            for (long i = 0; i < faces.size(); i++) {
                faceList.add(faces.get(i));
            }
            
            return faceList;
        } finally {
            gray.release();
        }
    }
    
    /**
     * Detects faces using XML cascade classifier.
     * Returns list of all detected faces (after filtering).
     */
    private static java.util.List<Rect> detectFaceRectsCascade(Mat image) {
        if (faceDetector == null) {
            throw new IllegalStateException("Face detector not initialized. Call initFaceDetector() first.");
        }

        // Convert to grayscale for detection
        Mat gray = new Mat();
        try {
            opencv_imgproc.cvtColor(image, gray, opencv_imgproc.COLOR_BGR2GRAY);
            opencv_imgproc.equalizeHist(gray, gray);

            // Detect faces with optimized parameters for better accuracy
            // Using stricter parameters to reduce false positives
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(
                gray,
                faces,
                1.1,         // scaleFactor: higher = fewer false positives (1.1 is more strict than 1.05)
                6,           // minNeighbors: higher = fewer false positives (6 is more strict than 5)
                0,           // flags
                new Size(40, 40), // minSize: larger minimum size reduces false positives
                new Size()        // maxSize (0,0 means no explicit max)
            );

            // Convert RectVector to List<Rect>
            java.util.List<Rect> faceList = new java.util.ArrayList<>();
            for (long i = 0; i < faces.size(); i++) {
                faceList.add(faces.get(i));
            }
            
            // Filter out very small faces (likely false positives)
            faceList = filterSmallFaces(faceList, image);
            
            // Filter out overlapping detections (likely the same face detected multiple times)
            if (faceList.size() > 1) {
                faceList = filterOverlappingFaces(faceList);
            }
            
            // Filter false positives using landmark validation if available
            if (landmarkDetector != null) {
                faceList = validateFacesWithLandmarks(image, faceList);
            }

            return faceList;
        } finally {
            gray.release();
        }
    }
    
    /**
     * Filters out extremely small face detections (only rejects clearly invalid sizes).
     * Uses very lenient thresholds - only rejects faces smaller than 20x20 pixels.
     */
    private static java.util.List<Rect> filterSmallFaces(java.util.List<Rect> faces, Mat image) {
        if (faces == null || faces.isEmpty()) {
            return faces;
        }
        
        // Very lenient: only reject faces smaller than 20x20 pixels (extremely small)
        int minSize = 20;
        
        java.util.List<Rect> filtered = new java.util.ArrayList<>();
        for (Rect face : faces) {
            // Only reject if face is extremely small (likely a detection error)
            if (face.width() >= minSize && face.height() >= minSize) {
                filtered.add(face);
            }
            // Don't print messages - just silently filter extremely small detections
        }
        
        return filtered;
    }
    
    /**
     * Filters out overlapping face detections, keeping only the largest non-overlapping faces.
     * This helps when the cascade detector finds multiple detections of the same face.
     */
    private static java.util.List<Rect> filterOverlappingFaces(java.util.List<Rect> faces) {
        if (faces.size() <= 1) {
            return faces;
        }
        
        // Sort by area (largest first)
        faces.sort((a, b) -> {
            int areaA = a.width() * a.height();
            int areaB = b.width() * b.height();
            return Integer.compare(areaB, areaA); // Descending order
        });
        
        java.util.List<Rect> filtered = new java.util.ArrayList<>();
        boolean[] used = new boolean[faces.size()];
        
        for (int i = 0; i < faces.size(); i++) {
            if (used[i]) {
                continue;
            }
            
            Rect current = faces.get(i);
            filtered.add(current);
            used[i] = true;
            
            // Mark overlapping faces as used
            for (int j = i + 1; j < faces.size(); j++) {
                if (used[j]) {
                    continue;
                }
                
                Rect other = faces.get(j);
                double overlap = calculateOverlapRatio(current, other);
                
                // If overlap is more than 50%, consider it the same face
                if (overlap > 0.5) {
                    used[j] = true;
                }
            }
        }
        
        return filtered;
    }
    
    /**
     * Calculates overlap ratio between two rectangles.
     * Returns the ratio of intersection area to the smaller rectangle's area.
     */
    private static double calculateOverlapRatio(Rect rect1, Rect rect2) {
        int x1 = Math.max(rect1.x(), rect2.x());
        int y1 = Math.max(rect1.y(), rect2.y());
        int x2 = Math.min(rect1.x() + rect1.width(), rect2.x() + rect2.width());
        int y2 = Math.min(rect1.y() + rect1.height(), rect2.y() + rect2.height());
        
        if (x2 <= x1 || y2 <= y1) {
            return 0.0; // No overlap
        }
        
        int intersectionArea = (x2 - x1) * (y2 - y1);
        int area1 = rect1.width() * rect1.height();
        int area2 = rect2.width() * rect2.height();
        int minArea = Math.min(area1, area2);
        
        return minArea > 0 ? (double) intersectionArea / minArea : 0.0;
    }
    
    /**
     * Validates detected faces using landmark detection to filter false positives.
     * Uses extremely lenient validation - keeps all faces regardless of landmark validation.
     * This is mainly for logging purposes, not for filtering.
     * 
     * @param image The full image
     * @param faces List of detected face rectangles
     * @return All faces (validation doesn't filter anything)
     */
    private static java.util.List<Rect> validateFacesWithLandmarks(Mat image, java.util.List<Rect> faces) {
        if (landmarkDetector == null || faces == null || faces.isEmpty()) {
            return faces; // No validation possible, return as-is
        }
        
        // Don't filter anything - just return all faces
        // Landmark validation is disabled to avoid rejecting valid faces
        return faces;
    }
    
    /**
     * Validates that detected landmarks are reasonable and within bounds.
     * Uses extremely lenient validation - only flags completely impossible landmarks.
     * Even if validation fails, faces are kept (this is just for logging).
     * 
     * @param landmarks Array of detected landmarks
     * @param faceRect The face bounding rectangle
     * @param image The full image
     * @return null if landmarks are valid, or a string describing the issue if invalid (for logging only)
     */
    private static String getLandmarkValidationIssue(Point2f[] landmarks, Rect faceRect, Mat image) {
        // Extremely lenient: accept any landmarks that exist
        if (landmarks == null || landmarks.length == 0) {
            return null; // Don't reject - landmark detection might have failed
        }
        
        // Accept any number of landmarks (even just 1)
        if (landmarks.length < 1) {
            return null; // Don't reject
        }
        
        int imgWidth = image.cols();
        int imgHeight = image.rows();
        
        // Only flag if ALL landmarks are way out of bounds (extremely strict)
        int outOfBoundsCount = 0;
        for (Point2f landmark : landmarks) {
            // Very generous bounds check (50% outside image)
            if (landmark.x() < -imgWidth * 0.5 || landmark.x() > imgWidth * 1.5 ||
                landmark.y() < -imgHeight * 0.5 || landmark.y() > imgHeight * 1.5) {
                outOfBoundsCount++;
            }
        }
        
        // Only flag if ALL landmarks are way out of bounds
        if (outOfBoundsCount == landmarks.length && landmarks.length > 0) {
            return "all landmarks way out of bounds (" + outOfBoundsCount + "/" + landmarks.length + ")";
        }
        
        // For 5-point models: only check for completely impossible configurations
        if (landmarks.length >= 5) {
            Point2f leftEye = landmarks[0];
            Point2f rightEye = landmarks[1];
            Point2f nose = landmarks[2];
            Point2f leftMouth = landmarks[3];
            Point2f rightMouth = landmarks[4];
            
            // Only flag if eyes are at exactly the same position (distance < 1 pixel)
            double eyeDistance = Math.sqrt(Math.pow(leftEye.x() - rightEye.x(), 2) + 
                                          Math.pow(leftEye.y() - rightEye.y(), 2));
            if (eyeDistance < 1.0) { // Eyes at same position (impossible)
                return "eyes at same position (distance: " + String.format("%.1f", eyeDistance) + ")";
            }
        }
        
        return null; // All checks passed - landmarks are valid (or acceptable)
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
        
        // DISABLED: Landmark-based alignment causes incorrect matches
        // The landmark detector produces unreliable coordinates that cause bad alignment,
        // making all faces look similar. Using estimated landmarks instead for better accuracy.
        // 
        // If you want to re-enable landmark-based alignment, uncomment the code below
        // and ensure the landmark detector produces accurate, consistent results.
        
        /*
        // Try to detect real landmarks if landmark detector is available
        if (landmarkDetector != null) {
            try {
                Point2f[] landmarks = landmarkDetector.detectLandmarks(image, faceRect);
                // ... landmark detection code ...
            } catch (Exception e) {
                // Fall back to estimated landmarks if detection fails
            }
        }
        */
        
        // Always use estimated landmarks (more reliable than detected landmarks)
        leftEye = null;
        rightEye = null;
        
        // Use estimated landmarks
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

    /**
     * Exports a detected face from input photo to the DetectedFaces folder.
     * @param imagePath Path to the original input image
     * @param faceRect The face rectangle to export
     * @param faceIndex The index of the face (1-based)
     */
    public static void exportDetectedFace(String imagePath, Rect faceRect, int faceIndex) {
        try {
            // Load the original image
            Mat image = opencv_imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("Failed to load image for export: " + imagePath);
                return;
            }
            
            try {
                // Extract the face region from the original image
                Mat faceRegion = new Mat(image, faceRect);
                
            // Use absolute path to ensure folder is created
            String exportFolder = "/Users/Arya/FaceRecognition/DetectedFaces";
            File exportDir = new File(exportFolder);
            if (!exportDir.exists()) {
                boolean created = exportDir.mkdirs();
                if (!created) {
                    System.err.println("    Warning: Failed to create DetectedFaces folder: " + exportFolder);
                } else {
                    System.out.println("    Created DetectedFaces folder: " + exportFolder);
                }
            }
            
            // Generate filename: originalName_faceN.jpg
            File imageFile = new File(imagePath);
            String baseName = imageFile.getName();
                String nameWithoutExt = baseName.replaceFirst("\\.[^.]+$", "");
                String ext = baseName.substring(baseName.lastIndexOf('.'));
                String outputFileName = String.format("%s_face%d%s", nameWithoutExt, faceIndex, ext);
                String outputPath = exportFolder + File.separator + outputFileName;
                
                // Save the face region
                opencv_imgcodecs.imwrite(outputPath, faceRegion);
                System.out.println("  Exported detected face to: " + outputPath);
                
                faceRegion.release();
            } finally {
                image.release();
            }
        } catch (Exception e) {
            System.err.println("Failed to export detected face: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Exports a face from input photo to the MatchedFaces folder.
     * @param imagePath Path to the original input image
     * @param faceRect The face rectangle to export
     * @param personName The matched person's name (or "Unmatched" if no match)
     * @param faceIndex The index of the face (1-based)
     * @param similarity The similarity score (or -1.0 if no match)
     */
    public static void exportMatchedFace(String imagePath, Rect faceRect, String personName, int faceIndex, double similarity) {
        try {
            // Load the original image
            Mat image = opencv_imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("Failed to load image for export: " + imagePath);
                return;
            }
            
            try {
                // Extract the face region from the original image
                Mat faceRegion = new Mat(image, faceRect);
                
                // Create export folder: FaceRecognition/MatchedFaces/personName/
                File imageFile = new File(imagePath);
                File projectRoot = imageFile.getParentFile().getParentFile(); // Go up from InputPhotos to FaceRecognition
                if (projectRoot == null || !projectRoot.getName().equals("FaceRecognition")) {
                    // Fallback: try to find FaceRecognition folder
                    File current = imageFile.getParentFile();
                    while (current != null && !current.getName().equals("FaceRecognition")) {
                        current = current.getParentFile();
                    }
                    if (current != null) {
                        projectRoot = current;
                    } else {
                        projectRoot = new File("/Users/Arya/FaceRecognition");
                    }
                }
                
                String exportFolder = projectRoot.getAbsolutePath() + File.separator + "MatchedFaces" + 
                                     File.separator + personName;
                File exportDir = new File(exportFolder);
                exportDir.mkdirs();
                
                // Generate filename: originalName_faceN_personName_similarity.jpg
                String baseName = imageFile.getName();
                String nameWithoutExt = baseName.replaceFirst("\\.[^.]+$", "");
                String ext = baseName.substring(baseName.lastIndexOf('.'));
                String outputFileName;
                if (similarity >= 0) {
                    outputFileName = String.format("%s_face%d_%s_%.4f%s", 
                        nameWithoutExt, faceIndex, personName, similarity, ext);
                } else {
                    // No match - just use face index
                    outputFileName = String.format("%s_face%d_%s%s", 
                        nameWithoutExt, faceIndex, personName, ext);
                }
                String outputPath = exportFolder + File.separator + outputFileName;
                
                // Save the face region
                opencv_imgcodecs.imwrite(outputPath, faceRegion);
                System.out.println("  Exported matched face to: " + outputPath);
                
                faceRegion.release();
            } finally {
                image.release();
            }
        } catch (Exception e) {
            System.err.println("Failed to export matched face: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
}