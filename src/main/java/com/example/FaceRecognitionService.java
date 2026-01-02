// ===== FaceRecognitionService.java =====
package com.example;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OrtException;

public class FaceRecognitionService implements AutoCloseable {

    private final FaceEmbeddingModel model;

    public FaceRecognitionService(String onnxModelPath) throws OrtException {
        this.model = new FaceEmbeddingModel(onnxModelPath);
    }

    /**
     * Processes a reference photo (expected to have exactly 1 face).
     * Detects only the largest face.
     * @param imagePath Path to the image file
     * @return Photo object for the detected face, or null if no face found
     */
    public Photo processReferencePhoto(String imagePath) throws Exception {
        // For reference photos, use the old method that detects only 1 face (largest)
        float[] imageData = ImagePreprocessor.preprocessImage(imagePath);
        
        float[] embedding = model.computeEmbedding(
                imageData,
                ImagePreprocessor.CHANNELS,
                ImagePreprocessor.TARGET_HEIGHT,
                ImagePreprocessor.TARGET_WIDTH
        );
        
        return new Photo(new File(imagePath).getName(), embedding);
    }
    
    /**
     * Helper class to store face information with its rectangle.
     */
    public static class FaceInfo {
        private final Photo photo;
        private final org.bytedeco.opencv.opencv_core.Rect faceRect;
        private final int faceIndex;
        
        public FaceInfo(Photo photo, org.bytedeco.opencv.opencv_core.Rect faceRect, int faceIndex) {
            this.photo = photo;
            this.faceRect = faceRect;
            this.faceIndex = faceIndex;
        }
        
        public Photo getPhoto() {
            return photo;
        }
        
        public org.bytedeco.opencv.opencv_core.Rect getFaceRect() {
            return faceRect;
        }
        
        public int getFaceIndex() {
            return faceIndex;
        }
    }
    
    /**
     * Processes an input photo and returns a list of FaceInfo objects, one for each detected face.
     * Detects ALL faces in the image.
     * @param imagePath Path to the image file
     * @return List of FaceInfo objects (one per detected face), empty list if no faces found
     */
    public List<FaceInfo> processInputPhoto(String imagePath) throws Exception {
        List<FaceInfo> faceInfos = new ArrayList<>();
        
        // Detect all faces in the image
        java.util.List<org.bytedeco.opencv.opencv_core.Rect> faceRects = ImagePreprocessor.detectAllFaces(imagePath);
        
        if (faceRects == null || faceRects.isEmpty()) {
            return faceInfos; // Return empty list if no faces found
        }
        
        System.out.println("    Found " + faceRects.size() + " face(s) in " + new File(imagePath).getName());
        
        // Process each detected face
        String baseFileName = new File(imagePath).getName();
        
        for (int i = 0; i < faceRects.size(); i++) {
            org.bytedeco.opencv.opencv_core.Rect faceRect = faceRects.get(i);
            
            try {
                // Preprocess this specific face
                float[] imageData = ImagePreprocessor.preprocessFace(imagePath, faceRect);
                
                // Compute embedding
                float[] embedding = model.computeEmbedding(
                        imageData,
                        ImagePreprocessor.CHANNELS,
                        ImagePreprocessor.TARGET_HEIGHT,
                        ImagePreprocessor.TARGET_WIDTH
                );
                
                // Create Photo object with face index in filename if multiple faces
                String photoFileName = faceRects.size() > 1 ? 
                    baseFileName.replaceFirst("(\\.(jpg|jpeg|png|bmp))$", "_face" + (i + 1) + "$1") : 
                    baseFileName;
                
                Photo photo = new Photo(photoFileName, embedding);
                faceInfos.add(new FaceInfo(photo, faceRect, i + 1));
            } catch (Exception e) {
                System.err.println("    Failed to process face " + (i + 1) + ": " + e.getMessage());
            }
        }
        
        return faceInfos;
    }

    /**
     * Loads photos from subfolders, groups them by person (folder name),
     * and calculates average embedding for each person.
     * @param folderPath Path to the ReferencePhotos folder containing person subfolders
     * @return Map of person name to their average embedding vector
     */
    public Map<String, Photo> loadPhotosByPerson(String folderPath) throws Exception {
        File folder = new File(folderPath);

        if (!folder.exists() || !folder.isDirectory()) {
            throw new IllegalArgumentException("Invalid folder: " + folderPath);
        }

        File[] subfolders = folder.listFiles();
        if (subfolders == null) {
            throw new IllegalArgumentException("Cannot read folder: " + folderPath);
        }

        Map<String, Photo> averageVectors = new HashMap<>();

        for (File personFolder : subfolders) {
            // Skip if not a directory or hidden
            if (!personFolder.isDirectory() || personFolder.getName().startsWith(".")) {
                continue;
            }

            String personName = personFolder.getName();
            System.out.println("\n  Processing person: " + personName);

            // Load all photos for this person
            List<Photo> personPhotos = new ArrayList<>();
            File[] files = personFolder.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (!file.isFile()) {
                        continue;
                    }

                    // Skip hidden files
                    if (file.getName().startsWith(".")) {
                        continue;
                    }

                    // Only process image files
                    String fileName = file.getName().toLowerCase();
                    if (!fileName.endsWith(".jpg") && !fileName.endsWith(".jpeg") 
                            && !fileName.endsWith(".png") && !fileName.endsWith(".bmp")) {
                        continue;
                    }

                    try {
                        // Reference photos have exactly 1 face - use single face detection
                        Photo photo = processReferencePhoto(file.getAbsolutePath());
                        if (photo != null) {
                            personPhotos.add(photo);
                            System.out.println("    Processed: " + file.getName());
                        } else {
                            System.err.println("    No face found in " + file.getName());
                        }
                    } catch (Exception e) {
                        System.err.println("    Failed to process " + file.getName() + ": " + e.getMessage());
                    }
                }
            }

            if (personPhotos.isEmpty()) {
                System.out.println("    Warning: No photos found for " + personName);
                continue;
            }

            // Calculate average embedding for this person
            Photo averagePhoto = calculateAverageEmbedding(personName, personPhotos);
            averageVectors.put(personName, averagePhoto);
            
            // Debug: Calculate intra-person similarity (how similar are photos of the same person?)
            if (personPhotos.size() > 1) {
                double minSim = 1.0, maxSim = 0.0, avgSim = 0.0;
                int comparisons = 0;
                for (int i = 0; i < personPhotos.size(); i++) {
                    for (int j = i + 1; j < personPhotos.size(); j++) {
                        double sim = personPhotos.get(i).cosineSimilarity(personPhotos.get(j));
                        minSim = Math.min(minSim, sim);
                        maxSim = Math.max(maxSim, sim);
                        avgSim += sim;
                        comparisons++;
                    }
                }
                if (comparisons > 0) {
                    avgSim /= comparisons;
                    System.out.println("    Averaged " + personPhotos.size() + " photos for " + personName + 
                                     " (intra-person similarity: min=" + String.format("%.4f", minSim) + 
                                     ", avg=" + String.format("%.4f", avgSim) + 
                                     ", max=" + String.format("%.4f", maxSim) + ")");
                } else {
                    System.out.println("    Averaged " + personPhotos.size() + " photos for " + personName);
                }
            } else {
                System.out.println("    Averaged " + personPhotos.size() + " photos for " + personName);
            }
        }

        return averageVectors;
    }

    /**
     * Calculates the average embedding vector from multiple photos of the same person.
     * Averages the vectors component-wise, then normalizes the result (L2 normalization).
     */
    private Photo calculateAverageEmbedding(String personName, List<Photo> photos) {
        if (photos == null || photos.isEmpty()) {
            throw new IllegalArgumentException("Photos list cannot be null or empty");
        }

        int vectorDim = photos.get(0).getVector().length;
        float[] averageVector = new float[vectorDim];

        // Sum all vectors component-wise
        for (Photo photo : photos) {
            float[] vector = photo.getVector();
            if (vector.length != vectorDim) {
                throw new IllegalArgumentException("All photos must have the same vector dimension");
            }
            for (int i = 0; i < vectorDim; i++) {
                averageVector[i] += vector[i];
            }
        }

        // Divide by count to get average
        int count = photos.size();
        for (int i = 0; i < vectorDim; i++) {
            averageVector[i] /= count;
        }

        // L2 normalize the average vector (important for cosine similarity with normalized vectors)
        double norm = 0.0;
        for (int i = 0; i < vectorDim; i++) {
            norm += averageVector[i] * averageVector[i];
        }
        norm = Math.sqrt(norm);
        if (norm > 0.0) {
            for (int i = 0; i < vectorDim; i++) {
                averageVector[i] /= norm;
            }
        }

        return new Photo(personName, averageVector);
    }

    @Deprecated
    public Photo[] loadPhotosFromFolder(String folderPath) throws Exception {
        File folder = new File(folderPath);

        if (!folder.exists() || !folder.isDirectory()) {
            throw new IllegalArgumentException("Invalid folder: " + folderPath);
        }

        File[] files = folder.listFiles();
        if (files == null) {
            throw new IllegalArgumentException("Cannot read folder: " + folderPath);
        }

        List<Photo> photoList = new ArrayList<>();

        for (File file : files) {
            if (!file.isFile()) {
                continue;
            }

            // Skip hidden files (like .DS_Store on macOS)
            if (file.getName().startsWith(".")) {
                continue;
            }

            // Only process image files
            String fileName = file.getName().toLowerCase();
            if (!fileName.endsWith(".jpg") && !fileName.endsWith(".jpeg") 
                    && !fileName.endsWith(".png") && !fileName.endsWith(".bmp")) {
                continue;
            }

            try {
                // Reference photos have exactly 1 face - use single face detection
                Photo photo = processReferencePhoto(file.getAbsolutePath());
                if (photo != null) {
                    photoList.add(photo);
                    System.out.println("Processed " + file.getName());
                } else {
                    System.err.println("No face found in " + file.getName());
                }
            } catch (Exception e) {
                System.err.println("Failed to process " + file.getName() + ": " + e.getMessage());
            }
        }

        return photoList.toArray(new Photo[0]);
    }

    /**
     * Finds the best matching person by comparing input photo against average embeddings.
     * @param inputPhoto The input photo to match
     * @param averageVectors Map of person name to their average embedding vector
     * @return The name of the best matching person
     */
    public String findBestMatch(Photo inputPhoto, Map<String, Photo> averageVectors) {
        if (averageVectors == null || averageVectors.isEmpty()) {
            throw new IllegalArgumentException("No reference vectors provided");
        }

        String bestMatchPerson = null;
        double bestSimilarity = -1.0;

        System.out.println("\nSimilarity scores (compared against average vectors):");
        for (Map.Entry<String, Photo> entry : averageVectors.entrySet()) {
            String personName = entry.getKey();
            Photo averageVector = entry.getValue();
            double similarity = inputPhoto.cosineSimilarity(averageVector);
            System.out.println("  " + personName + ": " + String.format("%.4f", similarity));

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatchPerson = personName;
            }
        }

        System.out.println("\nBest match: " + bestMatchPerson +
                " (similarity: " + String.format("%.4f", bestSimilarity) + ")");
        
        // Return null if similarity is below threshold (0.4 for more lenient matching)
        // Typical face recognition thresholds: 0.5-0.6 for reasonable match, 0.7+ for confident match
        // Using 0.4 to avoid rejecting valid faces
        if (bestSimilarity < 0.4) {
            System.out.println("  -> Excluded (similarity " + String.format("%.4f", bestSimilarity) + " < 0.4 threshold)");
            return null;
        }
        
        return bestMatchPerson;
    }
    
    /**
     * Finds the best matching person for an input photo and returns both name and similarity.
     * Uses lenient matching to avoid rejecting valid faces.
     * @param inputPhoto The input photo to match
     * @param averageVectors Map of person names to their average embedding vectors
     * @return MatchResult containing person name and similarity score, or null if similarity < 0.4
     */
    public MatchResult findBestMatchWithSimilarity(Photo inputPhoto, Map<String, Photo> averageVectors) {
        if (averageVectors == null || averageVectors.isEmpty()) {
            throw new IllegalArgumentException("No reference vectors provided");
        }

        String bestMatchPerson = null;
        double bestSimilarity = -1.0;

        // Find best match
        for (Map.Entry<String, Photo> entry : averageVectors.entrySet()) {
            String personName = entry.getKey();
            Photo averageVector = entry.getValue();
            double similarity = inputPhoto.cosineSimilarity(averageVector);

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatchPerson = personName;
            }
        }
        
        // Return null if similarity is below threshold (0.4 for lenient matching)
        // Removed ambiguity check to avoid rejecting valid faces
        if (bestSimilarity < 0.4) {
            System.out.println("  -> Excluded (similarity " + String.format("%.4f", bestSimilarity) + " < 0.4 threshold)");
            return null;
        }
        
        return new MatchResult(bestMatchPerson, bestSimilarity);
    }
    
    /**
     * Helper class to hold match result with similarity score.
     */
    public static class MatchResult {
        private final String personName;
        private final double similarity;
        
        public MatchResult(String personName, double similarity) {
            this.personName = personName;
            this.similarity = similarity;
        }
        
        public String getPersonName() {
            return personName;
        }
        
        public double getSimilarity() {
            return similarity;
        }
    }

    @Deprecated
    public String findBestMatch(Photo inputPhoto, Photo[] referencePhotos) {
        if (referencePhotos == null || referencePhotos.length == 0) {
            throw new IllegalArgumentException("No reference photos provided");
        }

        String bestMatchFileName = null;
        double bestSimilarity = -1.0;

        System.out.println("\nSimilarity scores:");
        for (Photo reference : referencePhotos) {
            double similarity = inputPhoto.cosineSimilarity(reference);
            System.out.println("  " + reference.getFileName() + ": " + String.format("%.4f", similarity));

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatchFileName = reference.getFileName();
            }
        }

        System.out.println("\nBest match: " + bestMatchFileName +
                " (similarity: " + String.format("%.4f", bestSimilarity) + ")");

        return bestMatchFileName;
    }

    @Override
    public void close() throws Exception {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Clears the MatchedFaces folder by deleting all files and subdirectories.
     */
    private static void clearMatchedFacesFolder() {
        try {
            File matchedFacesFolder = new File("/Users/Arya/FaceRecognition/MatchedFaces");
            if (matchedFacesFolder.exists() && matchedFacesFolder.isDirectory()) {
                System.out.println("Clearing MatchedFaces folder...");
                deleteDirectory(matchedFacesFolder);
                System.out.println("MatchedFaces folder cleared.");
            } else {
                // Folder doesn't exist yet, will be created when exporting
                System.out.println("MatchedFaces folder does not exist yet.");
            }
        } catch (Exception e) {
            System.err.println("Warning: Failed to clear MatchedFaces folder: " + e.getMessage());
        }
    }
    
    /**
     * Clears the DetectedFaces folder by deleting all files and subdirectories.
     */
    private static void clearDetectedFacesFolder() {
        try {
            File detectedFacesFolder = new File("/Users/Arya/FaceRecognition/DetectedFaces");
            if (detectedFacesFolder.exists() && detectedFacesFolder.isDirectory()) {
                System.out.println("Clearing DetectedFaces folder...");
                deleteDirectory(detectedFacesFolder);
                System.out.println("DetectedFaces folder cleared.");
            } else {
                // Folder doesn't exist yet, will be created when exporting
                System.out.println("DetectedFaces folder does not exist yet.");
            }
        } catch (Exception e) {
            System.err.println("Warning: Failed to clear DetectedFaces folder: " + e.getMessage());
        }
    }
    
    /**
     * Recursively deletes a directory and all its contents.
     */
    private static void deleteDirectory(File directory) {
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            directory.delete();
        }
    }

    // ===== MAIN METHOD - Everything happens here =====
    public static void main(String[] args) throws Exception {
        String modelPath = "/Users/Arya/FaceRecognition/models/webface_r50_pfc.onnx";         
        String cascadePath = "/Users/Arya/FaceRecognition/models/haarcascade_frontalface_default.xml"; // Download from OpenCV
        String referenceFolderPath = "/Users/Arya/FaceRecognition/ReferencePhotos";
        String inputPhotoPath = "/Users/Arya/FaceRecognition/InputPhotos/aryaaayush4.jpg";

        // Initialize face detector FIRST
        // Option 1: Use XML Cascade (currently active - more accurate for this use case)
        System.out.println("Initializing face detector...");
        ImagePreprocessor.initFaceDetector(cascadePath);
        
        // Option 2: Use ONNX RetinaFace (anchor decoding needs refinement)
        // String retinaFacePath = "/Users/Arya/FaceRecognition/models/retinaface.onnx";
        // System.out.println("Initializing RetinaFace detector...");
        // ImagePreprocessor.initOnnxFaceDetector(retinaFacePath);

        // Initialize landmark detector for false positive filtering
        // The landmark detector will validate detected faces and filter out false positives
        String landmarkModelPath = "/Users/Arya/FaceRecognition/models/Facial-Landmark-Detection.onnx";
        if (landmarkModelPath != null && !landmarkModelPath.isEmpty()) {
            System.out.println("Initializing landmark detector for false positive filtering...");
            try {
                ImagePreprocessor.initLandmarkDetector(landmarkModelPath);
                System.out.println("Landmark detector initialized successfully. False positive filtering enabled.");
            } catch (Exception e) {
                System.err.println("Warning: Failed to initialize landmark detector: " + e.getMessage());
                System.err.println("Continuing without landmark-based false positive filtering...");
                landmarkModelPath = null; // Disable if initialization fails
            }
        } else {
            System.out.println("No landmark detector model specified - false positive filtering disabled");
        }

        try (FaceRecognitionService service = new FaceRecognitionService(modelPath)) {

            // Step 0: Clear export folders before processing
            clearMatchedFacesFolder();
            clearDetectedFacesFolder();

            // Step 1: Load photos from person subfolders and calculate average embeddings
            System.out.println("\nLoading reference photos from person folders (detecting and cropping faces)...");
            Map<String, Photo> averageVectors = service.loadPhotosByPerson(referenceFolderPath);

            System.out.println("\nLoaded average vectors for " + averageVectors.size() + " persons:");
            for (String personName : averageVectors.keySet()) {
                System.out.println("  - " + personName);
            }

            // Step 2: Process input photo - detects and crops all faces
            // Input photo can have any number of faces (1, 2, 3, etc.)
            System.out.println("\nProcessing input photo (detecting and cropping faces)...");
            List<FaceInfo> inputFaces = service.processInputPhoto(inputPhotoPath);
            
            if (inputFaces.isEmpty()) {
                System.out.println("Error: No faces detected in input photo!");
                return;
            }
            
            System.out.println("Found " + inputFaces.size() + " face(s) in input photo");
            
            // Note: All detected faces (including rejected ones) are already exported in detectAllFaces()
            
            // Step 3: Find best match for each face (exclude faces with similarity < 0.4)
            System.out.println("\nComparing face vectors against average vectors...");
            
            List<MatchResult> validMatches = new ArrayList<>();
            
            for (int i = 0; i < inputFaces.size(); i++) {
                FaceInfo faceInfo = inputFaces.get(i);
                Photo inputPhoto = faceInfo.getPhoto();
                System.out.println("\n--- Face " + faceInfo.getFaceIndex() + " ---");
                System.out.println("Input: " + inputPhoto.getFileName());
                
                // Debug: Print embedding statistics
                float[] inputVector = inputPhoto.getVector();
                double inputNorm = 0.0;
                double inputMin = Double.MAX_VALUE, inputMax = Double.MIN_VALUE;
                for (float v : inputVector) {
                    inputNorm += v * v;
                    inputMin = Math.min(inputMin, v);
                    inputMax = Math.max(inputMax, v);
                }
                inputNorm = Math.sqrt(inputNorm);
                System.out.println("Input embedding: dim=" + inputVector.length + 
                                 ", norm=" + String.format("%.4f", inputNorm) + 
                                 ", min=" + String.format("%.4f", inputMin) + 
                                 ", max=" + String.format("%.4f", inputMax));

                // Find best match (returns null if similarity < 0.4)
                String bestMatch = service.findBestMatch(inputPhoto, averageVectors);
                
                // Get the best match with similarity (even if < 0.4)
                MatchResult matchResult = service.findBestMatchWithSimilarity(inputPhoto, averageVectors);
                String exportPersonName;
                double exportSimilarity;
                
                if (matchResult != null) {
                    exportPersonName = matchResult.getPersonName();
                    exportSimilarity = matchResult.getSimilarity();
                } else {
                    // No match found or similarity < 0.4
                    exportPersonName = "Unmatched";
                    exportSimilarity = -1.0;
                }

                if (bestMatch != null) {
                    // Valid match (similarity >= 0.4)
                    validMatches.add(matchResult);
                    System.out.println("Face " + faceInfo.getFaceIndex() + " best matches: " + bestMatch + 
                                     " (similarity: " + String.format("%.4f", matchResult.getSimilarity()) + ")");
                } else {
                    System.out.println("Face " + faceInfo.getFaceIndex() + " excluded (no match with similarity >= 0.4)");
                }
                
                // Export all faces (matched or unmatched)
                ImagePreprocessor.exportMatchedFace(
                    inputPhotoPath,
                    faceInfo.getFaceRect(),
                    exportPersonName,
                    faceInfo.getFaceIndex(),
                    exportSimilarity
                );
            }
            
            System.out.println("\n=== SUMMARY ===");
            System.out.println("Detected " + inputFaces.size() + " face(s) in input photo");
            System.out.println("Matched " + validMatches.size() + " face(s) (similarity >= 0.4)");
            if (validMatches.size() > 0) {
                System.out.println("\nValid matches:");
                for (int i = 0; i < validMatches.size(); i++) {
                    MatchResult match = validMatches.get(i);
                    System.out.println("  Face " + (i + 1) + ": " + match.getPersonName() + 
                                     " (similarity: " + String.format("%.4f", match.getSimilarity()) + ")");
                }
            }
        } finally {
            // Cleanup detectors
            ImagePreprocessor.closeOnnxFaceDetector(); // If using ONNX RetinaFace
            ImagePreprocessor.closeLandmarkDetector(); // If using landmark detector
        }
    }
}