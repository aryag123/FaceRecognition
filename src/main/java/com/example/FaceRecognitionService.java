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

    public Photo processImage(String imagePath) throws Exception {
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
                        Photo photo = processImage(file.getAbsolutePath());
                        personPhotos.add(photo);
                        System.out.println("    Processed: " + file.getName());
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
            System.out.println("    Averaged " + personPhotos.size() + " photos for " + personName);
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
                photoList.add(processImage(file.getAbsolutePath()));
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

        return bestMatchPerson;
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

    // ===== MAIN METHOD - Everything happens here =====
    public static void main(String[] args) throws Exception {
        String modelPath = "/Users/Arya/FaceRecognition/models/arcface.onnx";            
        String cascadePath = "/Users/Arya/FaceRecognition/models/haarcascade_frontalface_default.xml"; // Download from OpenCV
        String referenceFolderPath = "/Users/Arya/FaceRecognition/ReferencePhotos";
        String inputPhotoPath = "/Users/Arya/Downloads/arya1.jpg";

        // Initialize face detector FIRST
        System.out.println("Initializing face detector...");
        ImagePreprocessor.initFaceDetector(cascadePath);

        // Initialize landmark detector (optional - set to null to use estimated landmarks)
        // DISABLED: Landmark detector is broken (produces invalid coordinates)
        String landmarkModelPath = null; // Using estimated landmarks instead
        if (landmarkModelPath != null) {
            System.out.println("Initializing landmark detector...");
            ImagePreprocessor.initLandmarkDetector(landmarkModelPath);
        } else {
            System.out.println("Using estimated landmarks (no landmark detector model specified)");
        }

        try (FaceRecognitionService service = new FaceRecognitionService(modelPath)) {

            // Step 1: Load photos from person subfolders and calculate average embeddings
            System.out.println("\nLoading reference photos from person folders (detecting and cropping faces)...");
            Map<String, Photo> averageVectors = service.loadPhotosByPerson(referenceFolderPath);

            System.out.println("\nLoaded average vectors for " + averageVectors.size() + " persons:");
            for (String personName : averageVectors.keySet()) {
                System.out.println("  - " + personName);
            }

            // Step 2: Process input photo - detects and crops face
            System.out.println("\nProcessing input photo (detecting and cropping face)...");
            Photo inputPhoto = service.processImage(inputPhotoPath);
            System.out.println("Input: " + inputPhoto.getFileName());

            // Step 3: Find best match by comparing against average vectors
            System.out.println("\nComparing face vector against average vectors...");
            String bestMatch = service.findBestMatch(inputPhoto, averageVectors);

            System.out.println("\n=== RESULT ===");
            System.out.println("Input photo '" + inputPhoto.getFileName() +
                    "' best matches: " + bestMatch);
        } finally {
            // Cleanup landmark detector if it was initialized
            ImagePreprocessor.closeLandmarkDetector();
        }
    }
}