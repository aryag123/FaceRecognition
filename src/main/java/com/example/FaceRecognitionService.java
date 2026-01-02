// ===== FaceRecognitionService.java =====
package com.example;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
        
        // Return null if similarity is below threshold (0.325 for more lenient matching)
        // Typical face recognition thresholds: 0.5-0.6 for reasonable match, 0.7+ for confident match
        // Using 0.325 to avoid rejecting valid faces
        if (bestSimilarity < 0.325) {
            System.out.println("  -> Excluded (similarity " + String.format("%.4f", bestSimilarity) + " < 0.325 threshold)");
            return null;
        }
        
        return bestMatchPerson;
    }
    
    /**
     * Finds the best matching person for an input photo and returns both name and similarity.
     * Uses lenient matching to avoid rejecting valid faces.
     * @param inputPhoto The input photo to match
     * @param averageVectors Map of person names to their average embedding vectors
     * @return MatchResult containing person name and similarity score, or null if similarity < 0.325
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
        
        // Return null if similarity is below threshold (0.325 for lenient matching)
        // Removed ambiguity check to avoid rejecting valid faces
        if (bestSimilarity < 0.325) {
            System.out.println("  -> Excluded (similarity " + String.format("%.4f", bestSimilarity) + " < 0.325 threshold)");
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
     * Adds tags (person names) directly to the input photo's metadata.
     * Uses ExifTool to write IPTC keywords to the image file.
     * @param imagePath Path to the input image file
     * @param personNames List of person names to add as tags
     */
    private static void addTagsToImage(String imagePath, List<String> personNames) {
        if (personNames == null || personNames.isEmpty()) {
            return;
        }
        
        try {
            File imageFile = new File(imagePath);
            if (!imageFile.exists()) {
                System.err.println("Warning: Cannot add tags - image file does not exist: " + imagePath);
                return;
            }
            
            // Check if ExifTool is available
            ProcessBuilder checkProcess = new ProcessBuilder("which", "exiftool");
            checkProcess.redirectErrorStream(true);
            Process checkProc = checkProcess.start();
            
            // Consume output to avoid blocking
            java.io.BufferedReader checkReader = new java.io.BufferedReader(
                new java.io.InputStreamReader(checkProc.getInputStream()));
            while (checkReader.readLine() != null) {
                // Consume output
            }
            checkReader.close();
            
            int checkExit = checkProc.waitFor();
            
            if (checkExit != 0) {
                System.err.println("\nWarning: ExifTool is not installed. Cannot write tags to image file.");
                System.err.println("To install ExifTool on macOS, run: brew install exiftool");
                System.err.println("Falling back to sidecar file...");
                writeTagsToSidecarFile(imagePath, personNames);
                return;
            }
            
            System.out.println("  Writing tags to image file: " + imagePath);
            
            // Build ExifTool command to add keywords
            // ExifTool can add multiple keywords by repeating the -Keywords option
            List<String> command = new ArrayList<>();
            command.add("exiftool");
            command.add("-overwrite_original"); // Modify file in place (no backup)
            command.add("-P"); // Preserve file modification date
            
            // Add each person name as a keyword
            for (String personName : personNames) {
                command.add("-Keywords+=" + personName); // += adds to existing keywords
            }
            
            command.add(imagePath);
            
            // Execute ExifTool
            System.out.println("  Executing ExifTool command...");
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();
            
            // Read output
            java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
            reader.close();
            
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                System.out.println("  ExifTool completed successfully.");
                if (output.length() > 0) {
                    System.out.println("  ExifTool output: " + output.toString().trim());
                }
                
                // Also add macOS Finder tags for visibility in Finder
                System.out.println("  Adding macOS Finder tags...");
                addMacOSFinderTags(imagePath, personNames);
                
                System.out.println("\n✓ Added tags to input photo: " + String.join(", ", personNames));
                System.out.println("  Tags written to image file metadata (IPTC keywords + macOS Finder tags).");
            } else {
                System.err.println("ERROR: ExifTool failed to write tags. Exit code: " + exitCode);
                if (output.length() > 0) {
                    System.err.println("ExifTool output: " + output.toString());
                }
                System.err.println("Command was: " + String.join(" ", command));
                // Fallback: write to sidecar file
                writeTagsToSidecarFile(imagePath, personNames);
            }
            
        } catch (IOException | InterruptedException e) {
            System.err.println("Warning: Failed to add tags to image: " + e.getMessage());
            e.printStackTrace();
            // Fallback: write to sidecar file
            writeTagsToSidecarFile(imagePath, personNames);
        }
    }
    
    /**
     * Adds macOS Finder tags to the image file so they appear in Finder's right-click menu.
     * Uses the macOS 'tag' command or 'xattr' to set Finder tags.
     * @param imagePath Path to the input image file
     * @param personNames List of person names to add as Finder tags
     */
    private static void addMacOSFinderTags(String imagePath, List<String> personNames) {
        try {
            // Try using the 'tag' command first (if available)
            ProcessBuilder checkTag = new ProcessBuilder("which", "tag");
            checkTag.redirectErrorStream(true);
            Process checkProc = checkTag.start();
            
            // Consume output
            java.io.BufferedReader checkReader = new java.io.BufferedReader(
                new java.io.InputStreamReader(checkProc.getInputStream()));
            while (checkReader.readLine() != null) {
                // Consume output
            }
            checkReader.close();
            
            int checkExit = checkProc.waitFor();
            
            if (checkExit == 0) {
                // Use 'tag' command to add Finder tags
                // The tag command expects tags as a comma-separated string: tag --add "tag1,tag2" <file>
                List<String> tagCommand = new ArrayList<>();
                tagCommand.add("tag");
                tagCommand.add("--add");
                
                // Join all person names with commas (tag command expects comma-separated tags)
                String tagsString = String.join(",", personNames);
                tagCommand.add(tagsString);
                
                // File path comes after tags
                tagCommand.add(imagePath);
                
                System.out.println("    Executing: tag --add \"" + tagsString + "\" " + imagePath);
                ProcessBuilder tagProcess = new ProcessBuilder(tagCommand);
                tagProcess.redirectErrorStream(true);
                Process tagProc = tagProcess.start();
                
                // Read output
                java.io.BufferedReader tagReader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(tagProc.getInputStream()));
                StringBuilder tagOutput = new StringBuilder();
                String tagLine;
                while ((tagLine = tagReader.readLine()) != null) {
                    tagOutput.append(tagLine).append("\n");
                }
                tagReader.close();
                
                int tagExit = tagProc.waitFor();
                
                if (tagExit == 0) {
                    System.out.println("    macOS Finder tags command executed successfully.");
                    if (tagOutput.length() > 0) {
                        System.out.println("    Tag command output: " + tagOutput.toString().trim());
                    }
                    
                    // Verify tags were actually added by reading them back
                    try {
                        ProcessBuilder verifyProcess = new ProcessBuilder("tag", "--list", imagePath);
                        verifyProcess.redirectErrorStream(true);
                        Process verifyProc = verifyProcess.start();
                        java.io.BufferedReader verifyReader = new java.io.BufferedReader(
                            new java.io.InputStreamReader(verifyProc.getInputStream()));
                        String verifyLine = verifyReader.readLine();
                        verifyReader.close();
                        verifyProc.waitFor();
                        
                        if (verifyLine != null) {
                            boolean allTagsFound = true;
                            for (String personName : personNames) {
                                if (!verifyLine.contains(personName)) {
                                    allTagsFound = false;
                                    break;
                                }
                            }
                            if (allTagsFound) {
                                System.out.println("    ✓ Verified: All tags are present on file.");
                                System.out.println("    File tags: " + verifyLine);
                            } else {
                                System.err.println("    ⚠ Warning: Some tags may not have been added correctly.");
                                System.err.println("    Expected: " + String.join(", ", personNames));
                                System.err.println("    Found: " + verifyLine);
                            }
                        } else {
                            System.err.println("    ⚠ Warning: Could not verify tags (no output from tag --list).");
                        }
                    } catch (Exception verifyEx) {
                        System.err.println("    ⚠ Warning: Could not verify tags: " + verifyEx.getMessage());
                    }
                } else {
                    System.err.println("    ✗ ERROR: Tag command failed with exit code: " + tagExit);
                    if (tagOutput.length() > 0) {
                        System.err.println("    Tag command output: " + tagOutput.toString());
                    }
                    System.err.println("    Command was: tag --add \"" + tagsString + "\" " + imagePath);
                }
            } else {
                System.err.println("    Warning: 'tag' command not found.");
                // Fallback to xattr method
                addMacOSFinderTagsViaXattr(imagePath, personNames);
            }
        } catch (Exception e) {
            System.err.println("    Warning: Failed to add macOS Finder tags: " + e.getMessage());
            e.printStackTrace();
            // Try xattr method as fallback
            try {
                addMacOSFinderTagsViaXattr(imagePath, personNames);
            } catch (Exception e2) {
                System.err.println("    Warning: Failed to add Finder tags via xattr: " + e2.getMessage());
            }
        }
    }
    
    /**
     * Adds macOS Finder tags using xattr (extended attributes).
     * This is a fallback method when the 'tag' command is not available.
     * @param imagePath Path to the input image file
     * @param personNames List of person names to add as Finder tags
     */
    private static void addMacOSFinderTagsViaXattr(String imagePath, List<String> personNames) {
        try {
            // Use tag command via xattr - this requires the tag binary
            // For now, we'll use a simpler approach with tag command
            // If tag command doesn't work, we'll note that Finder tags require manual setup
            System.out.println("  Note: macOS Finder tags may require the 'tag' utility.");
            System.out.println("  Install with: brew install tag");
            System.out.println("  IPTC keywords are written and can be viewed with: exiftool -Keywords " + imagePath);
        } catch (Exception e) {
            System.err.println("Warning: Could not add Finder tags: " + e.getMessage());
        }
    }
    
    /**
     * Fallback: Writes tags to a sidecar .tags.txt file if ExifTool is not available.
     * @param imagePath Path to the input image file
     * @param personNames List of person names to add as tags
     */
    private static void writeTagsToSidecarFile(String imagePath, List<String> personNames) {
        try {
            File imageFile = new File(imagePath);
            String sidecarPath = imagePath + ".tags.txt";
            File sidecarFile = new File(sidecarPath);
            
            FileWriter writer = new FileWriter(sidecarFile);
            writer.write("# Face Recognition Tags\n");
            writer.write("# Generated automatically by FaceRecognition system\n");
            writer.write("# Each line below is a tag (matched person name)\n\n");
            
            // Write each person name as a tag (one per line)
            for (String personName : personNames) {
                writer.write(personName + "\n");
            }
            writer.close();
            
            System.out.println("Tags written to sidecar file: " + sidecarPath);
        } catch (IOException e) {
            System.err.println("Warning: Failed to write tags to sidecar file: " + e.getMessage());
            e.printStackTrace();
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
        String inputPhotosFolderPath = "/Users/Arya/FaceRecognition/InputPhotos";

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

            // Step 2: Get all image files from input photos folder
            File inputPhotosFolder = new File(inputPhotosFolderPath);
            if (!inputPhotosFolder.exists() || !inputPhotosFolder.isDirectory()) {
                System.err.println("Error: Input photos folder does not exist: " + inputPhotosFolderPath);
                return;
            }
            
            File[] imageFiles = inputPhotosFolder.listFiles((dir, name) -> {
                String lowerName = name.toLowerCase();
                return lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg") || 
                       lowerName.endsWith(".png") || lowerName.endsWith(".bmp");
            });
            
            if (imageFiles == null || imageFiles.length == 0) {
                System.err.println("Error: No image files found in input photos folder: " + inputPhotosFolderPath);
                return;
            }
            
            System.out.println("\nFound " + imageFiles.length + " image file(s) in input photos folder");
            System.out.println("================================================");
            
            // Process each image file
            int totalFacesDetected = 0;
            int totalFacesMatched = 0;
            
            for (int fileIndex = 0; fileIndex < imageFiles.length; fileIndex++) {
                String inputPhotoPath = imageFiles[fileIndex].getAbsolutePath();
                System.out.println("\n" + "=".repeat(50));
                System.out.println("Processing image " + (fileIndex + 1) + " of " + imageFiles.length + ": " + imageFiles[fileIndex].getName());
                System.out.println("=".repeat(50));
                
                // Step 3: Process input photo - detects and crops all faces
                // Input photo can have any number of faces (1, 2, 3, etc.)
                System.out.println("\nProcessing input photo (detecting and cropping faces)...");
                List<FaceInfo> inputFaces = service.processInputPhoto(inputPhotoPath);
                
                if (inputFaces.isEmpty()) {
                    System.out.println("  No faces detected in this photo. Skipping...");
                    continue;
                }
                
                System.out.println("  Found " + inputFaces.size() + " face(s) in this photo");
                totalFacesDetected += inputFaces.size();
                
                // Note: All detected faces (including rejected ones) are already exported in detectAllFaces()
                
                // Step 4: Find best match for each face (exclude faces with similarity < 0.325)
                System.out.println("\n  Comparing face vectors against average vectors...");
                
                List<MatchResult> validMatches = new ArrayList<>();
                
                for (int i = 0; i < inputFaces.size(); i++) {
                    FaceInfo faceInfo = inputFaces.get(i);
                    Photo inputPhoto = faceInfo.getPhoto();
                    System.out.println("\n  --- Face " + faceInfo.getFaceIndex() + " ---");
                    System.out.println("  Input: " + inputPhoto.getFileName());
                    
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
                    System.out.println("  Input embedding: dim=" + inputVector.length + 
                                     ", norm=" + String.format("%.4f", inputNorm) + 
                                     ", min=" + String.format("%.4f", inputMin) + 
                                     ", max=" + String.format("%.4f", inputMax));

                    // Find best match (returns null if similarity < 0.325)
                    String bestMatch = service.findBestMatch(inputPhoto, averageVectors);
                    
                    // Get the best match with similarity (even if < 0.325)
                    MatchResult matchResult = service.findBestMatchWithSimilarity(inputPhoto, averageVectors);
                    String exportPersonName;
                    double exportSimilarity;
                    
                    if (matchResult != null) {
                        exportPersonName = matchResult.getPersonName();
                        exportSimilarity = matchResult.getSimilarity();
                    } else {
                        // No match found or similarity < 0.325
                        exportPersonName = "Unmatched";
                        exportSimilarity = -1.0;
                    }

                    if (bestMatch != null) {
                        // Valid match (similarity >= 0.325)
                        validMatches.add(matchResult);
                        System.out.println("  Face " + faceInfo.getFaceIndex() + " best matches: " + bestMatch + 
                                         " (similarity: " + String.format("%.4f", matchResult.getSimilarity()) + ")");
                    } else {
                        System.out.println("  Face " + faceInfo.getFaceIndex() + " excluded (no match with similarity >= 0.325)");
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
                
                System.out.println("\n  === SUMMARY for " + imageFiles[fileIndex].getName() + " ===");
                System.out.println("  Detected " + inputFaces.size() + " face(s) in this photo");
                System.out.println("  Matched " + validMatches.size() + " face(s) (similarity >= 0.325)");
                if (validMatches.size() > 0) {
                    System.out.println("\n  Valid matches:");
                    for (int i = 0; i < validMatches.size(); i++) {
                        MatchResult match = validMatches.get(i);
                        System.out.println("    Face " + (i + 1) + ": " + match.getPersonName() + 
                                         " (similarity: " + String.format("%.4f", match.getSimilarity()) + ")");
                    }
                    
                    totalFacesMatched += validMatches.size();
                    
                    // Add tags to this input photo with matched person names
                    // Collect all unique person names from all matches (works with any number of people)
                    Set<String> uniquePersonNames = new HashSet<>();
                    for (MatchResult match : validMatches) {
                        uniquePersonNames.add(match.getPersonName());
                    }
                    
                    if (!uniquePersonNames.isEmpty()) {
                        System.out.println("\n  Adding tags to " + imageFiles[fileIndex].getName() + 
                                         " for " + uniquePersonNames.size() + 
                                         " unique person(s): " + String.join(", ", uniquePersonNames));
                        addTagsToImage(inputPhotoPath, new ArrayList<>(uniquePersonNames));
                    }
                }
            }
            
            // Final summary across all photos
            System.out.println("\n" + "=".repeat(50));
            System.out.println("=== FINAL SUMMARY (All Photos) ===");
            System.out.println("=".repeat(50));
            System.out.println("Processed " + imageFiles.length + " image file(s)");
            System.out.println("Total faces detected: " + totalFacesDetected);
            System.out.println("Total faces matched: " + totalFacesMatched + " (similarity >= 0.325)");
            
        } finally {
            // Cleanup detectors
            ImagePreprocessor.closeOnnxFaceDetector(); // If using ONNX RetinaFace
            ImagePreprocessor.closeLandmarkDetector(); // If using landmark detector
        }
    }
}