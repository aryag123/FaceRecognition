// ===== FaceEmbeddingModel.java =====
package com.example;

import ai.onnxruntime.*;
import java.nio.FloatBuffer;

public class FaceEmbeddingModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Object lock = new Object();

    public FaceEmbeddingModel(String onnxModelPath) throws OrtException {
        if (onnxModelPath == null || onnxModelPath.isEmpty()) {
            throw new IllegalArgumentException("ONNX model path cannot be null or empty");
        }

        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(onnxModelPath, new OrtSession.SessionOptions());
        
        // Print model metadata for debugging
        System.out.println("\n=== Model Information ===");
        System.out.println("Model path: " + onnxModelPath);
        System.out.println("Input names: " + session.getInputNames());
        System.out.println("Output names: " + session.getOutputNames());
        System.out.println("=======================\n");
    }

    public float[] computeEmbedding(float[] imageData, int channels, int height, int width)
            throws OrtException {
        if (imageData == null || imageData.length != channels * height * width) {
            throw new IllegalArgumentException("Invalid image data dimensions");
        }

        synchronized (lock) {
            long[] shape = new long[]{1, channels, height, width};

            // Try to get the actual input name from the model
            String inputName = session.getInputNames().iterator().next();
            
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), shape)) {
                try (OrtSession.Result result = session.run(
                        java.util.Collections.singletonMap(inputName, tensor))) {
                    // Get the first output (most models have one output)
                    String outputName = session.getOutputNames().iterator().next();
                    OnnxValue outputValue = result.get(outputName).orElseThrow(() -> 
                        new OrtException("Output " + outputName + " not found in result"));
                    Object rawValue = outputValue.getValue();
                    
                    float[] embedding;
                    if (rawValue instanceof float[][]) {
                        // Shape [1, embedding_dim] or [batch, embedding_dim]
                        float[][] embedding2d = (float[][]) rawValue;
                        embedding = embedding2d[0]; // Get first (and likely only) batch item
                    } else if (rawValue instanceof float[]) {
                        // Shape [embedding_dim]
                        embedding = (float[]) rawValue;
                    } else {
                        throw new OrtException("Unexpected embedding output format: " + rawValue.getClass().getName());
                    }
                    
                    // ArcFace models typically output already normalized embeddings
                    // But we normalize again to be safe (normalization is idempotent)
                    return normalizeEmbedding(embedding);
                }
            }
        }
    }

    /**
     * L2 normalizes the embedding vector for proper cosine similarity calculation.
     * This is critical for ArcFace models.
     */
    private float[] normalizeEmbedding(float[] embedding) {
        // Compute L2 norm
        double norm = 0.0;
        for (float value : embedding) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);

        // Normalize (avoid division by zero)
        if (norm < 1e-10) {
            return embedding.clone(); // Return original if norm is too small
        }

        float[] normalized = new float[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            normalized[i] = (float)(embedding[i] / norm);
        }
        return normalized;
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
