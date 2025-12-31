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
    }

    public float[] computeEmbedding(float[] imageData, int channels, int height, int width)
            throws OrtException {
        if (imageData == null || imageData.length != channels * height * width) {
            throw new IllegalArgumentException("Invalid image data dimensions");
        }

        synchronized (lock) {
            long[] shape = new long[]{1, channels, height, width};

            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), shape)) {
                try (OrtSession.Result result = session.run(
                        java.util.Collections.singletonMap("data", tensor))) {
                    float[] embedding = ((float[][]) result.get(0).getValue())[0];
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
