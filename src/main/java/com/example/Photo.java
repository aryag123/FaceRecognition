// ===== Photo.java =====
package com.example;

import java.util.Arrays;
import java.util.Objects;

public class Photo {
    private final String fileName;
    private final float[] vector;

    public Photo(String fileName, float[] vector) {
        if (fileName == null || fileName.isEmpty()) {
            throw new IllegalArgumentException("fileName cannot be null or empty");
        }
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("vector cannot be null or empty");
        }

        this.fileName = fileName;
        this.vector = vector.clone();
    }

    public String getFileName() {
        return fileName;
    }

    public float[] getVector() {
        return vector.clone();
    }

    public double cosineSimilarity(Photo other) {
        if (other == null) {
            throw new IllegalArgumentException("other photo cannot be null");
        }

        float[] v1 = this.vector;
        float[] v2 = other.vector;

        if (v1.length != v2.length) {
            throw new IllegalArgumentException(
                    "Vector dimensions don't match: " + v1.length + " vs " + v2.length
            );
        }

        double dot = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        double denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        if (denominator == 0.0) {
            return 0.0;
        }

        return dot / denominator;
    }

    @Override
    public String toString() {
        return "Photo{fileName='" + fileName + "', vectorDim=" + vector.length + "}";
    }
}