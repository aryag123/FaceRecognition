# Face Recognition Diagnosis

## Current Issue
Similarity scores between different people are too close (0.9777 vs 0.9738, difference of 0.0039), making it difficult to distinguish individuals.

## Root Cause Analysis

### 1. **Landmark Detection is Broken** ⚠️ CRITICAL
- Landmark coordinates are completely out of bounds (e.g., leftEye=(0.0,721.0) rightEye=(1798.0,721.0))
- This means face alignment is not working correctly
- **Impact**: Poor alignment causes all faces to look similar to the model

### 2. **Preprocessing May Not Match Model Requirements** ⚠️ LIKELY
Current preprocessing:
- BGR format (OpenCV default)
- Normalization: (pixel - 127.5) / 127.5 → [-1, 1]
- Size: 112x112
- RGB conversion: NOT being done

**Different ArcFace variants expect different preprocessing:**
- Some expect RGB, some expect BGR
- Some use (pixel - 127.5) / 128.0, others use / 127.5
- Some use mean/std normalization: (pixel - mean) / std

**Action**: Check the documentation/source of your specific ArcFace model to verify expected preprocessing.

### 3. **Embedding Model Quality** ⚠️ POSSIBLE
- The ArcFace model might not be well-trained or the wrong variant
- Some ArcFace models are trained on specific datasets and may not generalize well
- **Action**: Verify you're using a well-trained, general-purpose ArcFace model

### 4. **Face Detection Consistency** ✅ WORKING
- Face detection appears to be working (faces are being detected)
- However, inconsistent cropping (different face regions, angles) could cause issues

## Recommendations (in priority order)

### Immediate Fixes:
1. **Fix or Disable Landmark Detection**
   - The landmark detector output is clearly broken
   - Option A: Fix the coordinate parsing in `FaceLandmarkDetector.parseLandmarks()`
   - Option B: Disable landmark detection and use geometric estimation (already tested)
   - Option C: Try a different landmark detection model

2. **Verify Preprocessing Matches Model**
   - Check your ArcFace model's documentation or source
   - Common ArcFace preprocessing:
     - RGB conversion: YES/NO?
     - Normalization formula: (pixel - 127.5) / 128.0 OR / 127.5?
     - Mean/std values: Specific values?

3. **Test Different Preprocessing**
   - Try RGB instead of BGR
   - Try (pixel - 127.5) / 128.0 instead of / 127.5
   - Try simple [0, 1] normalization: pixel / 255.0

### Longer-term Solutions:
4. **Try a Different/Better Model**
   - Consider using a well-known, pre-trained ArcFace model
   - Models from InsightFace are typically well-documented
   - Ensure it's trained on diverse data

5. **Improve Face Alignment**
   - Use a reliable 5-point landmark detector (not 132-point)
   - Verify landmark coordinates are within image bounds
   - Test alignment quality by visualizing aligned faces

6. **Test with Standard Datasets**
   - Test the model on standard face recognition datasets to verify it works
   - If it doesn't work on standard datasets, the model itself is the issue

## Testing Strategy

1. **Test preprocessing variants:**
   ```java
   // Test 1: RGB + /128.0
   // Test 2: BGR + /127.5 (current)
   // Test 3: RGB + /127.5
   // Test 4: RGB + [0,1] normalization
   ```

2. **Test with alignment disabled** (already done - slightly better)
3. **Test with proper alignment** (needs landmark fix)
4. **Visualize preprocessed images** to verify they look correct
5. **Compare embeddings directly** to see if they're actually similar or just normalized wrong

## Most Likely Issues (Ranked)

1. **Landmark detection broken** → Poor alignment → Similar embeddings
2. **Preprocessing mismatch** → Model sees wrong input → Poor discrimination
3. **Model quality** → Model itself not discriminative enough

