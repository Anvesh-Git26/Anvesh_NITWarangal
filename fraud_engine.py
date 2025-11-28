import cv2
import numpy as np

def detect_fraud_advanced(image_path):
    # Load image
    orig = cv2.imread(image_path)
    if orig is None: return {"fraud_score": 0, "tampering_detected": False}
    
    try:
        # Simulate ELA (Error Level Analysis)
        cv2.imwrite("temp_ela.jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread("temp_ela.jpg")
        
        # Calculate difference
        diff = cv2.absdiff(orig, compressed)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Robust Scoring (Top 5% pixels)
        flattened = diff.flatten()
        flattened.sort()
        top_5_percent = int(len(flattened) * 0.05)
        
        if top_5_percent > 0:
            robust_score = np.mean(flattened[-top_5_percent:])
        else:
            robust_score = 0
        
        return {
            "fraud_score": round(float(robust_score), 2),
            "tampering_detected": robust_score > 15.0
        }
    except Exception as e:
        print(f"Fraud check error: {e}")
        return {"fraud_score": 0, "tampering_detected": False}
