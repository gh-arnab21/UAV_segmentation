import cv2
import numpy as np

# Hazard indices
BACKGROUND, ROCK_FACE, CRACK, OVERHANG, LOOSE_BLOCK, SEEPAGE = range(6)

def extract_features(mask):
    features = {}
    features["crack_density"] = np.sum(mask == CRACK) / mask.size
    features["overhang_area"] = np.sum(mask == OVERHANG)
    features["seepage_ratio"] = np.sum(mask == SEEPAGE) / mask.size
    num_blocks, _ = cv2.connectedComponents((mask == LOOSE_BLOCK).astype(np.uint8))
    features["loose_block_count"] = num_blocks
    return features

# Example usage
if __name__ == "__main__":
    mask = cv2.imread("example_mask.png", 0)  # grayscale mask with class indices
    feats = extract_features(mask)
    print(feats)
