import cv2
import numpy as np
from core.config import CONFIG

def prepare_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > CONFIG["preprocess"]["max_size"]:
        scale = CONFIG["preprocess"]["max_size"] / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if CONFIG["preprocess"].get("clahe", False):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if CONFIG["preprocess"].get("sharpen", False):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)

    return img