import cv2
import numpy as np
from core.config import CONFIG

def draw_ocr_boxes(img: np.ndarray, ocr_result) -> np.ndarray:
    img_copy = img.copy()
    for box, (txt, score) in ocr_result:
        if score >= CONFIG["postprocess"]["min_conf"]:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], True, (0, 255, 0), 3)
            x, y = int(box[0][0]), int(box[0][1])
            cv2.putText(img_copy, f"{txt} ({score:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return img_copy