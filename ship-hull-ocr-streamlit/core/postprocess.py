import re
from core.config import CONFIG

def clean_ship_name(text: str) -> str:
    if not CONFIG["postprocess"].get("clean", True):
        return text
    text = re.sub(r'[^A-Z0-9\s\-\']', '', text.upper())
    return ' '.join(text.split())

def extract_best_name(ocr_result) -> tuple[str, float]:
    if not ocr_result:
        return "", 0.0

    lines = [(box, txt, score) for box, (txt, score) in ocr_result]
    lines = [l for l in lines if l[2] >= CONFIG["postprocess"]["min_conf"]]

    if not lines:
        return "", 0.0

    lines.sort(key=lambda x: (x[0][0][1], x[0][0][0]))  # top-left sort

    texts = [txt for _, txt, _ in lines]
    scores = [score for _, _, score in lines]

    name = " ".join(texts) if CONFIG["postprocess"].get("merge_nearby", True) else texts[max(range(len(scores)), key=scores.__getitem__)]
    name = clean_ship_name(name)
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return name.strip(), round(avg_score, 3)