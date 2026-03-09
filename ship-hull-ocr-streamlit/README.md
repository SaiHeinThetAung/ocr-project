# Ship Hull OCR

**Extract ship names from hull photos using PaddleOCR + Streamlit**

A practical, production-ready web app that reads ship names (and other text) from vessel hull images — with preprocessing, confidence filtering, visualization, and clean post-processing.

Built for maritime spotting, port operations, logistics research, or just ship enthusiasts.

## ✨ Features

- Drag-and-drop or upload ship hull images (jpg/png/webp)
- Automatic image preprocessing (resize, CLAHE contrast, sharpening)
- PaddleOCR v2.8.1 with angle classification for rotated text
- Confidence-based filtering + smart text merging
- Clean uppercase ship name extraction (removes junk characters)
- Visual bounding boxes + per-line confidence scores
- Download annotated image
- Extraction history (session-based)
- Modular code structure (easy to maintain/extend)
- Beautiful modern UI with tabs and metrics

## Demo (local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ship-hull-ocr.git
cd ship-hull-ocr

# 2. Install dependencies (Python 3.9–3.12 recommended)
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
