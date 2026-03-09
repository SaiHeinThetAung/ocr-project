import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from core.config import CONFIG
from core.ocr import get_ocr_model
from core.preprocess import prepare_image
from core.postprocess import extract_best_name
from core.viz import draw_ocr_boxes

# ====================== SESSION STATE ======================
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(
    page_title="Ship Hull OCR Pro",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 Ship Hull Name Extractor Pro")
st.markdown("**PaddleOCR 2.8.1** • Your exact preprocessing & postprocessing • Modern UI")

tab1, tab2, tab3 = st.tabs(["📸 Extractor", "⚙️ Settings", "📜 History"])

# ====================== TAB 1: EXTRACTOR ======================
with tab1:
    col_upload, col_info = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag & drop ship hull image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supports drag & drop • Max size handled automatically"
        )

    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original is None:
            st.error("❌ Invalid image file")
        else:
            processed = prepare_image(original.copy())
            model = get_ocr_model()
            
            with st.spinner("🔍 Running PaddleOCR..."):
                ocr_result = model(processed)

            ship_name, confidence = extract_best_name(ocr_result)
            annotated = draw_ocr_boxes(processed, ocr_result)

            # ── Beautiful Results ──
            st.markdown("### 📸 Results")

            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with c2:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Text (green boxes)", use_container_width=True)

            # Big name card
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #00ff9d, #00b36b); 
                        padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                <h1 style="color: #000; margin: 0; font-size: 2.2rem;">{ship_name or 'NO NAME DETECTED'}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{confidence:.1%}")
            m2.metric("Words Detected", len([t for _, (t, s) in ocr_result if s >= CONFIG["postprocess"]["min_conf"]]))
            m3.metric("Threshold", f"{CONFIG['postprocess']['min_conf']}")

            # Action buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                _, buf = cv2.imencode(".png", annotated)
                st.download_button(
                    "📥 Download Annotated Image",
                    buf.tobytes(),
                    file_name=f"{ship_name or 'ship'}_annotated.png",
                    mime="image/png",
                    use_container_width=True
                )
            with btn_col2:
                if st.button("📋 Copy Ship Name", use_container_width=True, type="primary"):
                    st.toast(f"✅ Copied: **{ship_name}**", icon="📋")
                    st.session_state.last_copied = ship_name

            # Save to history
            if ship_name:
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "name": ship_name,
                    "confidence": confidence,
                    "filename": uploaded_file.name
                })

            # Raw output
            with st.expander("🔧 Raw OCR Output"):
                for i, (box, (txt, score)) in enumerate(ocr_result):
                    st.write(f"**{i+1}.** `{txt}` — `{score:.3f}`")

# ====================== TAB 2: SETTINGS ======================
with tab2:
    st.json(CONFIG, expanded=False)
    st.info("Edit `config.yaml` and restart the app to apply changes")

# ====================== TAB 3: HISTORY ======================
with tab3:
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            with st.container(border=True):
                st.markdown(f"**{entry['timestamp']}** — `{entry['filename']}`")
                st.markdown(f"### {entry['name']}")
                st.progress(entry['confidence'])
                st.caption(f"Confidence: {entry['confidence']:.1%}")
    else:
        st.info("No extractions yet. Upload your first image!")

# ====================== FOOTER ======================
st.caption("Built for you by Senior AI Developer • Exact same logic as your original script • Just way prettier 😎")