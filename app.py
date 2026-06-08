"""
app.py
Streamlit user interface for the Fake Payment Screenshot Detector.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from database import db_manager
from pipeline import verify_screenshot
from modules.ml_model import model_is_trained

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Payment Screenshot Detector",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Init DB on first run
# ---------------------------------------------------------------------------

db_manager.init_db()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🔍 Fake Payment Screenshot Detector")
st.caption("Upload a payment screenshot to verify whether it is genuine, suspicious, or fraudulent.")

if not model_is_trained():
    st.warning(
        "ML model not trained yet. The system will run forensics and hashing checks only. "
        "Run `python train.py` once you have collected training images.",
        icon="⚠️",
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload a payment screenshot",
    type=["png", "jpg", "jpeg"],
    help="Supports PNG and JPEG screenshots from any payment platform."
)

if uploaded_file:
    # Show image preview
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption="Uploaded screenshot", use_column_width=True)

    with col2:
        st.markdown("### Analysis")
        with st.spinner("Analysing screenshot..."):
            # Save to temp file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            result = verify_screenshot(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            verdict = result["verdict"]
            confidence = result["confidence"]

            # Verdict badge
            verdict_colours = {
                "Genuine": "green",
                "Suspicious": "orange",
                "Fraudulent": "red",
            }
            colour = verdict_colours.get(verdict, "grey")
            st.markdown(
                f"<div style='text-align:center; padding:12px; border-radius:8px; "
                f"background-color:{'#d4edda' if verdict=='Genuine' else '#fff3cd' if verdict=='Suspicious' else '#f8d7da'};"
                f"border: 2px solid {'green' if verdict=='Genuine' else 'orange' if verdict=='Suspicious' else 'red'};'>"
                f"<h2 style='margin:0; color:{colour};'>{verdict}</h2>"
                f"<p style='margin:4px 0 0 0; color:grey;'>Confidence: {confidence:.0%}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Flags
    flags = result.get("flags", [])
    if flags:
        st.markdown("### ⚠️ Issues Detected")
        for flag in flags:
            st.markdown(f"- {flag}")
    else:
        st.success("No issues detected.")

    # Expandable details
    with st.expander("📋 OCR Results"):
        ocr = result["details"].get("ocr", {})
        st.text_area("Extracted Text", ocr.get("raw_text", "N/A"), height=120)
        st.write("**Transaction IDs found:**", ocr.get("transaction_ids", []))
        st.write("**Amount:**", ocr.get("amount", "Not found"))

    with st.expander("🔬 Forensics Details"):
        forensics = result["details"].get("forensics", {})
        st.write("**Metadata flags:**", forensics.get("metadata_check", {}).get("reasons", []))
        comp = forensics.get("compression_check", {})
        st.write("**Re-compression diff:**", comp.get("mean_recompression_diff", "N/A"))
        ss = forensics.get("screenshot_check", {})
        st.write("**Screenshot confidence:**", ss.get("screenshot_confidence", "N/A"))

    with st.expander("🤖 ML Model"):
        ml = result["details"].get("ml", {})
        if ml:
            st.write("**Prediction:**", ml.get("label", "N/A"))
            st.write("**Raw score:**", ml.get("raw_score", "N/A"))
            st.write("**Confidence:**", ml.get("confidence", "N/A"))
        else:
            st.write("Model not run.")

    with st.expander("🔁 Duplicate Check"):
        hashing = result["details"].get("hashing", {})
        st.write("**Duplicate found:**", hashing.get("duplicate_found", False))
        st.write("**Matches:**", hashing.get("matches", []))
        st.write("**Image hash:**", hashing.get("new_hash", "N/A"))

st.divider()

# Recent verification history
with st.expander("📜 Recent Verifications"):
    history = db_manager.get_recent_verifications(limit=10)
    if history:
        import pandas as pd
        df = pd.DataFrame(history)[["checked_at", "final_result", "confidence", "is_duplicate"]]
        df.columns = ["Time", "Result", "Confidence", "Duplicate"]
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No verifications yet.")
