"""
modules/forensics.py
Image forensics module — analyses metadata, compression artifacts,
layout consistency, and screenshot vs photo detection.
"""

import io
import cv2
import numpy as np
from PIL import Image
import exifread


# ---------------------------------------------------------------------------
# Metadata analysis
# ---------------------------------------------------------------------------

def extract_metadata(image_path: str) -> dict:
    """
    Extract EXIF metadata from an image file.
    Returns a dict of tag -> value pairs (empty dict if none found).
    """
    metadata = {}
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="UNDEF", details=False)
        for tag, value in tags.items():
            metadata[str(tag)] = str(value)
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def check_metadata_suspicious(metadata: dict) -> dict:
    """
    Flag potentially suspicious metadata patterns.
    Returns a result dict with 'suspicious' bool and 'reasons' list.
    """
    reasons = []

    # Real screenshots from phones/apps typically have no EXIF or minimal EXIF.
    # If rich camera EXIF data is present on a supposed screenshot, that's odd.
    camera_tags = [k for k in metadata if "Image Make" in k or "Image Model" in k]
    if camera_tags:
        reasons.append("Camera EXIF tags found — image may be a photo, not a screenshot.")

    if "Image Software" in metadata:
        software = metadata["Image Software"].lower()
        for editor in ["photoshop", "gimp", "lightroom", "affinity", "paint"]:
            if editor in software:
                reasons.append(f"Editing software detected in metadata: {metadata['Image Software']}")

    return {"suspicious": len(reasons) > 0, "reasons": reasons}


# ---------------------------------------------------------------------------
# Compression artifact detection
# ---------------------------------------------------------------------------

def detect_compression_artifacts(image_path: str) -> dict:
    """
    Estimate JPEG compression quality and look for re-compression artifacts.
    Very low quality scores can indicate the image has been re-saved after editing.
    """
    img = Image.open(image_path)

    # Re-encode to JPEG in memory and measure quality degradation
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    reencoded = Image.open(buffer)

    # Mean absolute difference between original and re-encoded
    orig_arr = np.array(img.convert("RGB"), dtype=np.float32)
    reenc_arr = np.array(reencoded, dtype=np.float32)
    diff = np.mean(np.abs(orig_arr - reenc_arr))

    suspicious = diff > 14.0  # JPEG re-save noise; lower values false-positive on clean shots

    return {
        "mean_recompression_diff": round(float(diff), 4),
        "suspicious": suspicious,
        "reason": "High re-compression difference — possible re-save after editing." if suspicious else None,
    }


# ---------------------------------------------------------------------------
# Screenshot vs photo detection (basic heuristic)
# ---------------------------------------------------------------------------

def is_likely_screenshot(image_path: str) -> dict:
    """
    Screenshots tend to have:
    - Mostly uniform regions (UI backgrounds)
    - Relatively low edge density compared to natural photos
    - PNG format or very high JPEG quality
    Returns a confidence score and decision.
    """
    img_pil = Image.open(image_path)
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return {
            "format": img_pil.format,
            "edge_density": None,
            "screenshot_confidence": 0.5,
            "is_likely_screenshot": True,
        }

    # Format check
    fmt = img_pil.format  # PNG is typical for screenshots
    format_score = 1.0 if fmt == "PNG" else 0.65

    # Edge density via Canny
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Low edge density → more likely a screenshot (UI is flat)
    edge_score = 1.0 if edge_density < 0.05 else (0.5 if edge_density < 0.15 else 0.1)

    confidence = round((format_score * 0.4) + (edge_score * 0.6), 3)

    return {
        "format": fmt,
        "edge_density": round(float(edge_density), 4),
        "screenshot_confidence": confidence,
        "is_likely_screenshot": confidence >= 0.5,
    }


# ---------------------------------------------------------------------------
# Combined forensics report
# ---------------------------------------------------------------------------

def run_forensics(image_path: str) -> dict:
    """
    Run all forensic checks and return a combined report.
    """
    metadata = extract_metadata(image_path)
    meta_check = check_metadata_suspicious(metadata)
    compression = detect_compression_artifacts(image_path)
    screenshot_check = is_likely_screenshot(image_path)

    suspicious_flags = sum([
        meta_check["suspicious"],
        compression["suspicious"],
        not screenshot_check["is_likely_screenshot"],
    ])

    return {
        "metadata": metadata,
        "metadata_check": meta_check,
        "compression_check": compression,
        "screenshot_check": screenshot_check,
        "suspicious_flag_count": suspicious_flags,
        "forensics_score": round(suspicious_flags / 3, 3),  # 0 = clean, 1 = very suspicious
    }
