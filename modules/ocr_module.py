"""
modules/ocr_module.py
OCR module — extracts text from payment screenshots and pulls out
transaction IDs for duplicate detection.
"""

import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """
    Enhance image for better OCR accuracy:
    - Convert to grayscale
    - Apply adaptive thresholding
    - Slight denoising
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(image_path: str) -> str:
    """
    Extract all text from a payment screenshot using pytesseract.
    Returns the raw extracted string.
    """
    processed = preprocess_for_ocr(image_path)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img, config="--psm 6")
    return text.strip()


# ---------------------------------------------------------------------------
# Transaction ID extraction
# ---------------------------------------------------------------------------

# Common patterns for payment transaction IDs across platforms.
# Extend this list as you add support for more platforms.
TRANSACTION_PATTERNS = [
    r"\b[A-Z0-9]{10,20}\b",          # Generic alphanumeric ID (10–20 chars)
    r"UPI[:/\s]+([A-Z0-9]{8,16})",   # UPI reference number
    r"UTR[:/\s]+([0-9]{12})",        # UTR number (12 digits)
    r"Ref[.:\s]+([A-Z0-9]{8,18})",   # Generic "Ref" field
    r"TXN[:/\s]+([A-Z0-9]{8,18})",   # TXN prefix
    r"Order[:/\s]+([A-Z0-9#\-]{6,18})",  # Order number
]


def extract_transaction_ids(text: str) -> list[str]:
    """
    Extract potential transaction / reference IDs from OCR text.
    Returns a deduplicated list of candidate IDs.
    """
    ids = set()
    for pattern in TRANSACTION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            # Clean up and filter very short/common false positives
            m = m.strip().upper()
            if len(m) >= 8:
                ids.add(m)
    return list(ids)


# ---------------------------------------------------------------------------
# Amount extraction (bonus check)
# ---------------------------------------------------------------------------

def extract_amount(text: str) -> str | None:
    """
    Try to extract the payment amount from OCR text.
    Returns the first match or None.
    """
    patterns = [
        r"₹\s?[\d,]+(?:\.\d{1,2})?",
        r"Rs\.?\s?[\d,]+(?:\.\d{1,2})?",
        r"\$\s?[\d,]+(?:\.\d{1,2})?",
        r"INR\s?[\d,]+(?:\.\d{1,2})?",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# Combined OCR report
# ---------------------------------------------------------------------------

def run_ocr(image_path: str) -> dict:
    """
    Run OCR pipeline and return extracted fields.
    """
    raw_text = extract_text(image_path)
    transaction_ids = extract_transaction_ids(raw_text)
    amount = extract_amount(raw_text)

    return {
        "raw_text": raw_text,
        "transaction_ids": transaction_ids,
        "amount": amount,
        "text_length": len(raw_text),
        "has_transaction_id": len(transaction_ids) > 0,
    }
