"""
modules/decision_engine.py
Combines outputs from forensics, ML, OCR, and hashing modules
to produce a final verification verdict.
"""


# ---------------------------------------------------------------------------
# Weights for each signal (must sum to 1.0)
# Tune these based on your testing and validation results.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "ml_score":        0.50,   # CNN prediction (most important)
    "forensics_score": 0.25,   # Metadata + compression + screenshot checks
    "duplicate":       0.15,   # Hash-based duplicate detection
    "no_txn_id":       0.10,   # Missing transaction ID (OCR often misses UIs)
}

THRESHOLDS = {
    "genuine":    0.22,   # Combined score below this → Genuine
    "suspicious": 0.42,   # Between genuine and this → Suspicious
    # Above suspicious threshold → Fraudulent
}


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def compute_combined_score(
    ml_result: dict,
    forensics_result: dict,
    ocr_result: dict,
    hash_result: dict,
) -> float:
    """
    Compute a weighted fraud score between 0.0 (genuine) and 1.0 (fraudulent).
    """
    # ML score: 0 = real, 1 = fake
    ml_score = ml_result.get("ml_score", 0.5) if ml_result else 0.5

    # Forensics score: 0 = clean, 1 = very suspicious
    forensics_score = forensics_result.get("forensics_score", 0.0)

    # Duplicate: 1.0 if a duplicate was found, else 0.0
    duplicate_score = 1.0 if hash_result.get("duplicate_found", False) else 0.0

    # Missing transaction ID is a weak signal of fakeness
    no_txn_score = 0.0 if ocr_result.get("has_transaction_id", True) else 1.0

    combined = (
        WEIGHTS["ml_score"]        * ml_score +
        WEIGHTS["forensics_score"] * forensics_score +
        WEIGHTS["duplicate"]       * duplicate_score +
        WEIGHTS["no_txn_id"]       * no_txn_score
    )
    return round(combined, 4)


def get_verdict(combined_score: float) -> str:
    """Map a combined score to a human-readable verdict."""
    if combined_score < THRESHOLDS["genuine"]:
        return "Genuine"
    elif combined_score < THRESHOLDS["suspicious"]:
        return "Suspicious"
    else:
        return "Fraudulent"


def make_decision(
    ml_result: dict,
    forensics_result: dict,
    ocr_result: dict,
    hash_result: dict,
) -> dict:
    """
    Produce the final verification report combining all module outputs.
    """
    combined_score = compute_combined_score(
        ml_result, forensics_result, ocr_result, hash_result
    )
    verdict = get_verdict(combined_score)
    confidence = 1.0 - combined_score if verdict == "Genuine" else combined_score

    # Collect all human-readable flags for the UI
    flags = []

    if ml_result and ml_result.get("label") == "Fake":
        flags.append(f"ML model predicts FAKE (confidence {ml_result['confidence']:.0%})")

    if forensics_result:
        meta_reasons = forensics_result.get("metadata_check", {}).get("reasons", [])
        flags.extend(meta_reasons)
        if forensics_result.get("compression_check", {}).get("suspicious"):
            flags.append("Suspicious re-compression artifacts detected")
        if not forensics_result.get("screenshot_check", {}).get("is_likely_screenshot"):
            flags.append("Image does not appear to be a genuine screenshot")

    if hash_result and hash_result.get("duplicate_found"):
        for match in hash_result.get("matches", []):
            if match.get("exact_duplicate"):
                flags.append(f"Exact duplicate of transaction {match['stored_transaction_id']}")
            else:
                flags.append(
                    f"Near-duplicate of transaction {match['stored_transaction_id']} "
                    f"(distance={match['hamming_distance']})"
                )

    if ocr_result and not ocr_result.get("has_transaction_id"):
        flags.append("No transaction ID could be extracted from the screenshot")

    return {
        "verdict": verdict,
        "combined_score": combined_score,
        "confidence": round(confidence, 4),
        "flags": flags,
        "details": {
            "ml": ml_result,
            "forensics": forensics_result,
            "ocr": ocr_result,
            "hashing": hash_result,
        },
    }
