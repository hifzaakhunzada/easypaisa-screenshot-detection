"""
pipeline.py
Main orchestrator — runs all modules in sequence and returns a final result.
Import this in app.py (Streamlit) or use directly in scripts.
"""

import os
from modules import forensics, ocr_module, hashing, decision_engine
from database import db_manager

# Only import ml_model if a trained model exists
from modules import ml_model


def verify_screenshot(image_path: str) -> dict:
    """
    Run the full verification pipeline on a payment screenshot.

    Steps:
    1. Image forensics
    2. OCR + transaction ID extraction
    3. Duplicate hash check
    4. ML model prediction (if model is trained)
    5. Decision engine combines everything

    Returns a complete result dict.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    # --- 1. Forensics ---
    forensics_result = forensics.run_forensics(image_path)

    # --- 2. OCR ---
    ocr_result = ocr_module.run_ocr(image_path)

    # --- 3. Hash duplicate check ---
    stored_hashes = db_manager.get_all_hashes()
    hash_result = hashing.check_against_stored(image_path, stored_hashes)

    # --- 4. ML prediction ---
    ml_result = None
    if ml_model.model_is_trained():
        try:
            ml_result = ml_model.predict(image_path)
        except Exception as e:
            ml_result = {"error": str(e), "ml_score": 0.5, "label": "Unknown", "confidence": 0.0}
    else:
        # If no model is trained yet, use a neutral score
        ml_result = {
            "label": "Unknown (model not trained)",
            "ml_score": 0.5,
            "confidence": 0.0,
            "note": "Train the model first using train.py"
        }

    # --- 5. Decision ---
    decision = decision_engine.make_decision(
        ml_result=ml_result,
        forensics_result=forensics_result,
        ocr_result=ocr_result,
        hash_result=hash_result,
    )

    # --- 6. Save to database ---
    new_hash = hash_result.get("new_hash")
    for txn_id in ocr_result.get("transaction_ids", []):
        if not db_manager.transaction_exists(txn_id):
            db_manager.save_transaction(
                txn_id=txn_id,
                image_hash=new_hash,
                result=decision["verdict"]
            )

    if new_hash:
        db_manager.save_hash(
            phash=new_hash,
            txn_id=ocr_result.get("transaction_ids", [None])[0]
        )

    db_manager.log_verification(
        image_path=image_path,
        final_result=decision["verdict"],
        confidence=decision["confidence"],
        forensics_score=forensics_result.get("forensics_score", 0),
        ml_score=ml_result.get("ml_score", 0.5) if ml_result else 0.5,
        is_duplicate=hash_result.get("duplicate_found", False),
    )

    return decision


# Quick CLI test
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_image>")
        sys.exit(1)

    db_manager.init_db()
    result = verify_screenshot(sys.argv[1])
    print(json.dumps(result, indent=2))
