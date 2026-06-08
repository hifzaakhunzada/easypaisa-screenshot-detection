"""
modules/hashing.py
Perceptual hashing module — detects duplicate or slightly modified screenshots.
"""

import imagehash
from PIL import Image


HASH_SIMILARITY_THRESHOLD = 10  # Hamming distance; tune based on testing


def generate_phash(image_path: str) -> str:
    """
    Generate a perceptual hash (pHash) for an image.
    Returns the hash as a hex string.
    """
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(img))


def generate_dhash(image_path: str) -> str:
    """
    Generate a difference hash (dHash) — faster, good for near-duplicates.
    Returns the hash as a hex string.
    """
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.dhash(img))


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute the Hamming distance between two perceptual hash strings.
    Lower = more similar. 0 = identical.
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def is_duplicate(hash1: str, hash2: str, threshold: int = HASH_SIMILARITY_THRESHOLD) -> bool:
    """
    Return True if two hashes are close enough to be considered duplicates.
    """
    return hamming_distance(hash1, hash2) <= threshold


def check_against_stored(image_path: str, stored_hashes: list[dict]) -> dict:
    """
    Compare a new image's hash against a list of stored hashes.

    stored_hashes: list of dicts with keys 'hash' and 'transaction_id'
    Returns match info if a duplicate is found, or a clean result.
    """
    new_hash = generate_phash(image_path)
    matches = []

    for entry in stored_hashes:
        dist = hamming_distance(new_hash, entry["hash"])
        if dist <= HASH_SIMILARITY_THRESHOLD:
            matches.append({
                "stored_transaction_id": entry.get("transaction_id", "unknown"),
                "hamming_distance": dist,
                "exact_duplicate": dist == 0,
            })

    return {
        "new_hash": new_hash,
        "duplicate_found": len(matches) > 0,
        "matches": matches,
    }
