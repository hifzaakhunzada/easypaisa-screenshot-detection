"""
augment_data.py
Augments BOTH real and fake classes to expand the dataset.
Run this once before training.

Usage:
    python augment_data.py
    python augment_data.py --copies 5 --data ./data
"""

import argparse
import os
import random
from PIL import Image, ImageEnhance, ImageFilter

def augment_image(img: Image.Image) -> Image.Image:
    """Apply random augmentations suitable for screenshot images."""
    aug = img

    # Slight rotation (screenshots rarely get rotated much)
    if random.random() > 0.5:
        aug = aug.rotate(random.uniform(-3, 3), expand=False)

    # Brightness variation (different screen brightness levels)
    if random.random() > 0.4:
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.80, 1.20))

    # Contrast variation
    if random.random() > 0.4:
        aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.85, 1.15))

    # Slight blur (simulates camera photos of screens)
    if random.random() > 0.5:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.8)))

    # Sharpness variation
    if random.random() > 0.5:
        aug = ImageEnhance.Sharpness(aug).enhance(random.uniform(0.8, 1.5))

    return aug


def augment_folder(folder: str, copies: int):
    """Generate `copies` augmented versions of every original image in folder."""
    exts = (".png", ".jpg", ".jpeg")

    # Only augment originals, not previously augmented files
    originals = [
        f for f in os.listdir(folder)
        if f.lower().endswith(exts) and not f.startswith("aug_")
    ]

    if not originals:
        print(f"  No images found in {folder}")
        return 0

    created = 0
    for fname in originals:
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        for i in range(copies):
            aug = augment_image(img)
            out_name = f"aug_{i}_{fname}"
            aug.save(os.path.join(folder, out_name))
            created += 1

    return created


def main():
    parser = argparse.ArgumentParser(description="Augment training dataset")
    parser.add_argument("--data", default="./data", help="Path to data directory")
    parser.add_argument("--copies", type=int, default=4,
                        help="Number of augmented copies per original image (default: 4)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove previously generated augmented images instead of creating new ones")
    args = parser.parse_args()

    real_dir = os.path.join(args.data, "real")
    fake_dir = os.path.join(args.data, "fake")

    for d in [real_dir, fake_dir]:
        if not os.path.exists(d):
            print(f"ERROR: Directory not found: {d}")
            return

    # Clean mode — remove augmented files
    if args.clean:
        for folder, name in [(real_dir, "real"), (fake_dir, "fake")]:
            removed = 0
            for f in os.listdir(folder):
                if f.startswith("aug_"):
                    os.remove(os.path.join(folder, f))
                    removed += 1
            print(f"  Removed {removed} augmented images from {name}/")
        return

    # Count originals
    exts = (".png", ".jpg", ".jpeg")
    real_orig = len([f for f in os.listdir(real_dir) if f.lower().endswith(exts) and not f.startswith("aug_")])
    fake_orig = len([f for f in os.listdir(fake_dir) if f.lower().endswith(exts) and not f.startswith("aug_")])

    print(f"Original images — real: {real_orig}, fake: {fake_orig}")
    print(f"Creating {args.copies} augmented copies per image...\n")

    real_created = augment_folder(real_dir, args.copies)
    fake_created = augment_folder(fake_dir, args.copies)

    real_total = real_orig + real_created
    fake_total = fake_orig + fake_created

    print(f"Done!")
    print(f"  real/ : {real_orig} originals + {real_created} augmented = {real_total} total")
    print(f"  fake/ : {fake_orig} originals + {fake_created} augmented = {fake_total} total")
    print(f"\nNow run:  python train.py")


if __name__ == "__main__":
    main()