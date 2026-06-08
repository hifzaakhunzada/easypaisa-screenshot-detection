"""
train.py
Standalone training script.
Run this once you have collected enough images in data/real/ and data/fake/.

Usage:
    python train.py
    python train.py --epochs 30 --data ./data
"""

import argparse
import os
from database import db_manager
from modules.ml_model import train_model


def main():
    parser = argparse.ArgumentParser(description="Train the fake payment detector CNN")
    parser.add_argument("--data", default="./data", help="Path to data directory (must contain real/ and fake/)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    # Validate data directory
    real_dir = os.path.join(args.data, "real")
    fake_dir = os.path.join(args.data, "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"ERROR: Expected directories: {real_dir} and {fake_dir}")
        print("Please add training images before running this script.")
        return

    real_count = len([f for f in os.listdir(real_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    fake_count = len([f for f in os.listdir(fake_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    print(f"Found {real_count} real images and {fake_count} fake images.")

    if real_count < 10 or fake_count < 10:
        print("WARNING: Very few training images. Aim for at least 50 per class for meaningful results.")
        if real_count == 0 or fake_count == 0:
            print("Cannot train with 0 images in a class. Exiting.")
            return

    db_manager.init_db()
    print(f"Training for {args.epochs} epochs...")

    model, history = train_model(data_dir=args.data, epochs=args.epochs)

    # Print final metrics
    final_val_acc = history.history.get("val_accuracy", [0])[-1]
    final_val_loss = history.history.get("val_loss", [0])[-1]
    print(f"\nTraining complete.")
    print(f"Validation accuracy: {final_val_acc:.2%}")
    print(f"Validation loss:     {final_val_loss:.4f}")
    print("Model saved. You can now run app.py.")


if __name__ == "__main__":
    main()
