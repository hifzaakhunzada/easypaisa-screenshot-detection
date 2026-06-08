# Fake Payment Screenshot Detector

A hybrid AI system combining image forensics, OCR, perceptual hashing, and a CNN to detect tampered or reused payment screenshots.

---

## Project Structure

```
fake_payment_detector/
├── app.py                  ← Streamlit UI (run this to use the system)
├── pipeline.py             ← Main orchestrator
├── train.py                ← Model training script
├── requirements.txt
│
├── modules/
│   ├── forensics.py        ← Metadata, compression, screenshot detection
│   ├── ml_model.py         ← MobileNetV2 CNN classifier
│   ├── ocr_module.py       ← Text + transaction ID extraction
│   ├── hashing.py          ← Perceptual hash duplicate detection
│   └── decision_engine.py  ← Combines all signals into final verdict
│
├── database/
│   └── db_manager.py       ← SQLite: stores hashes, txn IDs, history
│
├── utils/
│   └── image_utils.py      ← Shared image loading/preprocessing helpers
│
├── data/
│   ├── real/               ← Place GENUINE payment screenshots here
│   └── fake/               ← Place TAMPERED screenshots here
│
└── models/
    └── payment_detector.keras  ← Saved after training
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Tesseract OCR installed on your system

**Install Tesseract:**
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
- **macOS:** `brew install tesseract`
- **Linux (Ubuntu):** `sudo apt install tesseract-ocr`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialise the database

```bash
python -c "from database import db_manager; db_manager.init_db()"
```

---

## Getting Training Data

Since you don't have a dataset yet, here are your options:

### Option A — Collect manually (recommended for best results)
- **Real screenshots:** Take genuine payment confirmation screenshots from apps like Google Pay, PhonePe, Paytm, PayPal, etc.
- **Fake screenshots:** Use image editing tools (GIMP, Photoshop, or online editors) to tamper with real ones — change the amount, recipient name, or transaction ID.
- Aim for **at least 50–100 images per class** to start. 200+ per class is better.

### Option B — Use the Kaggle dataset
Search Kaggle for "fake payment screenshot" or "payment fraud detection" datasets.
Some relevant datasets:
- Search: https://www.kaggle.com/search?q=payment+screenshot+fake

### Option C — Data augmentation
Once you have a small seed set, use augmentation (rotation, brightness, cropping) to expand it. The training script already includes basic augmentation.

Place your images in:
```
data/real/   ← genuine screenshots
data/fake/   ← tampered/fake screenshots
```

---

## Training the Model

Once you have at least 20+ images per class:

```bash
python train.py
```

Options:
```bash
python train.py --epochs 30 --data ./data
```

The trained model is saved to `models/payment_detector.keras`.

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser. You can upload a screenshot and get an instant verdict.

---

## Using the Pipeline Directly (CLI)

```bash
python pipeline.py path/to/screenshot.png
```

Returns a full JSON report.

---

## How the System Works

| Module            | What it checks                                | Weight |
|-------------------|-----------------------------------------------|--------|
| ML Model (CNN)    | Overall image authenticity                    | 40%    |
| Image Forensics   | Metadata, compression, screenshot detection   | 30%    |
| Duplicate Hash    | Has this image been submitted before?         | 20%    |
| OCR               | Is there a transaction ID present?            | 10%    |

**Verdict thresholds** (combined fraud score 0.0–1.0):
- `< 0.25` → **Genuine**
- `0.25–0.55` → **Suspicious**
- `> 0.55` → **Fraudulent**

---

## Development Roadmap

- [x] Project structure
- [x] Image forensics module
- [x] OCR + transaction ID extraction
- [x] Perceptual hash duplicate detection
- [x] CNN classifier (MobileNetV2)
- [x] Decision engine
- [x] SQLite database
- [x] Streamlit UI
- [ ] Collect and label training dataset
- [ ] Train and evaluate the model
- [ ] Fine-tune decision weights
- [ ] Add support for more payment platforms
- [ ] Deploy as web app

---

## Tech Stack

| Component         | Technology                |
|-------------------|---------------------------|
| Language          | Python 3.10+              |
| ML Framework      | TensorFlow / Keras        |
| Image Processing  | OpenCV, Pillow            |
| OCR               | pytesseract (Tesseract)   |
| Hashing           | imagehash                 |
| Database          | SQLite (built-in)         |
| UI                | Streamlit                 |
