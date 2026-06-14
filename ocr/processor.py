"""
OCR processor — extracts text from images using pytesseract + Pillow.

Install system dependency:  brew install tesseract
Install Python packages:    pip install pytesseract Pillow

OCR is entirely optional.  If either the system binary or the Python packages
are missing, extract_text() silently returns None so the pipeline continues
using caption-only text.
"""
from __future__ import annotations
from typing import Optional

try:
    import pytesseract
    from PIL import Image
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def is_available() -> bool:
    """Return True if tesseract is installed and importable."""
    if not _AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text(image_path: Optional[str]) -> Optional[str]:
    """
    Run OCR on *image_path* and return the extracted string, or None if OCR is
    unavailable / the file cannot be opened.
    """
    if not image_path or not _AVAILABLE:
        return None
    try:
        img = Image.open(image_path)
        # Convert to RGB so tesseract handles PNGs with alpha channels
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6").strip()
        return text or None
    except Exception as e:
        print(f"[ocr] failed on {image_path}: {e}")
        return None
