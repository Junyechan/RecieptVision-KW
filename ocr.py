"""OCR utilities for ReceiptVision-KW.

Uses EasyOCR with Korean and English support. The reader is created lazily so
importing this module does not download models or initialize PyTorch.
"""
from __future__ import annotations

from typing import Any, Iterable
import numpy as np


class ReceiptOCRError(RuntimeError):
    """Raised when OCR cannot be run or the OCR engine is unavailable."""


def recognize_text(
    image: np.ndarray,
    languages: Iterable[str] = ("ko", "en"),
    detail: bool = True,
    reader: Any = None,
) -> dict[str, Any]:
    """Recognize text in an image array.

    Args:
        image: OpenCV image as a NumPy array (grayscale or BGR).
        languages: EasyOCR language codes. Korean and English are the defaults.
        detail: If True, return recognized lines with confidence and boxes.
        reader: Optional existing EasyOCR Reader, useful for reuse/testing.

    Returns:
        A dictionary containing ``text``, ``lines``, and ``engine``.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or color")

    if reader is None:
        try:
            import easyocr
        except ImportError as exc:
            raise ReceiptOCRError(
                "EasyOCR is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        reader = easyocr.Reader(list(languages), gpu=False)

    try:
        results = reader.readtext(image, detail=1, paragraph=False)
    except Exception as exc:
        raise ReceiptOCRError(f"OCR failed: {exc}") from exc

    lines = []
    for result in results:
        if len(result) < 3:
            continue
        box, text, confidence = result[0], str(result[1]).strip(), float(result[2])
        if text:
            lines.append({
                "text": text,
                "confidence": confidence,
                "box": [[float(x), float(y)] for x, y in box],
            })

    return {
        "text": "\n".join(item["text"] for item in lines),
        "lines": lines if detail else [item["text"] for item in lines],
        "engine": "EasyOCR",
        "languages": list(languages),
    }
