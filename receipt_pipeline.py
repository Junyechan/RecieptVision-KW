"""End-to-end helper connecting receipt preprocessing, OCR and field parsing."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from preprocessing import preprocess_receipt
from ocr import recognize_text
from information_extractor import extract_receipt_info


def process_receipt(
    image_path: str | Path,
    method: str = "clahe_gaussian_b21_c10",
    languages: tuple[str, ...] = ("ko", "en"),
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Preprocess a receipt image, recognize text and extract key fields."""
    prep = preprocess_receipt(image_path, method=method)
    if not prep["success"]:
        return {"success": False, "stage": "preprocessing", "error": prep["error"]}

    ocr_result = recognize_text(prep["ocr_image"], languages=languages)
    info = extract_receipt_info(ocr_result)
    result = {
        "success": True,
        "preprocessing": {
            "method": prep["method"],
            "detection_score": prep["detection_score"],
            "detection_source": prep["detection_source"],
        },
        "ocr": ocr_result,
        "receipt": info,
    }

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR and extract fields from a receipt image")
    parser.add_argument("image", help="Path to a receipt image")
    parser.add_argument("--output", default="outputs/receipt_result.json", help="JSON output path")
    parser.add_argument("--method", default="clahe_gaussian_b21_c10", help="Preprocessing variant")
    args = parser.parse_args()

    result = process_receipt(args.image, method=args.method, output_json=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
