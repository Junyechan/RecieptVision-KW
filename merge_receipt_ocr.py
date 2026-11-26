"""Merge full-receipt OCR with multi-variant item-table OCR."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from information_extractor import extract_receipt_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", default="outputs/clear_korean_receipt_final.json")
    parser.add_argument("--items", default="outputs/items_ocr/items_ocr_comparison.json")
    parser.add_argument("--output", default="outputs/clear_korean_receipt_merged.json")
    args = parser.parse_args()

    full_path, items_path = Path(args.full), Path(args.items)
    full = json.loads(full_path.read_text(encoding="utf-8"))
    item_ocr = json.loads(items_path.read_text(encoding="utf-8"))
    if not full.get("success"):
        raise SystemExit(f"Full receipt OCR failed: {full.get('error')}")
    ocr = full.get("ocr") or {}
    ocr["item_table_ocr"] = item_ocr
    receipt = extract_receipt_info(ocr)
    full["receipt"] = receipt
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "merchant": receipt["merchant"],
        "date": receipt["date"], "items": receipt["items"],
        "total": receipt["total"], "validation": receipt["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
