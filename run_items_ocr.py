from pathlib import Path
import json
import cv2

from ocr import recognize_text

IMAGE_PATH = Path("images/clear_korean_receipt_items_crop.png")
OUTPUT_DIR = Path("outputs/items_ocr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

variants = {
    "original_upscaled": image,
    "grayscale_upscaled": gray,
    "clahe_upscaled": clahe,
}

results = {}
for name, variant in variants.items():
    print(f"\n===== {name} =====")
    result = recognize_text(variant, languages=("ko", "en"), detail=True)
    results[name] = result
    print(result["text"])
    print("Detected lines:", len(result["lines"]))

output_path = OUTPUT_DIR / "items_ocr_comparison.json"
output_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"\nSaved OCR comparison to: {output_path}")
