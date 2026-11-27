
"""ReceiptVision V5.4 preprocessing example."""

from pathlib import Path
import argparse

from preprocessing import preprocess_receipt, save_image


def main():

    parser = argparse.ArgumentParser(
        description="Receipt preprocessing example"
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="images/receipt_01.jpg",
        help="Input receipt image",
    )

    parser.add_argument(
        "--method",
        default="hybrid",
        help="Preprocessing method (default: hybrid)",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    result = preprocess_receipt(
        image_path,
        method=args.method,
    )

    if not result["success"]:
        raise RuntimeError(
            f"영수증 전처리 실패: {result['error']}"
        )

    out = Path("outputs/ocr_handoff")
    out.mkdir(parents=True, exist_ok=True)

    save_image(
        out / "warped.png",
        result["warped"],
    )

    save_image(
        out / "ocr_ready.png",
        result["ocr_image"],
    )

    save_image(
        out / "detection_overlay.png",
        result["detection_overlay"],
    )

    print("Corners:", result["corners"])
    print("OCR corners:", result["ocr_corners"])
    print("Version:", result["preprocessing_version"])
    print("Selected method:", result["selected_method"])
    print("Hybrid selection:", result["hybrid_selection"])
    extension = result["boundary_extension"]
    print("Top extended:", extension["top_extended"])
    print("Bottom extended:", extension["bottom_extended"])
    print("Area ratio:", extension["area_ratio"])
    print("Method:", result["method"])
    print("Actual method:", result["actual_method"])
    print("Block size:", result["block_size"])
    print("OCR image shape:", result["ocr_image"].shape)
    print("Saved:", out)


if __name__ == "__main__":
    main()
