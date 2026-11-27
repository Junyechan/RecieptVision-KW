
"""ReceiptVision V5.2 preprocessing example."""

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
        default="dynamic",
        help="Preprocessing method",
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
    print("Method:", result["method"])
    print("Actual method:", result["actual_method"])
    print("Block size:", result["block_size"])
    print("OCR image shape:", result["ocr_image"].shape)
    print("Saved:", out)


if __name__ == "__main__":
    main()
