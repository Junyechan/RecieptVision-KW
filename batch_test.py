
"""Batch preprocessing test for ReceiptVision V5.2."""

from pathlib import Path
import argparse
import csv
import numpy as np

from preprocessing import (
    preprocess_receipt,
    read_image,
    save_image,
)


BASE_DIR = Path(__file__).resolve().parent

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

FIELDNAMES = [
    "filename",
    "status",
    "receipt_detected",
    "original_height",
    "original_width",
    "warped_height",
    "warped_width",
    "method",
    "actual_method",
    "block_size",
    "detection_score",
    "detection_source",
    "valid_ocr_image",
    "error",
]


def batch_test(image_dir, output_dir, method="dynamic"):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"이미지 폴더가 없습니다: {image_dir}"
        )

    image_files = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )

    if not image_files:
        raise FileNotFoundError(
            "테스트 이미지가 없습니다."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print(f"테스트 이미지 수: {len(image_files)}")

    for index, image_path in enumerate(image_files, start=1):

        print(
            f"\n[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        record = {
            key: "" for key in FIELDNAMES
        }

        record.update({
            "filename": image_path.name,
            "status": "error",
            "receipt_detected": False,
            "valid_ocr_image": False,
            "method": method,
        })

        try:
            original = read_image(image_path)

            record["original_height"] = original.shape[0]
            record["original_width"] = original.shape[1]

            result = preprocess_receipt(
                original,
                method=method,
            )

            if not result["success"]:
                raise RuntimeError(result["error"])

            warped = result["warped"]
            ocr_image = result["ocr_image"]

            record.update({
                "receipt_detected": True,
                "warped_height": warped.shape[0],
                "warped_width": warped.shape[1],
                "method": result["method"],
                "actual_method": result["actual_method"],
                "block_size": result["block_size"],
                "detection_score": result["detection_score"],
                "detection_source": result["detection_source"],
            })

            valid_ocr_image = (
                isinstance(ocr_image, np.ndarray)
                and ocr_image.ndim == 2
                and ocr_image.dtype == np.uint8
                and ocr_image.size > 0
            )

            record["valid_ocr_image"] = valid_ocr_image

            if not valid_ocr_image:
                raise RuntimeError(
                    "유효하지 않은 OCR 입력 이미지"
                )

            result_dir = output_dir / image_path.stem
            result_dir.mkdir(parents=True, exist_ok=True)

            save_image(
                result_dir / "warped.png",
                warped,
            )

            save_image(
                result_dir / "ocr_ready.png",
                ocr_image,
            )

            save_image(
                result_dir / "detection_overlay.png",
                result["detection_overlay"],
            )

            record["status"] = "success"

            print("영수증 검출: True")
            print("OCR 이미지 유효:", valid_ocr_image)
            print("전처리 방식:", result["actual_method"])
            print("Block Size:", result["block_size"])
            print("Result: PASS")

        except Exception as exc:
            record["error"] = str(exc)

            print("Result: FAIL")
            print("오류:", exc)

        records.append(record)

    csv_path = output_dir / "test_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(records)

    total = len(records)

    detected_count = sum(
        r["receipt_detected"]
        for r in records
    )

    valid_count = sum(
        r["valid_ocr_image"]
        for r in records
    )

    success_count = sum(
        r["status"] == "success"
        for r in records
    )

    print("\n========== 테스트 결과 ==========")
    print("전체 이미지:", total)
    print("영수증 후보 검출:", detected_count)
    print("OCR 이미지 형식 정상:", valid_count)
    print("처리 성공:", success_count)
    print("처리 오류:", total - success_count)

    print(
        "처리 성공률:",
        f"{success_count / total * 100:.1f}%"
    )

    print("CSV 저장:", csv_path)

    return records


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Batch receipt preprocessing test"
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=BASE_DIR / "images",
        help="Input image directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "outputs" / "batch_test",
        help="Output directory",
    )

    parser.add_argument(
        "--method",
        default="dynamic",
        help="Preprocessing method",
    )

    args = parser.parse_args()

    records = batch_test(
        args.images,
        args.output,
        args.method,
    )

    if any(r["status"] != "success" for r in records):
        raise SystemExit(1)
