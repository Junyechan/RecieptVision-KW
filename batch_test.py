
from pathlib import Path
import csv
import numpy as np

from preprocessing import (
    BASE_DIR,
    preprocess_receipt,
    save_results
)

# 경로 설정
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs" / "batch_test"

# 지원 이미지 형식
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 테스트할 이미지 검색
image_files = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in EXTENSIONS
)

if not image_files:
    raise FileNotFoundError(
        "images 폴더에 테스트 이미지가 없습니다."
    )

records = []

print(f"테스트 이미지 수: {len(image_files)}")

for index, image_path in enumerate(image_files, start=1):
    print(
        f"\n[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    try:
        # 전처리 실행
        results = preprocess_receipt(image_path)

        # 이미지별 결과 폴더
        result_dir = OUTPUT_DIR / image_path.stem

        # 결과 저장
        save_results(results, result_dir)

        # 영수증 검출 여부
        detected = results["receipt_detected"]

        # 이미지 크기
        original_shape = results["original"].shape
        warped_shape = results["warped"].shape

        # OCR용 이미지 유효성 검사
        ocr_image = results["warped_otsu"]

        valid_ocr_image = (
            isinstance(ocr_image, np.ndarray)
            and ocr_image.ndim == 2
            and ocr_image.dtype == np.uint8
            and ocr_image.size > 0
        )

        records.append({
            "filename": image_path.name,
            "status": "success",
            "receipt_detected": detected,
            "original_height": original_shape[0],
            "original_width": original_shape[1],
            "warped_height": warped_shape[0],
            "warped_width": warped_shape[1],
            "valid_ocr_image": valid_ocr_image,
            "error": ""
        })

        print("영수증 검출:", detected)
        print("OCR 이미지 유효:", valid_ocr_image)

    except Exception as e:
        records.append({
            "filename": image_path.name,
            "status": "error",
            "receipt_detected": False,
            "original_height": "",
            "original_width": "",
            "warped_height": "",
            "warped_width": "",
            "valid_ocr_image": False,
            "error": str(e)
        })

        print("오류:", e)

# CSV 결과 저장
csv_path = OUTPUT_DIR / "test_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "status",
            "receipt_detected",
            "original_height",
            "original_width",
            "warped_height",
            "warped_width",
            "valid_ocr_image",
            "error"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

# 결과 통계
total = len(records)

detected_count = sum(
    record["receipt_detected"]
    for record in records
)

valid_count = sum(
    record["valid_ocr_image"]
    for record in records
)

error_count = sum(
    record["status"] == "error"
    for record in records
)

print("\n========== 테스트 결과 ==========")
print("전체 이미지:", total)
print("영수증 후보 검출:", detected_count)
print("OCR 이미지 형식 정상:", valid_count)
print("처리 오류:", error_count)

print(
    "후보 검출 비율:",
    f"{detected_count / total * 100:.1f}%"
)

print("CSV 저장:", csv_path)
