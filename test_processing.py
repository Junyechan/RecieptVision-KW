
from preprocessing import preprocess_receipt
from ocr import recognize_text

results = preprocess_receipt(
    "images/sample_receipt.png"
)

if not results["success"]:
    raise RuntimeError(results["error"])

print("영수증 검출:", results["success"])
print("보정 이미지 크기:", results["warped"].shape)

ocr_image = results["ocr_image"]

print("OCR 입력 이미지 크기:", ocr_image.shape)
print("OCR 입력 데이터 타입:", ocr_image.dtype)

# OCR 테스트
ocr_result = recognize_text(ocr_image)

print("\n--- OCR 인식 결과 ---")
print(ocr_result["text"])

print("\n--- OCR 엔진 ---")
print(ocr_result["engine"])