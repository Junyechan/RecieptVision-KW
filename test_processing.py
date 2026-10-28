
from preprocessing import preprocess_receipt

results = preprocess_receipt(
    "images/sample_receipt.jpg"
)

print("영수증 검출:", results["receipt_detected"])

print("원본 이미지 크기:", results["original"].shape)
print("보정 이미지 크기:", results["warped"].shape)

# OCR에 전달할 이미지
ocr_image = results["warped_otsu"]

print("OCR 입력 이미지 크기:", ocr_image.shape)
print("OCR 입력 데이터 타입:", ocr_image.dtype)
