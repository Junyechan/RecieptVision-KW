"""OCR 담당 팀원용: 전처리 모듈의 최소 사용 예제."""
from pathlib import Path
from preprocessing import preprocess_receipt, save_image

image_path = Path('images/receipt_01.jpg')
result = preprocess_receipt(image_path, method='clahe_gaussian_b21_c10')
if not result['success']:
    raise RuntimeError(f"영수증 전처리 실패: {result['error']}")

out = Path('outputs/ocr_handoff')
out.mkdir(parents=True, exist_ok=True)
save_image(out / 'warped.png', result['warped'])
save_image(out / 'ocr_ready.png', result['ocr_image'])
save_image(out / 'detection_overlay.png', result['detection_overlay'])

# OCR 담당자는 result['ocr_image'] (numpy uint8, 2D)를 OCR 엔진에 전달하면 됩니다.
# 원본 컬러 영수증이 필요하면 result['warped'] (numpy uint8, HxWx3)를 사용하세요.
print('corners:', result['corners'])
print('method:', result['method'])
print('ocr_image shape:', result['ocr_image'].shape)
print('saved:', out)
