
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 폴더 경로
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "images" / "sample_receipt.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 1. 원본 이미지 불러오기

# 1. 원본 이미지 불러오기
import numpy as np

print("프로젝트 경로:", BASE_DIR)
print("이미지 경로:", IMAGE_PATH)
print("파일 존재 여부:", IMAGE_PATH.exists())

if IMAGE_PATH.exists():
    print("파일 크기:", IMAGE_PATH.stat().st_size, "bytes")

    image_data = np.fromfile(
        str(IMAGE_PATH),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR
    )

    print("이미지 디코딩 성공:", image is not None)
else:
    image = None
    print("images 폴더에 있는 파일:")
    for file in (BASE_DIR / "images").iterdir():
        print(repr(file.name))


if image is None:
    raise FileNotFoundError(
        f"이미지를 찾을 수 없습니다: {IMAGE_PATH}"
    )

# 2. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Otsu Threshold
_, otsu = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 5. Adaptive Threshold
adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# 결과 저장
results = {
    "original": image,
    "gray": gray,
    "blur": blur,
    "otsu": otsu,
    "adaptive": adaptive
}

for name, result in results.items():
    cv2.imwrite(
        str(OUTPUT_DIR / f"{name}.png"),
        result
    )

# 결과 시각화
plt.figure(figsize=(15, 8))

for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, i + 1)

    if result.ndim == 3:
        result = cv2.cvtColor(
            result, cv2.COLOR_BGR2RGB
        )
        plt.imshow(result)
    else:
        plt.imshow(
            result, cmap="gray",
            vmin=0, vmax=255
        )

    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("이미지 전처리 및 저장 완료!")
