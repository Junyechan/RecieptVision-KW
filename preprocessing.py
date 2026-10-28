
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


# 1. 이미지 불러오기
def load_image(image_path):
    path = Path(image_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {path}"
        )

    # 한글 경로 지원
    data = np.fromfile(str(path), dtype=np.uint8)

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"이미지 디코딩 실패: {path}"
        )

    return image


# 2. 네 꼭짓점 정렬
def order_points(points):
    points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(4, 2)

    x_sorted = points[np.argsort(points[:, 0])]

    left = x_sorted[:2]
    right = x_sorted[2:]

    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    tl, bl = left
    tr, br = right

    return np.array(
        [tl, tr, br, bl],
        dtype=np.float32
    )


# 3. 영수증 영역 검출
def detect_receipt(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]

    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    receipt_contour = None

    for contour in sorted_contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if (
            len(approx) == 4
            and cv2.isContourConvex(approx)
        ):
            receipt_contour = approx
            break

    return receipt_contour, edges, contours


# 4. 원근 보정
def four_point_transform(image, points):
    rect = order_points(points)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(
        1,
        int(round(max(width_a, width_b)))
    )

    max_height = max(
        1,
        int(round(max(height_a, height_b)))
    )

    if max_width < 2 or max_height < 2:
        raise ValueError("원근 보정 크기가 너무 작습니다.")

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )


# 5. 전체 이미지 전처리
def preprocess_receipt(image_path):
    image = load_image(image_path)

    # 기본 전처리
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 영수증 영역 검출
    receipt_contour, edges, contours = detect_receipt(
        image
    )

    contour_image = image.copy()
    detected_image = image.copy()

    cv2.drawContours(
        contour_image,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    detected = receipt_contour is not None

    if detected:
        cv2.drawContours(
            detected_image,
            [receipt_contour],
            -1,
            (0, 0, 255),
            3
        )

        warped = four_point_transform(
            image,
            receipt_contour.reshape(4, 2)
        )
    else:
        # 검출 실패 시 원본 사용
        warped = image.copy()

    # 원근 보정 후 전처리
    warped_gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    warped_blur = cv2.GaussianBlur(
        warped_gray,
        (3, 3),
        0
    )

    _, warped_otsu = cv2.threshold(
        warped_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    warped_adaptive = cv2.adaptiveThreshold(
        warped_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    return {
        "original": image,
        "gray": gray,
        "blur": blur,
        "otsu": otsu,
        "adaptive": adaptive,
        "canny": edges,
        "contours": contour_image,
        "detected_receipt": detected_image,
        "warped": warped,
        "warped_gray": warped_gray,
        "warped_otsu": warped_otsu,
        "warped_adaptive": warped_adaptive,
        "receipt_detected": detected
    }


# 6. 이미지 저장
def save_results(results, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, image in results.items():
        if not isinstance(image, np.ndarray):
            continue

        output_path = output_dir / f"{name}.png"

        success, encoded = cv2.imencode(
            ".png",
            image
        )

        if not success:
            raise ValueError(
                f"이미지 인코딩 실패: {name}"
            )

        encoded.tofile(str(output_path))


# 7. 결과 시각화
def show_results(results):
    images = [
        (name, value)
        for name, value in results.items()
        if isinstance(value, np.ndarray)
    ]

    plt.figure(figsize=(20, 14))

    for i, (name, image) in enumerate(images):
        plt.subplot(3, 4, i + 1)

        if image.ndim == 3:
            display_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
            plt.imshow(display_image)
        else:
            plt.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=255
            )

        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# 직접 실행할 때만 테스트
if __name__ == "__main__":
    image_path = "images/sample_receipt.jpg"

    results = preprocess_receipt(image_path)

    print(
        "영수증 검출:",
        "성공" if results["receipt_detected"] else "실패"
    )

    save_results(results)
    show_results(results)

    print("이미지 전처리 및 저장 완료!")
