"""ReceiptVision Pipeline V5 — self-contained experimental hybrid detector.
No GT/CSV/JSON is read for inference. Legacy algorithms are embedded in memory.
"""
from pathlib import Path
from itertools import combinations
from types import SimpleNamespace
import cv2
import numpy as np


def read_image(path):
    data=np.fromfile(str(path),dtype=np.uint8)
    image=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"이미지 읽기 실패: {path}")
    return image

load_image=read_image

def save_image(path,image):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    ext=path.suffix.lower()
    ok,buf=cv2.imencode(ext,image)
    if not ok: raise ValueError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))

def order_points(points):
    pts=np.asarray(points,dtype=np.float32).reshape(4,2)
    center=pts.mean(axis=0)
    angles=np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0])
    pts=pts[np.argsort(angles)]
    pts=np.roll(pts,-int(np.argmin(pts[:,0]+pts[:,1])),axis=0)
    v1=pts[1]-pts[0];v2=pts[2]-pts[1]
    if float(v1[0]*v2[1]-v1[1]*v2[0])<0:
        pts=pts[[0,3,2,1]]
    return pts.copy()

def valid_quad(points,width,height):
    p=order_points(points)
    if not np.isfinite(p).all() or not cv2.isContourConvex(p): return False
    area=abs(cv2.contourArea(p))/(width*height)
    if not .08<=area<=.985:return False
    sides=[np.linalg.norm(p[(i+1)%4]-p[i]) for i in range(4)]
    if min(sides)<20:return False
    if np.any(p[:,0]<-width*.15) or np.any(p[:,0]>width*1.15):return False
    if np.any(p[:,1]<-height*.15) or np.any(p[:,1]>height*1.15):return False
    return True

def warp_receipt(image,corners):
    p=order_points(corners);tl,tr,br,bl=p
    w=int(round(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))))
    h=int(round(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))
    if w<30 or h<30 or w*h>30000000:raise ValueError('잘못된 원근 보정 크기')
    dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(p,dst),(w,h),borderMode=cv2.BORDER_REPLICATE)

four_point_transform=warp_receipt

def make_ocr_variants(warped):
    gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    gb=cv2.GaussianBlur(gray,(3,3),0)
    cb=cv2.GaussianBlur(clahe,(3,3),0)
    def adapt(im,block):
        return cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,10)
    return {'grayscale':gray,'baseline_adaptive':adapt(gb,31),
            'clahe_gaussian_b21_c10':adapt(cb,21),
            'clahe_gaussian_b31_c10':adapt(cb,31)}

def draw_detection(image,corners):
    result=image.copy();pts=np.rint(corners).astype(np.int32)
    cv2.polylines(result,[pts],True,(0,255,0),3)
    for i,(x,y) in enumerate(pts):
        cv2.circle(result,(int(x),int(y)),6,(0,0,255),-1)
        cv2.putText(result,str(i),(int(x)+8,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
    return result

def _make_namespace(source,label):
    # Embedded original algorithm functions, not imported from test files.
    scope={'np':np,'cv2':cv2,'combinations':combinations,
           'order_points':order_points,'four_point_transform':four_point_transform,
           'MIN_AREA_RATIO':.08,'DEDUP_IOU':.995,'SAMPLE_DISTANCE':9,
           'WEIGHTS':{'edge':.27,'contrast':.25,'continuity':.18,'geometry':.10,'prior':.20},
           'EXTENSION_RATIOS':np.linspace(.03,.75,60),
           'MIN_EDGE_SCORE':.30,'MIN_CONTRAST_SCORE':.08,'MIN_SCORE_IMPROVEMENT':.04,
           'EDGE_WEIGHT':.20,'CONTRAST_WEIGHT':.40,'BOUNDARY_WEIGHT':.25,
           'COVERAGE_WEIGHT':.15,'EXTENSION_PENALTY':.08,
           'MIN_IMPROVEMENT':.025,'REQUIRE_BASELINE':True}
    exec(compile(source,'<embedded_'+label+'>','exec'),scope)
    return SimpleNamespace(**scope)

def polygon_iou(a,b):
    a=order_points(a);b=order_points(b)
    ia,_=cv2.intersectConvexConvex(a,b)
    union=abs(cv2.contourArea(a))+abs(cv2.contourArea(b))-ia
    return float(ia/union) if union>0 else 0.0

def detect_receipt(image):
    h,w=image.shape[:2]
    # Legacy detectors use pixel distances; keep original resolution for repeatability.
    v1 = _make_namespace('def make_masks(image):\n    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n    blur = cv2.GaussianBlur(gray, (5, 5), 0)\n\n    clahe = cv2.createCLAHE(\n        clipLimit=2.0, tileGridSize=(8, 8)\n    )\n    enhanced = clahe.apply(gray)\n\n    edges = cv2.Canny(\n        cv2.GaussianBlur(enhanced, (5, 5), 0),\n        40, 120\n    )\n\n    kernel = cv2.getStructuringElement(\n        cv2.MORPH_RECT, (5, 5)\n    )\n    closed = cv2.morphologyEx(\n        edges, cv2.MORPH_CLOSE, kernel\n    )\n\n    _, bright = cv2.threshold(\n        blur, 0, 255,\n        cv2.THRESH_BINARY + cv2.THRESH_OTSU\n    )\n\n    bright = cv2.morphologyEx(\n        bright, cv2.MORPH_CLOSE,\n        np.ones((9, 9), np.uint8)\n    )\n\n    return gray, edges, closed, bright\n\ndef collect_candidates(mask, source, image_shape):\n    h, w = image_shape[:2]\n    image_area = h * w\n\n    contours, _ = cv2.findContours(\n        mask, cv2.RETR_LIST,\n        cv2.CHAIN_APPROX_SIMPLE\n    )\n\n    candidates = []\n\n    for contour in contours:\n        area = cv2.contourArea(contour)\n\n        if not 0.08 * image_area <= area <= 0.85 * image_area:\n            continue\n\n        hull = cv2.convexHull(contour)\n        perimeter = cv2.arcLength(hull, True)\n\n        if perimeter == 0:\n            continue\n\n        for epsilon in [0.01, 0.02, 0.03, 0.05]:\n            approx = cv2.approxPolyDP(\n                hull, epsilon * perimeter, True\n            )\n\n            if len(approx) != 4:\n                continue\n\n            if not cv2.isContourConvex(approx):\n                continue\n\n            points = order_points(\n                approx.reshape(4, 2)\n            )\n\n            candidates.append({\n                "points": points,\n                "source": source\n            })\n            break\n\n        # 회전 사각형은 별도 후보로 추가\n        rect = cv2.minAreaRect(hull)\n        box = cv2.boxPoints(rect)\n        box = order_points(box)\n\n        candidates.append({\n            "points": box,\n            "source": source + "_rect"\n        })\n\n    return candidates\n\ndef line_support(points, edges):\n    h, w = edges.shape\n    support = []\n\n    for i in range(4):\n        p1 = points[i]\n        p2 = points[(i + 1) % 4]\n\n        length = np.linalg.norm(p2 - p1)\n        samples = max(10, int(length))\n\n        xs = np.linspace(p1[0], p2[0], samples)\n        ys = np.linspace(p1[1], p2[1], samples)\n\n        xs = np.clip(np.rint(xs).astype(int), 0, w - 1)\n        ys = np.clip(np.rint(ys).astype(int), 0, h - 1)\n\n        support.append(\n            np.mean(edges[ys, xs] > 0)\n        )\n\n    return float(np.mean(support))\n\ndef hough_support(points, lines, tolerance=8):\n    if lines is None:\n        return 0.0\n\n    values = []\n\n    for i in range(4):\n        a = points[i]\n        b = points[(i + 1) % 4]\n\n        side = b - a\n        side_length = np.linalg.norm(side)\n\n        if side_length < 1:\n            values.append(0.0)\n            continue\n\n        side_unit = side / side_length\n        best = 0.0\n\n        for line in lines.reshape(-1, 4):\n            p = np.array(line[:2], dtype=float)\n            q = np.array(line[2:], dtype=float)\n            direction = q - p\n            length = np.linalg.norm(direction)\n\n            if length < 1:\n                continue\n\n            parallel = abs(\n                np.dot(side_unit, direction / length)\n            )\n\n            if parallel < 0.97:\n                continue\n\n            # 후보 경계선과 검출 직선 사이 거리\n            distance = abs(\n                side[0] * (p[1] - a[1])\n                - side[1] * (p[0] - a[0])\n            ) / side_length\n\n            if distance > tolerance:\n                continue\n\n            # 후보 변 위에 투영된 겹침 길이\n            t1 = np.dot(p - a, side_unit)\n            t2 = np.dot(q - a, side_unit)\n\n            overlap = max(\n                0.0,\n                min(max(t1, t2), side_length)\n                - max(min(t1, t2), 0.0)\n            )\n\n            best = max(best, overlap / side_length)\n\n        values.append(best)\n\n    return float(np.mean(values))\n\ndef score_candidate(candidate, gray, edges, lines):\n    points = candidate["points"]\n    h, w = gray.shape\n\n    area = cv2.contourArea(\n        points.astype(np.float32)\n    )\n    area_ratio = area / (h * w)\n\n    if not 0.08 <= area_ratio <= 0.85:\n        return None\n\n    widths = [\n        np.linalg.norm(points[1] - points[0]),\n        np.linalg.norm(points[2] - points[3])\n    ]\n    heights = [\n        np.linalg.norm(points[3] - points[0]),\n        np.linalg.norm(points[2] - points[1])\n    ]\n\n    avg_width = np.mean(widths)\n    avg_height = np.mean(heights)\n\n    if avg_width < 10 or avg_height < 10:\n        return None\n\n    # 세로형 영수증을 우선 평가\n    aspect = avg_height / avg_width\n    aspect_score = float(\n        np.clip(1.0 - abs(aspect - 1.8) / 1.8, 0, 1)\n    )\n\n    polygon_mask = np.zeros_like(gray)\n    cv2.fillConvexPoly(\n        polygon_mask, points.astype(np.int32), 255\n    )\n\n    inside = gray[polygon_mask > 0]\n\n    if len(inside) == 0:\n        return None\n\n    brightness = float(np.mean(inside) / 255.0)\n\n    edge_score = line_support(points, edges)\n    hough_score = hough_support(points, lines)\n\n    # 면적이 너무 작은 후보를 억제\n    area_score = float(\n        np.clip(area_ratio / 0.30, 0, 1)\n    )\n\n    score = (\n        0.20 * aspect_score\n        + 0.25 * brightness\n        + 0.25 * edge_score\n        + 0.20 * hough_score\n        + 0.10 * area_score\n    )\n\n    return float(score)\n\ndef detect_hybrid(image):\n    gray, edges, closed, bright = make_masks(image)\n\n    candidates = []\n    candidates += collect_candidates(\n        closed, "contour", image.shape\n    )\n    candidates += collect_candidates(\n        bright, "brightness", image.shape\n    )\n\n    h, w = gray.shape\n\n    lines = cv2.HoughLinesP(\n        edges,\n        rho=1,\n        theta=np.pi / 180,\n        threshold=70,\n        minLineLength=int(min(h, w) * 0.35),\n        maxLineGap=15\n    )\n\n    evaluated = []\n\n    for candidate in candidates:\n        score = score_candidate(\n            candidate, gray, edges, lines\n        )\n\n        if score is None:\n            continue\n\n        candidate["score"] = score\n        evaluated.append(candidate)\n\n    evaluated.sort(\n        key=lambda x: x["score"],\n        reverse=True\n    )\n\n    best = evaluated[0] if evaluated else None\n\n    return best, evaluated, edges, bright', 'v1')
    v2 = _make_namespace('def detect_boundary_lines(image):\n    gray, edges, _, _ = make_masks(image)\n    h, w = gray.shape\n\n    raw_lines = cv2.HoughLinesP(\n        edges,\n        rho=1,\n        theta=np.pi / 360,\n        threshold=55,\n        minLineLength=int(min(h, w) * 0.25),\n        maxLineGap=25\n    )\n\n    vertical = []\n    horizontal = []\n\n    if raw_lines is None:\n        return vertical, horizontal, edges\n\n    for item in raw_lines.reshape(-1, 4):\n        x1, y1, x2, y2 = map(float, item)\n\n        dx = x2 - x1\n        dy = y2 - y1\n        length = np.hypot(dx, dy)\n\n        if length < 20:\n            continue\n\n        angle = abs(\n            np.degrees(np.arctan2(dy, dx))\n        )\n\n        line = {\n            "p": np.array([x1, y1], dtype=float),\n            "q": np.array([x2, y2], dtype=float),\n            "length": length,\n            "angle": angle\n        }\n\n        # 세로 방향 경계\n        if 55 <= angle <= 125:\n            vertical.append(line)\n\n        # 가로 방향 경계\n        elif angle <= 35 or angle >= 145:\n            horizontal.append(line)\n\n    vertical.sort(\n        key=lambda x: x["length"], reverse=True\n    )\n    horizontal.sort(\n        key=lambda x: x["length"], reverse=True\n    )\n\n    return vertical, horizontal, edges\n\ndef deduplicate_lines(lines, limit=12):\n    selected = []\n\n    for line in lines:\n        p = line["p"]\n        q = line["q"]\n        direction = q - p\n        norm = np.linalg.norm(direction)\n\n        if norm < 1:\n            continue\n\n        unit = direction / norm\n        midpoint = (p + q) / 2\n\n        duplicate = False\n\n        for old in selected:\n            op, oq = old["p"], old["q"]\n            old_dir = oq - op\n            old_unit = old_dir / np.linalg.norm(old_dir)\n\n            parallel = abs(np.dot(unit, old_unit))\n\n            # 두 직선이 거의 평행하고 같은 선 위에 있는지\n            distance = abs(\n                direction[0] * (op[1] - p[1])\n                - direction[1] * (op[0] - p[0])\n            ) / norm\n\n            old_mid = (op + oq) / 2\n\n            # 같은 직선 위라도 멀리 떨어진 선분은 보존\n            separation = np.linalg.norm(midpoint - old_mid)\n\n            if (\n                parallel > 0.995\n                and distance < 12\n                and separation < max(norm, np.linalg.norm(old_dir))\n            ):\n                duplicate = True\n                break\n\n        if not duplicate:\n            selected.append(line)\n\n        if len(selected) >= limit:\n            break\n\n    return selected\n\ndef intersection(line1, line2):\n    p = line1["p"]\n    r = line1["q"] - p\n\n    q = line2["p"]\n    s = line2["q"] - q\n\n    denominator = r[0] * s[1] - r[1] * s[0]\n\n    if abs(denominator) < 1e-6:\n        return None\n\n    qp = q - p\n\n    t = (\n        qp[0] * s[1] - qp[1] * s[0]\n    ) / denominator\n\n    return p + t * r\n\ndef validate_quad(points, image_shape):\n    h, w = image_shape[:2]\n\n    if points is None:\n        return False\n\n    points = np.asarray(points, dtype=np.float32)\n\n    if points.shape != (4, 2):\n        return False\n\n    if not np.isfinite(points).all():\n        return False\n\n    # 이미지 경계 밖의 꼭짓점도 일정 범위 허용\n    margin_x = w * 0.15\n    margin_y = h * 0.15\n\n    if (\n        np.any(points[:, 0] < -margin_x)\n        or np.any(points[:, 0] > w - 1 + margin_x)\n        or np.any(points[:, 1] < -margin_y)\n        or np.any(points[:, 1] > h - 1 + margin_y)\n    ):\n        return False\n\n    area = cv2.contourArea(points)\n    area_ratio = area / (w * h)\n\n    if not 0.08 <= area_ratio <= 0.85:\n        return False\n\n    if not cv2.isContourConvex(\n        points.reshape(-1, 1, 2)\n    ):\n        return False\n\n    tl, tr, br, bl = points\n\n    width = (\n        np.linalg.norm(tr - tl)\n        + np.linalg.norm(br - bl)\n    ) / 2\n\n    height = (\n        np.linalg.norm(bl - tl)\n        + np.linalg.norm(br - tr)\n    ) / 2\n\n    if width < 30 or height < 30:\n        return False\n\n    aspect = height / width\n\n    # 현재 테스트 데이터의 세로형 영수증 가정\n    if not 0.9 <= aspect <= 4.5:\n        return False\n\n    return True\n\ndef generate_hough_candidates(image):\n    vertical, horizontal, edges = (\n        detect_boundary_lines(image)\n    )\n\n    vertical = deduplicate_lines(vertical, 12)\n    horizontal = deduplicate_lines(horizontal, 12)\n\n    candidates = []\n\n    for left, right in combinations(vertical, 2):\n        d1 = left["q"] - left["p"]\n        d2 = right["q"] - right["p"]\n\n        cosine = abs(\n            np.dot(d1, d2)\n            / (np.linalg.norm(d1) * np.linalg.norm(d2))\n        )\n\n        if cosine < 0.94:\n            continue\n\n        for top, bottom in combinations(horizontal, 2):\n            p1 = intersection(left, top)\n            p2 = intersection(right, top)\n            p3 = intersection(right, bottom)\n            p4 = intersection(left, bottom)\n\n            if any(\n                p is None for p in [p1, p2, p3, p4]\n            ):\n                continue\n\n            points = order_points(\n                np.array([p1, p2, p3, p4])\n            )\n\n            if not validate_quad(points, image.shape):\n                continue\n\n            candidates.append({\n                "points": points,\n                "source": "hough"\n            })\n\n    return candidates, edges, vertical, horizontal\n\ndef evaluate_candidates(image, candidates):\n    gray, edges, _, _ = make_masks(image)\n    h, w = gray.shape\n\n    lines = cv2.HoughLinesP(\n        edges, 1, np.pi / 180,\n        threshold=70,\n        minLineLength=int(min(h, w) * 0.35),\n        maxLineGap=15\n    )\n\n    evaluated = []\n\n    for candidate in candidates:\n        points = candidate["points"]\n\n        # 기존 V1 평가 함수 사용\n        score = score_candidate(\n            candidate, gray, edges, lines\n        )\n\n        if score is None:\n            continue\n\n        evaluated.append({\n            "points": points,\n            "source": candidate["source"],\n            "score": float(score)\n        })\n\n    evaluated.sort(\n        key=lambda x: x["score"],\n        reverse=True\n    )\n\n    return evaluated', 'v2')
    v3 = _make_namespace('def make_feature_maps(image):\n    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n\n    # 조명 변화에 따른 밝기 편차를 줄이기 위한 보정\n    gray_float = gray.astype(np.float32)\n    background = cv2.GaussianBlur(\n        gray_float, (0, 0), sigmaX=25\n    )\n\n    normalized = np.clip(\n        gray_float - background + 128,\n        0, 255\n    ).astype(np.uint8)\n\n    blur = cv2.GaussianBlur(gray, (5, 5), 0)\n    edges = cv2.Canny(blur, 50, 150)\n\n    return gray, normalized, edges\n\ndef sample_values(image, coordinates):\n    height, width = image.shape[:2]\n\n    x = np.rint(coordinates[:, 0]).astype(int)\n    y = np.rint(coordinates[:, 1]).astype(int)\n\n    valid = (\n        (x >= 0) & (x < width) &\n        (y >= 0) & (y < height)\n    )\n\n    values = np.zeros(len(x), dtype=np.float32)\n    values[valid] = image[y[valid], x[valid]]\n\n    return values, valid\n\ndef evaluate_boundary(gray, edges, points):\n    """\n    각 변의 내부와 외부 밝기 차이,\n    에지 연속성 및 경계의 밝기 방향 평가.\n    """\n    center = np.mean(points, axis=0)\n\n    contrast_scores = []\n    edge_scores = []\n    direction_scores = []\n    valid_scores = []\n\n    for i in range(4):\n        a = points[i]\n        b = points[(i + 1) % 4]\n\n        side = b - a\n        length = np.linalg.norm(side)\n\n        if length < 20:\n            return 0.0, 0.0, 0.0, 0.0\n\n        tangent = side / length\n        normal = np.array(\n            [-tangent[1], tangent[0]],\n            dtype=np.float32\n        )\n\n        midpoint = (a + b) / 2\n\n        # 법선이 사각형 내부를 향하도록 정렬\n        if np.dot(normal, center - midpoint) < 0:\n            normal = -normal\n\n        count = max(20, int(length / 3))\n        t = np.linspace(0.08, 0.92, count)\n        samples = a[None, :] + t[:, None] * side\n\n        inside = samples + normal[None, :] * 6\n        outside = samples - normal[None, :] * 6\n\n        in_values, in_valid = sample_values(gray, inside)\n        out_values, out_valid = sample_values(gray, outside)\n\n        valid = in_valid & out_valid\n        valid_scores.append(float(np.mean(valid)))\n\n        if np.sum(valid) < 10:\n            contrast_scores.append(0.0)\n            direction_scores.append(0.0)\n            edge_scores.append(0.0)\n            continue\n\n        difference = in_values[valid] - out_values[valid]\n\n        # 절대 밝기 차이\n        contrast = np.mean(\n            np.clip(np.abs(difference) / 60.0, 0, 1)\n        )\n\n        # 영수증이 배경보다 밝다는 가정\n        direction = np.mean(\n            np.clip(difference / 60.0, 0, 1)\n        )\n\n        # 경계 주변 3px 범위의 에지 존재 여부\n        edge_presence = np.zeros(count, dtype=bool)\n\n        for offset in [-2, -1, 0, 1, 2]:\n            positions = (\n                samples + normal[None, :] * offset\n            )\n            values, edge_valid = sample_values(\n                edges, positions\n            )\n            edge_presence |= (values > 0) & edge_valid\n\n        edge_score = np.mean(edge_presence)\n\n        contrast_scores.append(float(contrast))\n        direction_scores.append(float(direction))\n        edge_scores.append(float(edge_score))\n\n    return (\n        float(np.mean(contrast_scores)),\n        float(np.mean(edge_scores)),\n        float(np.mean(direction_scores)),\n        float(np.mean(valid_scores))\n    )\n\ndef evaluate_geometry(points, image_shape):\n    height, width = image_shape[:2]\n\n    area = abs(cv2.contourArea(points))\n    area_ratio = area / (height * width)\n\n    lengths = [\n        np.linalg.norm(\n            points[(i + 1) % 4] - points[i]\n        )\n        for i in range(4)\n    ]\n\n    if min(lengths) < 20 or area_ratio < 0.08:\n        return 0.0\n\n    if not cv2.isContourConvex(\n        points.astype(np.float32).reshape(-1, 1, 2)\n    ):\n        return 0.0\n\n    # 서로 마주 보는 변의 길이 일관성\n    horizontal = min(lengths[0], lengths[2]) / max(\n        lengths[0], lengths[2]\n    )\n    vertical = min(lengths[1], lengths[3]) / max(\n        lengths[1], lengths[3]\n    )\n\n    shape_score = (horizontal + vertical) / 2\n\n    # 매우 작은 후보만 불리하게 평가\n    area_score = np.clip(area_ratio / 0.25, 0, 1)\n\n    return float(\n        0.65 * shape_score + 0.35 * area_score\n    )\n\ndef score_candidate(candidate, gray, edges):\n    points = candidate["points"]\n\n    contrast, edge, direction, valid = (\n        evaluate_boundary(gray, edges, points)\n    )\n\n    geometry = evaluate_geometry(points, gray.shape)\n\n    # V2 점수와 새로운 경계 점수를 결합\n    score = (\n        0.20 * candidate["v2_score"]\n        + 0.20 * contrast\n        + 0.20 * edge\n        + 0.25 * direction\n        + 0.15 * geometry\n    )\n\n    # 이미지 외부에 걸친 후보에 대한 불이익\n    score *= 0.5 + 0.5 * valid\n\n    return {\n        **candidate,\n        "v3_score": float(score),\n        "contrast": contrast,\n        "edge": edge,\n        "direction": direction,\n        "geometry": geometry,\n        "valid": valid\n    }', 'v3')
    v41 = _make_namespace('def polygon_iou(points1, points2):\n    """\n    볼록 사각형 두 개의 연속 좌표 기반 IoU.\n    """\n    p1 = order_points(points1)\n    p2 = order_points(points2)\n\n    area1 = abs(cv2.contourArea(p1))\n    area2 = abs(cv2.contourArea(p2))\n\n    if area1 < 1 or area2 < 1:\n        return 0.0\n\n    try:\n        intersection, _ = cv2.intersectConvexConvex(\n            p1, p2\n        )\n    except cv2.error:\n        return 0.0\n\n    union = area1 + area2 - intersection\n\n    if union <= 0:\n        return 0.0\n\n    return float(\n        np.clip(intersection / union, 0, 1)\n    )\n\ndef remove_duplicates(candidates):\n    """\n    거의 동일한 사각형을 하나로 통합한다.\n    기존 V4 점수가 높은 후보부터 보존한다.\n    """\n    candidates = sorted(\n        candidates,\n        key=lambda c: c["v4_score"],\n        reverse=True\n    )\n\n    unique = []\n\n    for candidate in candidates:\n        points = candidate["points"]\n\n        duplicated = False\n\n        for saved in unique:\n            overlap = polygon_iou(\n                points,\n                saved["points"]\n            )\n\n            if overlap >= DEDUP_IOU:\n                duplicated = True\n                break\n\n        if not duplicated:\n            unique.append(candidate)\n\n    return unique\n\ndef sample_side(gray, edges, p1, p2, center, distance):\n    """\n    후보 사각형의 한 변에서\n    안쪽과 바깥쪽의 밝기 차이,\n    에지 일치율,\n    바깥쪽 종이 연속성 등을 평가한다.\n\n    이미지 바깥의 픽셀은 평가에서 제외한다.\n    """\n    h, w = gray.shape\n\n    p1 = np.asarray(p1, dtype=np.float32)\n    p2 = np.asarray(p2, dtype=np.float32)\n\n    vector = p2 - p1\n    length = float(np.linalg.norm(vector))\n\n    if length < 5:\n        return None\n\n    tangent = vector / length\n\n    normal = np.array(\n        [-tangent[1], tangent[0]],\n        dtype=np.float32\n    )\n\n    midpoint = (p1 + p2) / 2\n\n    # 중심을 향하는 방향이 내부 방향\n    if np.dot(normal, center - midpoint) < 0:\n        normal = -normal\n\n    count = max(20, int(length / 3))\n\n    ts = np.linspace(\n        0.08, 0.92, count\n    ).astype(np.float32)\n\n    base = (\n        p1[None, :]\n        + ts[:, None] * vector[None, :]\n    )\n\n    inside = base + normal * distance\n    outside = base - normal * distance\n\n    # 에지 위치의 약간의 오차 허용\n    edge_map = cv2.dilate(\n        edges,\n        np.ones((5, 5), np.uint8)\n    )\n\n    def valid_positions(positions):\n        x = positions[:, 0]\n        y = positions[:, 1]\n\n        return (\n            (x >= 0) & (x < w)\n            & (y >= 0) & (y < h)\n        )\n\n    valid = (\n        valid_positions(base)\n        & valid_positions(inside)\n        & valid_positions(outside)\n    )\n\n    # 화면에 걸친 경계는 평가할 정보가 부족함\n    if np.count_nonzero(valid) < 8:\n        return None\n\n    base = base[valid]\n    inside = inside[valid]\n    outside = outside[valid]\n\n    def sample(image, positions):\n        xs = np.clip(\n            np.rint(positions[:, 0]).astype(int),\n            0, w - 1\n        )\n        ys = np.clip(\n            np.rint(positions[:, 1]).astype(int),\n            0, h - 1\n        )\n        return image[ys, xs].astype(np.float32)\n\n    inner_values = sample(gray, inside)\n    outer_values = sample(gray, outside)\n\n    boundary_edges = sample(edge_map, base)\n\n    # 절대 밝기 차이\n    difference = np.abs(\n        inner_values - outer_values\n    )\n\n    contrast_score = float(\n        np.clip(np.mean(difference) / 65.0, 0, 1)\n    )\n\n    edge_score = float(\n        np.mean(boundary_edges > 0)\n    )\n\n    # 내부 구분선의 경우 양쪽 밝기가 비슷할 가능성이 큼\n    # 밝기 차이가 작은 샘플의 비율\n    similar_ratio = float(\n        np.mean(difference < 18)\n    )\n\n    # 안쪽이 바깥쪽보다 밝은 비율\n    # 흰 영수증 + 어두운 배경을 가정한 보조 신호\n    brighter_inside = float(\n        np.mean(inner_values > outer_values + 12)\n    )\n\n    # 내부와 외부가 비슷하고 경계 대비가 낮을수록 감점\n    continuity_penalty = (\n        similar_ratio * (1.0 - contrast_score)\n    )\n\n    continuity_score = float(\n        np.clip(\n            0.55 * (1.0 - continuity_penalty)\n            + 0.45 * brighter_inside,\n            0, 1\n        )\n    )\n\n    return {\n        "edge": edge_score,\n        "contrast": contrast_score,\n        "continuity": continuity_score,\n        "valid_count": int(np.count_nonzero(valid))\n    }\n\ndef geometry_score(points, image_shape):\n    h, w = image_shape[:2]\n\n    points = order_points(points)\n\n    area = abs(cv2.contourArea(points))\n    ratio = area / (h * w)\n\n    if ratio < MIN_AREA_RATIO:\n        return 0.0\n\n    if not cv2.isContourConvex(points):\n        return 0.0\n\n    lengths = [\n        float(np.linalg.norm(\n            points[(i + 1) % 4] - points[i]\n        ))\n        for i in range(4)\n    ]\n\n    if min(lengths) < 5:\n        return 0.0\n\n    # 서로 마주 보는 변의 길이 비율\n    opposite1 = (\n        min(lengths[0], lengths[2])\n        / max(lengths[0], lengths[2])\n    )\n\n    opposite2 = (\n        min(lengths[1], lengths[3])\n        / max(lengths[1], lengths[3])\n    )\n\n    # 원근 왜곡을 허용하도록 약한 가중치 적용\n    shape_score = (\n        opposite1 + opposite2\n    ) / 2\n\n    return float(\n        np.clip(shape_score, 0, 1)\n    )\n\ndef evaluate_candidate(image, gray, edges, candidate):\n    points = candidate["points"]\n\n    h, w = image.shape[:2]\n\n    area_ratio = (\n        abs(cv2.contourArea(points))\n        / (h * w)\n    )\n\n    if area_ratio < MIN_AREA_RATIO:\n        return None\n\n    if not cv2.isContourConvex(points):\n        return None\n\n    center = np.mean(points, axis=0)\n\n    side_results = []\n\n    for i in range(4):\n        p1 = points[i]\n        p2 = points[(i + 1) % 4]\n\n        result = sample_side(\n            gray,\n            edges,\n            p1,\n            p2,\n            center,\n            SAMPLE_DISTANCE\n        )\n\n        if result is not None:\n            side_results.append(result)\n\n    if not side_results:\n        return None\n\n    edge_values = [\n        s["edge"] for s in side_results\n    ]\n\n    contrast_values = [\n        s["contrast"] for s in side_results\n    ]\n\n    continuity_values = [\n        s["continuity"] for s in side_results\n    ]\n\n    # 네 변 중 약한 변도 고려\n    edge_score = (\n        0.65 * np.mean(edge_values)\n        + 0.35 * np.min(edge_values)\n    )\n\n    contrast_score = (\n        0.70 * np.mean(contrast_values)\n        + 0.30 * np.min(contrast_values)\n    )\n\n    continuity_score = float(\n        np.mean(continuity_values)\n    )\n\n    geo_score = geometry_score(\n        points, image.shape\n    )\n\n    # V4 점수는 보조 지표로만 사용\n    prior_score = float(\n        np.clip(candidate["v4_score"], 0, 1)\n    )\n\n    # 평가 가능한 변이 적을 때 신뢰도 감소\n    coverage = len(side_results) / 4\n\n    score = (\n        WEIGHTS["edge"] * edge_score\n        + WEIGHTS["contrast"] * contrast_score\n        + WEIGHTS["continuity"] * continuity_score\n        + WEIGHTS["geometry"] * geo_score\n        + WEIGHTS["prior"] * prior_score\n    )\n\n    # 이미지 경계에 걸친 영수증도 있으므로\n    # 평가 불가능한 변을 강하게 감점하지 않음\n    score *= (\n        0.90 + 0.10 * coverage\n    )\n\n    return {\n        **candidate,\n        "v4_1_score": float(score),\n        "edge_score": float(edge_score),\n        "contrast_score": float(contrast_score),\n        "continuity_score": float(continuity_score),\n        "geometry_score": float(geo_score),\n        "evaluated_sides": len(side_results),\n        "area_ratio": float(area_ratio)\n    }', 'v41')
    v43 = _make_namespace('def as_points(points):\n    return np.asarray(\n        points,\n        dtype=np.float32\n    ).reshape(4, 2)\n\ndef extend_top(points, ratio):\n    p = as_points(points)\n\n    tl, tr, br, bl = p\n\n    new_tl = (\n        tl + ratio * (tl - bl)\n    )\n\n    new_tr = (\n        tr + ratio * (tr - br)\n    )\n\n    return np.array(\n        [\n            new_tl,\n            new_tr,\n            br,\n            bl\n        ],\n        dtype=np.float32\n    )\n\ndef valid_quad(points, shape):\n    h, w = shape[:2]\n\n    p = as_points(points)\n\n    if not np.isfinite(p).all():\n        return False\n\n    if not cv2.isContourConvex(p):\n        return False\n\n    area = abs(\n        cv2.contourArea(p)\n    )\n\n    if area < 0.08 * w * h:\n        return False\n\n    if np.any(\n        p[:, 0] < 0\n    ):\n        return False\n\n    if np.any(\n        p[:, 0] >= w\n    ):\n        return False\n\n    if np.any(\n        p[:, 1] < 0\n    ):\n        return False\n\n    if np.any(\n        p[:, 1] >= h\n    ):\n        return False\n\n    return True\n\ndef evaluate_top(\n    gray,\n    edges,\n    points\n):\n    """\n    상단 경계의 에지 강도,\n    밝기 차이, 평가 가능 범위를 계산.\n\n    내부가 외부보다 밝아야 한다는\n    조건은 사용하지 않는다.\n    """\n    h, w = gray.shape\n\n    p = as_points(points)\n\n    tl, tr, br, bl = p\n\n    top_vector = tr - tl\n\n    length = float(\n        np.linalg.norm(\n            top_vector\n        )\n    )\n\n    if length < 10:\n        return None\n\n    top_mid = (\n        tl + tr\n    ) / 2\n\n    bottom_mid = (\n        bl + br\n    ) / 2\n\n    inward = (\n        bottom_mid - top_mid\n    )\n\n    norm = float(\n        np.linalg.norm(\n            inward\n        )\n    )\n\n    if norm < 1e-6:\n        return None\n\n    inward /= norm\n\n    count = max(\n        30,\n        int(length / 2)\n    )\n\n    ts = np.linspace(\n        0.05,\n        0.95,\n        count\n    ).astype(\n        np.float32\n    )\n\n    base = (\n        tl[None, :]\n        + ts[:, None]\n        * top_vector[None, :]\n    )\n\n    inside = (\n        base\n        + inward\n        * SAMPLE_DISTANCE\n    )\n\n    outside = (\n        base\n        - inward\n        * SAMPLE_DISTANCE\n    )\n\n    def valid(positions):\n        return (\n            (positions[:, 0] >= 0)\n            & (positions[:, 0] < w)\n            & (positions[:, 1] >= 0)\n            & (positions[:, 1] < h)\n        )\n\n    mask = (\n        valid(base)\n        & valid(inside)\n        & valid(outside)\n    )\n\n    valid_count = int(\n        np.count_nonzero(mask)\n    )\n\n    if valid_count < 8:\n        return None\n\n    base = base[mask]\n    inside = inside[mask]\n    outside = outside[mask]\n\n    def sample(img, pos):\n        x = np.rint(\n            pos[:, 0]\n        ).astype(int)\n\n        y = np.rint(\n            pos[:, 1]\n        ).astype(int)\n\n        return img[y, x].astype(\n            np.float32\n        )\n\n    inner_values = sample(\n        gray,\n        inside\n    )\n\n    outer_values = sample(\n        gray,\n        outside\n    )\n\n    expanded_edges = cv2.dilate(\n        edges,\n        np.ones(\n            (5, 5),\n            np.uint8\n        )\n    )\n\n    edge_values = sample(\n        expanded_edges,\n        base\n    )\n\n    difference = np.abs(\n        inner_values\n        - outer_values\n    )\n\n    edge_score = float(\n        np.mean(\n            edge_values > 0\n        )\n    )\n\n    contrast_score = float(\n        np.clip(\n            np.mean(\n                difference\n            ) / 65,\n            0,\n            1\n        )\n    )\n\n    coverage = (\n        valid_count / count\n    )\n\n    # 경계가 명확하면서\n    # 평가 가능한 구간이 넓은\n    # 후보에 높은 점수를 준다.\n    score = (\n        0.55 * edge_score\n        + 0.35 * contrast_score\n        + 0.10 * coverage\n    )\n\n    return {\n        "score": float(score),\n        "edge": edge_score,\n        "contrast": contrast_score,\n        "coverage": float(coverage)\n    }\n\ndef search_top_candidates(\n    image,\n    original\n):\n    gray = cv2.cvtColor(\n        image,\n        cv2.COLOR_BGR2GRAY\n    )\n\n    gray = cv2.GaussianBlur(\n        gray,\n        (5, 5),\n        0\n    )\n\n    edges = cv2.Canny(\n        gray,\n        40,\n        120\n    )\n\n    candidates = []\n\n    # 원본 상단 포함\n    ratios = [\n        0.0\n    ] + list(\n        EXTENSION_RATIOS\n    )\n\n    for ratio in ratios:\n        points = extend_top(\n            original,\n            float(ratio)\n        )\n\n        if not valid_quad(\n            points,\n            image.shape\n        ):\n            continue\n\n        info = evaluate_top(\n            gray,\n            edges,\n            points\n        )\n\n        if info is None:\n            continue\n\n        candidates.append({\n            "ratio": float(ratio),\n            "points": points,\n            **info\n        })\n\n    candidates.sort(\n        key=lambda c: c["score"],\n        reverse=True\n    )\n\n    return candidates', 'v43')
    v44 = _make_namespace('def score_candidate(candidate):\n    edge = float(np.clip(\n        candidate["edge"], 0, 1\n    ))\n    contrast = float(np.clip(\n        candidate["contrast"], 0, 1\n    ))\n    coverage = float(np.clip(\n        candidate["coverage"], 0, 1\n    ))\n    ratio = max(0.0, candidate["ratio"])\n\n    # 밝기 차이가 클수록\n    # 내부 구분선이 아닐 가능성을 높게 평가\n    # (contrast와 상관된 보조 지표)\n    boundary = float(np.clip(\n        contrast / 0.45, 0, 1\n    ))\n\n    score = (\n        EDGE_WEIGHT * edge\n        + CONTRAST_WEIGHT * contrast\n        + BOUNDARY_WEIGHT * boundary\n        + COVERAGE_WEIGHT * coverage\n        - EXTENSION_PENALTY * ratio\n    )\n\n    return float(score), boundary\n\ndef choose_candidate(candidates, original):\n    if not candidates:\n        return original, "no_candidates", None\n\n    # 기존 상단은 ratio=0\n    baseline = min(\n        candidates,\n        key=lambda c: abs(c["ratio"])\n    )\n\n    # V4.3에서 원본 후보가 누락된 경우\n    if (\n        REQUIRE_BASELINE\n        and abs(baseline["ratio"]) > 1e-6\n    ):\n        return (\n            original,\n            "baseline_not_evaluated",\n            None\n        )\n\n    best = max(\n        candidates,\n        key=lambda c: c["v44_score"]\n    )\n\n    improvement = (\n        best["v44_score"]\n        - baseline["v44_score"]\n    )\n\n    if best["ratio"] <= 0:\n        return (\n            original,\n            "baseline_best",\n            baseline\n        )\n\n    if improvement < MIN_IMPROVEMENT:\n        return (\n            original,\n            "insufficient_improvement",\n            baseline\n        )\n\n    if best["coverage"] < 0.60:\n        return (\n            original,\n            "low_coverage",\n            baseline\n        )\n\n    return (\n        best["points"],\n        "corrected",\n        best\n    )', 'v44')
    # V2 imports the V1 mask/scoring functions in the original project.
    v2.detect_boundary_lines.__globals__['make_masks']=v1.make_masks
    v2.evaluate_candidates.__globals__['make_masks']=v1.make_masks
    v2.evaluate_candidates.__globals__['score_candidate']=v1.score_candidate

    # V1: contour / Otsu candidates
    _, base_candidates, _, _ = v1.detect_hybrid(image)
    # V2: Hough line-intersection candidates, plus V1
    hough,_,_,_ = v2.generate_hough_candidates(image)
    candidates=base_candidates+hough
    if not candidates:
        return {'success':False,'error':'후보 없음','corners':None,'candidate_count':0,'top_candidates':[]}
    v2_scores=v2.evaluate_candidates(image,candidates)
    if not v2_scores:
        return {'success':False,'error':'점수 평가 후보 없음','corners':None,'candidate_count':0,'top_candidates':[]}
    gray,_,edges=v3.make_feature_maps(image)
    rescored=[]
    for rank,c in enumerate(v2_scores,1):
        p=order_points(c['points'])
        if not valid_quad(p,w,h):continue
        item=v3.score_candidate({'points':p,'v2_score':c['score'],'v2_rank':rank,'source':c['source']},gray,edges)
        # V4 prior score, calculated in memory using the original V4 function.
        rescored.append(item)
    if not rescored:
        return {'success':False,'error':'유효 후보 없음','corners':None,'candidate_count':0,'top_candidates':[]}
    # V4-style merged candidate evaluation, with v3 score as prior.
    from math import isfinite
    merged=[]
    for c in rescored:
        p=c['points']
        # V4 weighted edge, area, corner validity
        blur=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(5,5),0)
        canny=cv2.Canny(blur,40,120)
        boundary=np.zeros((h,w),np.uint8)
        cv2.polylines(boundary,[np.rint(p).astype(np.int32)],True,255,2)
        near=cv2.dilate(canny,np.ones((5,5),np.uint8))
        edge=float(np.mean(near[boundary>0]>0)) if np.any(boundary) else 0.0
        area=abs(cv2.contourArea(p))/(h*w)
        validity=float(np.mean((p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)))
        v4score=.55*edge+.25*min(area/.5,1)+.20*validity
        merged.append({**c,'v4_score':v4score})
    merged.sort(key=lambda c:c['v4_score'],reverse=True)
    # V4.1: duplicate suppression + edge/contrast/continuity reranking
    unique=v41.remove_duplicates(merged)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),40,120)
    evaluated=[v41.evaluate_candidate(image,gray,edges,c) for c in unique]
    evaluated=[c for c in evaluated if c is not None and isfinite(c['v4_1_score'])]
    evaluated.sort(key=lambda c:c['v4_1_score'],reverse=True)
    if not evaluated:
        return {'success':False,'error':'V4.1 후보 없음','corners':None,'candidate_count':0,'top_candidates':[]}
    best=evaluated[0]
    original=order_points(best['points'])
    # V4.3: generate top-boundary extensions in memory
    extensions=v43.search_top_candidates(image,original)
    # V4.4: score each extension and select if sufficiently better
    rescored_top=[]
    for c in extensions:
        score,boundary=v44.score_candidate(c)
        rescored_top.append({**c,'v44_score':score,'boundary':boundary})
    selected,reason,chosen=v44.choose_candidate(rescored_top,original)
    selected=order_points(selected)
    if not valid_quad(selected,w,h):selected=original;reason='invalid_extension'
    top=[{'rank':i+1,'score':float(c['v4_1_score']),
          'source':c['source'],'corners':c['points'].tolist()}
         for i,c in enumerate(evaluated[:10])]
    return {'success':True,'error':None,'corners':selected,'score':float(best['v4_1_score']),
            'source':'hybrid_v5_'+reason,'candidate_count':len(evaluated),
            'top_candidates':top,'top_correction':reason}

def make_ocr_safe_corners(corners, image_shape, correction_reason):
    """Keep detection corners intact; expand OCR crop toward top image boundary
    only when the automatic top-edge correction was used and a slanted
    near-top edge could cut off merchant text. No GT or image-specific names.
    """
    points = order_points(corners).copy()
    height, width = image_shape[:2]
    top_y = points[:2, 1]
    if correction_reason != 'corrected':
        return points
    # A corrected top edge near the image boundary, with substantial slope.
    if not (np.min(top_y) < 0.05 * height and
            np.max(top_y) < 0.16 * height and
            np.max(top_y) - np.min(top_y) > 0.04 * height):
        return points
    for top_index, bottom_index in ((0, 3), (1, 2)):
        top = points[top_index]
        bottom = points[bottom_index]
        dy = float(bottom[1] - top[1])
        if dy <= 1.0:
            continue
        # Extrapolate each side to the upper image boundary, without
        # synthesizing pixels beyond the original photograph.
        factor = float(top[1]) / dy
        extended = top + factor * (top - bottom)
        if 0 <= extended[0] < width:
            points[top_index] = extended
    if valid_quad(points, width, height):
        return points
    return order_points(corners)


def preprocess_receipt(image_input,method='clahe_gaussian_b21_c10'):
    try:
        image=read_image(image_input) if isinstance(image_input,(str,Path)) else image_input.copy()
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('BGR 이미지 필요')
        result=detect_receipt(image)
        if not result['success']:raise ValueError(result['error'])
        corners=result['corners']
        ocr_corners=make_ocr_safe_corners(corners,image.shape,result['top_correction'])
        warped=warp_receipt(image,ocr_corners)
        variants=make_ocr_variants(warped)
        if method not in variants:raise ValueError('지원하지 않는 OCR 전처리: '+method)
        return {'success':True,'error':None,'corners':corners.tolist(),
                'ocr_corners':ocr_corners.tolist(),
                'detection_score':result['score'],'detection_source':result['source'],
                'candidate_count':result['candidate_count'],'top_candidates':result['top_candidates'],
                'warped':warped,'ocr_image':variants[method],'variants':variants,
                'detection_overlay':draw_detection(image,corners),'method':method,
                'top_correction':result['top_correction']}
    except (ValueError,TypeError,OSError,cv2.error) as exc:
        return {'success':False,'error':str(exc),'corners':None,'detection_score':None,
                'detection_source':None,'ocr_corners':None,'candidate_count':0,'top_candidates':[],
                'warped':None,'ocr_image':None,'variants':{},'detection_overlay':None,
                'method':method,'top_correction':None}
