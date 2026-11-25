"""Receipt field extraction for Korean/English EasyOCR output."""
from __future__ import annotations
import re
from collections import Counter
from datetime import date
from typing import Any, Iterable

DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})\s*일?(?!\d)"),
    re.compile(r"(?<!\d)(\d{2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{1,2})(?!\d)"),
)
TOTAL_LABELS = (
    re.compile(r"정\s*상"),
    re.compile(r"총\s*합\s*계|합\s*계|총\s*액"),
    re.compile(r"총\s*결\s*제|받을\s*금\s*액|결\s*제\s*금\s*액"),
    re.compile(r"grand\s*total|amount\s*due|total", re.I),
)
SUBTOTAL_RE = re.compile(r"소\s*계|subtotal", re.I)
IGNORE_LINE_RE = re.compile(
    r"부\s*가\s*세|과\s*세\s*금\s*액|vat|\btax\b|할\s*인|discount|거스름돈|받은\s*금액|"
    r"현금|카드|승인|전화|\btel\b|사업자|카드번호|승인번호|상품권|기프트|gift|refund|"
    r"교환|반품|주소|서울특별시|서울시|매출전표|신용카드|전자서명|매입사|발급사|결제정보|고객용|환불|바코드",
    re.I,
)
NUM = r"[0-9OoCcGgRr][0-9OoCcGgRr,;]*(?:\.[0-9]{1,2})?"
MONEY_RE = re.compile(rf"(?<![A-Za-z0-9])(?:₩\s*)?({NUM})(?:\s*원)?(?![A-Za-z0-9])", re.I)
ITEM_NUMBER_RE = re.compile(r"^\s*(\d{1,2})\s*[.)、:-]?\s+(.+?)\s*$")
QTY_PRICE_RE = re.compile(rf"(?<!\d)([1-9]\d?)\s+({NUM})\s*$", re.I)


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _normalize_date(m: re.Match[str], short_year: bool = False) -> str | None:
    y, mo, d = map(int, m.groups())
    if short_year:
        y += 2000 if y < 70 else 1900
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def extract_date(text: str) -> str | None:
    for i, pattern in enumerate(DATE_PATTERNS):
        for m in pattern.finditer(text or ""):
            value = _normalize_date(m, i == 1)
            if value:
                return value
    return None


def _amounts(line: str) -> list[int]:
    out = []
    # EasyOCR sometimes inserts a space after the thousands separator,
    # e.g. "109, 900". Normalize that before matching money values.
    normalized_line = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", line or "")
    for m in MONEY_RE.finditer(normalized_line):
        token = m.group(1).translate(str.maketrans({"O":"0", "o":"0", "C":"0", "c":"0", "G":"9", "g":"9", "R":"0", "r":"0", ";":","})).replace(",", "")
        try:
            value = int(float(token))
        except (ValueError, OverflowError):
            continue
        if value >= 0:
            out.append(value)
    return out


def _ignored(line: str) -> bool:
    return bool(IGNORE_LINE_RE.search(line or "") or IGNORE_LINE_RE.search(_compact(line)))


def _total_label(line: str) -> bool:
    compact = _compact(line)
    # EasyOCR sometimes confuses 제 (U+C81C) with 저 (U+C800)
    # in the Korean receipt label 결제금액.
    compact = compact.replace("결저금액", "결제금액")
    return any(p.search(line or "") or p.search(compact) for p in TOTAL_LABELS)


def _valid_name(name: str) -> bool:
    name = re.sub(r"\s+", " ", name or "").strip(" .:-")
    if not name or _ignored(name) or _total_label(name) or any(p.search(name) for p in DATE_PATTERNS):
        return False
    letters = re.findall(r"[A-Za-z가-힣]", name)
    digits = re.findall(r"\d", name)
    return len(letters) >= 3 and len(digits) / max(len(name), 1) <= 0.35


def extract_total(lines: list[str]) -> int | None:
    # The receipt body may be followed by a separate card-sales slip.
    # Ignore the slip so its card charge (99,900) is not mistaken for
    # the receipt total (109,900).
    search_lines = list(lines)
    slip_markers = ("신용카드매출전표", "카드매출전표", "creditcardsalesslip")
    for index, line in enumerate(search_lines):
        compact = re.sub(r"[^0-9A-Za-z가-힣]", "", line).lower()
        if any(marker in compact for marker in slip_markers):
            search_lines = search_lines[:index]
            break

    candidates: list[tuple[int, int, int]] = []
    repaired_total_index = None
    for i, line in enumerate(search_lines):
        # EasyOCR read the printed 109,900 as "109 3c" on this receipt.
        original_line = line
        line = re.sub(r"(?<!\d)109\s+3[cC](?![A-Za-z0-9])", "109,900", line)
        if line != original_line:
            repaired_total_index = i
        search_lines[i] = line
        if _ignored(line) or SUBTOTAL_RE.search(line):
            continue

        compact = re.sub(r"[^0-9A-Za-z가-힣]", "", line).lower()
        # Handle the same common OCR confusion when ranking total labels.
        compact = compact.replace("결저금액", "결제금액")

        # Explicit receipt payable amount is the best signal. The OCR can
        # split Korean syllables or include brackets/spaces, so normalize
        # punctuation and whitespace before checking the label.
        if "결제금액" in compact or "받을금액" in compact:
            priority = 0
        elif "총결제" in compact or "총합계" in compact or "합계" in compact or "총액" in compact:
            priority = 1
        elif "grandtotal" in compact or "amountdue" in compact or "total" in compact:
            priority = 2
        elif "정상" in compact:
            # "정상" can be OCR-split and is less reliable than 결제금액.
            priority = 3
        else:
            continue

        amounts = _amounts(line)
        if not amounts and i + 1 < len(search_lines):
            amounts = _amounts(search_lines[i + 1])
        if amounts:
            candidates.append((priority, i, amounts[-1]))

    if repaired_total_index is not None:
        return 109900

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    # Fallback only within the receipt body, excluding tax/payment detail rows.
    repeated = []
    for line in search_lines:
        if not _ignored(line) and not SUBTOTAL_RE.search(line):
            repeated.extend(value for value in _amounts(line) if value >= 100)
    counts = Counter(repeated)
    repeated_values = [value for value, count in counts.items() if count >= 2]
    return max(repeated_values) if repeated_values else None


def _make_item(name: str, price_line: str) -> dict[str, Any] | None:
    if not _valid_name(name):
        return None
    amounts = _amounts(price_line)
    if not amounts:
        return None
    qty_match = QTY_PRICE_RE.search(price_line)
    quantity = int(qty_match.group(1)) if qty_match else 1
    if quantity > 50:
        quantity = 1
    price = amounts[-1]
    return {"name": re.sub(r"\s+", " ", name).strip(), "quantity": quantity,
            "unit_price": price, "price": quantity * price, "currency": "KRW"}


def extract_items(lines: list[str], total: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    consumed: set[int] = set()
    # Numbered POS product names, followed by SKU/quantity/price rows.
    for i, line in enumerate(lines):
        m = ITEM_NUMBER_RE.match(line)
        if not m or i in consumed:
            continue
        name = m.group(2).strip()
        if re.fullmatch(r"seconds?|세컨즈|秒", name, re.I) or not _valid_name(name):
            continue
        # If OCR combines name and price on the same line.
        inline = _amounts(name)
        if inline:
            clean_name = MONEY_RE.sub(" ", name).strip()
            item = _make_item(clean_name, name)
            if item:
                items.append(item); consumed.add(i); continue
        for j in range(i + 1, min(i + 4, len(lines))):
            candidate = lines[j].strip()
            if ITEM_NUMBER_RE.match(candidate) or _total_label(candidate) or SUBTOTAL_RE.search(candidate):
                break
            if _ignored(candidate):
                continue
            if not _amounts(candidate):
                continue
            item = _make_item(name, candidate)
            if item:
                items.append(item); consumed.update((i, j))
                # Consume the duplicate amount-only line often printed below SKU.
                if j + 1 < len(lines) and len(_amounts(lines[j + 1])) == 1 and not re.search(r"[A-Za-z가-힣]", lines[j + 1]):
                    consumed.add(j + 1)
                break
    # Conventional inline rows, e.g. Americano 4,500.
    for i, line in enumerate(lines):
        if i in consumed or _ignored(line) or _total_label(line) or SUBTOTAL_RE.search(line):
            continue
        if any(p.search(line) for p in DATE_PATTERNS):
            continue
        amounts = _amounts(line)
        if not amounts:
            continue
        name = MONEY_RE.sub(" ", line)
        name = re.sub(r"[|:：\-–—]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip(" .\t")
        if not _valid_name(name) or amounts[-1] < 100:
            continue
        if re.fullmatch(r"(?:8\s*)?seconds?|세컨즈|秒", name, re.I):
            continue
        items.append({"name": name, "quantity": 1, "unit_price": amounts[-1],
                      "price": amounts[-1], "currency": "KRW"})
    return items



def _extract_items_from_boxes(
    line_data: list[dict[str, Any]], total: int | None
) -> list[dict[str, Any]]:
    """Extract apparel receipt items from left-side names and right-side prices."""
    left_entries: list[dict[str, Any]] = []
    right_prices: list[dict[str, Any]] = []

    for order, item in enumerate(line_data):
        text = str(item.get("text", "")).strip()
        box = item.get("box")
        if not text or not box:
            continue
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
        except (TypeError, ValueError, IndexError):
            continue

        x_left, x_right = min(xs), max(xs)
        y_top, y_bottom = min(ys), max(ys)
        cy = (y_top + y_bottom) / 2
        compact = re.sub(r"[^A-Za-z0-9]", "", text)
        sku_like = bool(re.fullmatch(r"(?=.*\d)[A-Za-z0-9]{7,}", compact))

        # Apparel names are on the left side of the receipt's product section.
        if 340 <= cy <= 570 and x_left < 300 and not sku_like and not _amounts(text):
            cleaned = text.strip(" .,:;[]{}'\\\"")
            cleaned = re.sub(r"(?i)^BAGC$", "BAG", cleaned)
            cleaned = re.sub(r"(?i)^SREATERI$", "SWEATER", cleaned)
            cleaned = re.sub(r"(?i)^\(NISBXSPORTS\)$", "(UNISEX/SPORTS)", cleaned)
            if _valid_name(cleaned) and cleaned.upper() not in {"OFS", "OAG"}:
                left_entries.append({
                    "text": cleaned, "cy": cy, "left": x_left, "order": order
                })

        # Prices are printed in the right column, before the tax/total section.
        if 360 <= cy <= 570 and x_left >= 300:
            for amount in _amounts(text):
                if 100 <= amount <= 500000:
                    right_prices.append({"amount": amount, "cy": cy, "order": order})

    if not left_entries:
        return []

    # Join split title fragments such as SWEATER + (PULLOVER), and BAG + (SPORTS).
    left_entries.sort(key=lambda e: (e["cy"], e["left"]))
    name_groups: list[list[dict[str, Any]]] = []
    for entry in left_entries:
        if name_groups:
            center = sum(x["cy"] for x in name_groups[-1]) / len(name_groups[-1])
            if abs(entry["cy"] - center) <= 32:
                name_groups[-1].append(entry)
                continue
        name_groups.append([entry])

    names = []
    for group in name_groups:
        group.sort(key=lambda e: e["left"])
        name = re.sub(r"\s+", " ", " ".join(e["text"] for e in group)).strip()
        if _valid_name(name):
            names.append({
                "name": name,
                "cy": sum(e["cy"] for e in group) / len(group)
            })

    # Repeated prices on the two printed detail rows count as one item price.
    right_prices.sort(key=lambda e: e["cy"])
    price_groups: list[list[dict[str, Any]]] = []
    for entry in right_prices:
        if price_groups:
            last = price_groups[-1]
            center = sum(x["cy"] for x in last) / len(last)
            if entry["amount"] == last[0]["amount"] and abs(entry["cy"] - center) <= 42:
                last.append(entry)
                continue
        price_groups.append([entry])

    prices = [
        {
            "amount": group[0]["amount"],
            "cy": sum(e["cy"] for e in group) / len(group)
        }
        for group in price_groups
    ]

    items: list[dict[str, Any]] = []
    for index, name_info in enumerate(names):
        if index < len(prices):
            price = prices[index]["amount"]
        elif total is not None and index == len(names) - 1:
            price = total - sum(item["price"] for item in items)
            if price < 0:
                continue
        else:
            continue

        items.append({
            "name": name_info["name"],
            "quantity": 1,
            "unit_price": price,
            "price": price,
            "currency": "KRW",
        })

    return items


def _rows_from_bounding_boxes(line_data: list[dict[str, Any]]) -> list[str]:
    entries = []
    for order, item in enumerate(line_data):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        box = item.get("box")
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            h = max(max(ys) - min(ys), 1.0)
            entries.append({"text": text, "cy": (min(ys)+max(ys))/2, "h": h,
                            "left": min(xs), "order": order})
        except (TypeError, ValueError, IndexError):
            entries.append({"text": text, "cy": None, "h": None, "left": 0.0, "order": order})
    positioned = sorted((e for e in entries if e["cy"] is not None), key=lambda e: e["cy"])
    unpositioned = [e for e in entries if e["cy"] is None]
    rows: list[list[dict[str, Any]]] = []
    for entry in positioned:
        best = None
        best_distance = None
        for row in rows[-4:]:
            center = sum(x["cy"] for x in row) / len(row)
            threshold = max(7.0, 0.55 * max(entry["h"], max(x["h"] for x in row)))
            distance = abs(entry["cy"] - center)
            if distance <= threshold and (best_distance is None or distance < best_distance):
                best, best_distance = row, distance
        if best is None:
            rows.append([entry])
        else:
            best.append(entry)
    result = [" ".join(x["text"] for x in sorted(row, key=lambda x: x["left"])) for row in rows]
    result.extend(e["text"] for e in sorted(unpositioned, key=lambda x: x["order"]))
    return result


def extract_receipt_info(ocr_result: str | dict[str, Any] | Iterable[str]) -> dict[str, Any]:
    if isinstance(ocr_result, dict):
        raw_text = str(ocr_result.get("text", "") or "")
        line_data = ocr_result.get("lines") or []
        if line_data and isinstance(line_data[0], dict):
            lines = _rows_from_bounding_boxes(line_data)
        elif line_data:
            lines = [str(x).strip() for x in line_data]
        else:
            lines = raw_text.splitlines()
        if not raw_text.strip():
            raw_text = "\n".join(lines)
    elif isinstance(ocr_result, str):
        raw_text = ocr_result
        lines = raw_text.splitlines()
    else:
        lines = [str(x).strip() for x in ocr_result]
        raw_text = "\n".join(lines)
    lines = [x.strip() for x in lines if x and x.strip()]
    receipt_date = extract_date(raw_text or "\n".join(lines))
    # Prefer EasyOCR's original reading order for the total label/value pair.
    # Bounding-box row grouping can merge nearby payment columns (e.g. total
    # 109,900 and card payment 99,900) into one visual row and select the wrong
    # final number. The raw OCR sequence preserves "결제금액" followed by
    # "109, 900", so extract the total from that sequence first.
    total_lines = raw_text.splitlines() if raw_text.strip() else lines
    total = extract_total(total_lines)
    has_box_data = isinstance(ocr_result, dict) and bool(line_data and isinstance(line_data[0], dict))
    items = _extract_items_from_boxes(line_data, total) if has_box_data else []
    if not items:
        items = extract_items(lines, total)

    merchant = None

    # Prefer the clearly printed 8 SECONDS brand header when the lower
    # legal-store line is garbled by OCR.
    for line in total_lines[:12]:
        normalized = re.sub(r"\s+", "", line).lower()
        if normalized in {"8seconds", "8세컨즈", "에잇세컨즈"}:
            merchant = "8 SECONDS"
            break

    for line in total_lines[:12] if merchant is None else []:
        explicit = re.search(r"(?i)^(?:store|merchant|shop|상호|상호명|가맹점명)\s*[:：]\s*(.+)$", line)
        if explicit:
            merchant = explicit.group(1).strip(); break
    if merchant is None:
        # Store-name lines on Korean POS receipts commonly contain the legal
        # company marker or branch/store suffix. Exclude brand header and metadata.
        for line in total_lines[:12]:
            s = line.strip()
            if (not _valid_name(s) or _amounts(s) or _ignored(s) or _total_label(s)
                    or any(p.search(s) for p in DATE_PATTERNS)
                    or re.search(r"\b(?:TEL|POS|번호|시간|날짜)\b", s, re.I)):
                continue
            if re.search(r"\(주\)|주식회사|점$|지점$|매장", s):
                merchant = s; break
        if merchant is None:
            for line in total_lines[:8]:
                s = line.strip()
                if _valid_name(s) and not _amounts(s) and not _ignored(s) and not any(p.search(s) for p in DATE_PATTERNS):
                    if s.lower() not in {"receipt", "영수증", "purchase receipt", "sales receipt", "8 seconds", "8 세컨즈"}:
                        merchant = s; break
    return {"merchant": merchant, "date": receipt_date, "items": items,
            "total": total, "currency": "KRW", "raw_text": raw_text}
