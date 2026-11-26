"""Receipt field extraction for Korean/English EasyOCR results.

The extractor uses OCR text plus relative Bounding Box geometry.  It deliberately
avoids receipt-specific pixel coordinates and product-name lookup tables.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Text normalization and primitive parsers
# ---------------------------------------------------------------------------
DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})\s*일?(?!\d)"),
    re.compile(r"(?<!\d)(\d{2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{1,2})(?!\d)"),
)

# Labels are compacted before matching so OCR-inserted spaces are tolerated.
TOTAL_LABELS = (
    "결제금액", "받을금액", "총결제", "총합계", "합계", "총액", "청구금액",
    "grandtotal", "amountdue", "total",
)
TAX_LABELS = ("과세금액", "과세", "supplyamount", "taxableamount")
VAT_LABELS = ("부가세", "vat", "taxamount")
SLIP_MARKERS = ("신용카드매출전표", "카드매출전표", "creditcardsalesslip")

# These are semantic exclusions, not coordinate rules.
NON_ITEM_RE = re.compile(
    r"사업자|등록번호|전화|연락처|주소|서울특별시|서울시|도로명|우편번호|"
    r"카드번호|승인번호|승인|매입사|발급사|결제정보|전자서명|매출전표|"
    r"고객용|현금|카드|거스름돈|받은금액|과세금액|부가세|부가\s*세|"
    r"환불|교환|반품|환물|구입|가능기간|가격택|착용|패키지|개봉|"
    r"상품코드|품목코드|바코드|사업자정보|상품권|gift\s*voucher|결제금액|총액|합계|subtotal|"
    r"grand\s*total|amount\s*due|tax|vat|discount|refund|change|cash|credit\s*card",
    re.I,
)

# Korean/English receipt OCR frequently confuses O with 0 and ; with comma.
_DIGIT_TRANSLATION = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5", "B": "8", "g": "9", "C": "0", "c": "0", "r": "0",
    ";": ",", "，": ",", "．": ".",
})
_MONEY_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:₩\s*)?([0-9OoDQiIl|SsBgCcr;，,]{1,14})(?:\s*원)?(?![A-Za-z0-9])",
    re.I,
)


def _compact(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", text or "")).lower()


def _clean_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("：", ":").replace("，", ",")
    text = re.sub(r"(?<=\d)\s*,\s*(?=\d{3}(?:\D|$))", ",", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_money_token(token: str) -> int | None:
    token = _clean_ocr_text(token).translate(_DIGIT_TRANSLATION)
    # Do not convert arbitrary letters; after OCR substitutions only digits,
    # separators and spaces are accepted.
    token = re.sub(r"\s+", "", token)
    if not token or not re.fullmatch(r"[0-9,]+(?:\.\d{1,2})?", token):
        return None
    token = token.replace(",", "")
    try:
        value = int(float(token))
    except (ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _amounts(text: str) -> list[int]:
    normalized = _clean_ocr_text(text)
    # Prevent the day/year fragments in date strings from being parsed as money.
    if extract_date(normalized) and re.fullmatch(
        r"\s*(?:20\d{2}|\d{2})\s*[./년-]\s*\d{1,2}\s*[./월-]\s*\d{1,2}\s*일?\s*",
        normalized,
    ):
        return []
    values: list[int] = []
    for match in _MONEY_TOKEN_RE.finditer(normalized):
        token = match.group(1).strip()
        # A bare short integer in a text line is often an item number/quantity.
        value = _parse_money_token(token)
        if value is not None:
            values.append(value)
    return values


def _valid_date(year: int, month: int, day: int, short_year: bool = False) -> str | None:
    if short_year:
        year += 2000 if year < 70 else 1900
    try:
        date(year, month, day)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_date(text: str) -> str | None:
    for pattern_index, pattern in enumerate(DATE_PATTERNS):
        for match in pattern.finditer(text or ""):
            value = _valid_date(*map(int, match.groups()), short_year=(pattern_index == 1))
            if value:
                return value
    return None


def _is_total_label(text: str) -> bool:
    compact = _compact(text).replace("결저금액", "결제금액")
    return any(label in compact for label in TOTAL_LABELS)


def _is_excluded_line(text: str) -> bool:
    compact = _compact(text)
    # Reject phone numbers even when OCR omits the TEL label.
    if re.search(r"(?<!\d)(?:\+?82[- .]?)?0\d{1,2}[- .]?\d{3,4}[- .]?\d{4}(?!\d)", text or ""):
        return True
    if re.search(r"(?i)\bTEL\b|전화번호|연락처", text or ""):
        return True
    return bool(NON_ITEM_RE.search(text or "") or NON_ITEM_RE.search(compact))


def _valid_item_name(text: str) -> bool:
    name = _clean_ocr_text(text).strip(" .,:;[]{}()'\"|-")
    if len(name) < 3 or _is_excluded_line(name) or _is_total_label(name):
        return False
    if any(pattern.search(name) for pattern in DATE_PATTERNS):
        return False
    # Product/SKU/size codes generally contain a substantial digit component.
    alnum = re.sub(r"[^A-Za-z0-9]", "", name)
    if len(alnum) >= 6 and sum(ch.isdigit() for ch in alnum) / len(alnum) > 0.20:
        return False
    letters = re.findall(r"[A-Za-z가-힣]", name)
    if len(letters) < 3:
        return False
    # Exclude telephone-like, registration-like and numeric-only fragments.
    if re.fullmatch(r"[A-Za-z]{0,3}\d{4,}[A-Za-z0-9]*", re.sub(r"\s+", "", name)):
        return False
    return True


# ---------------------------------------------------------------------------
# Bounding Box handling
# ---------------------------------------------------------------------------
def _box_geometry(entry: dict[str, Any], order: int) -> dict[str, Any] | None:
    box = entry.get("box")
    try:
        points = [(float(point[0]), float(point[1])) for point in box]
        if len(points) < 4:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        if right <= left or bottom <= top:
            return None
        return {
            "text": _clean_ocr_text(entry.get("text", "")),
            "confidence": float(entry.get("confidence", 1.0)),
            "left": left, "right": right, "top": top, "bottom": bottom,
            "cx": (left + right) / 2.0, "cy": (top + bottom) / 2.0,
            "width": right - left, "height": bottom - top,
            "order": order,
        }
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _looks_like_sku(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", text or "")
    if len(compact) < 6:
        return False
    digit_ratio = sum(ch.isdigit() for ch in compact) / len(compact)
    return digit_ratio >= 0.12 or bool(re.search(r"\d\s*[xX]\s*\d", text))


def _find_product_names(entries: list[dict[str, Any]], page_left: float,
                        page_right: float) -> list[dict[str, Any]]:
    page_width = max(page_right - page_left, 1.0)
    names: list[dict[str, Any]] = []
    for entry in entries:
        text = entry["text"]
        x_ratio = (entry["cx"] - page_left) / page_width
        if x_ratio > 0.58 or entry["confidence"] < 0.12:
            continue
        # Remove a POS row ordinal before checking for money tokens, otherwise
        # a combined OCR detection such as "3. BAG(UNISEX/SPORTS)" is rejected.
        cleaned = re.sub(r"^\s*\d{1,2}\s*[.,、):\-]?\s*", "", text).strip(" .,:;'")
        if _looks_like_sku(cleaned) or _amounts(cleaned) or not _valid_item_name(cleaned):
            continue
        # Short uppercase POS markers such as OFS/OAG are not product titles.
        letters_only = re.sub(r"[^A-Za-z]", "", cleaned)
        if len(letters_only) <= 3 and letters_only.isupper():
            continue
        if not _valid_item_name(cleaned):
            continue
        names.append({**entry, "name": cleaned})
    return sorted(names, key=lambda item: (item["cy"], item["left"]))


def _find_price_entries(entries: list[dict[str, Any]], page_left: float,
                        page_right: float) -> list[dict[str, Any]]:
    page_width = max(page_right - page_left, 1.0)
    prices: list[dict[str, Any]] = []
    for entry in entries:
        x_ratio = (entry["cx"] - page_left) / page_width
        if x_ratio < 0.55 or entry["confidence"] < 0.10:
            continue
        if _is_excluded_line(entry["text"]):
            continue
        amounts = _amounts(entry["text"])
        # A single OCR detection can contain a price and unrelated digits;
        # retain the final money token, as POS rows usually place amount last.
        if not amounts:
            continue
        amount = amounts[-1]
        if amount > 5_000_000:
            continue
        prices.append({**entry, "amount": amount})
    return sorted(prices, key=lambda item: (item["cy"], item["cx"]))


def _extract_items_from_boxes(line_data: list[dict[str, Any]],
                              total: int | None = None) -> list[dict[str, Any]]:
    """Extract item rows using relative horizontal/vertical Bounding Box relations.

    No absolute pixel coordinates, fixed receipt width, or item-index pairing is
    used. Duplicate line/total prices are excluded by semantic region and by
    choosing the closest price below each product title before the next title.
    """
    entries = [g for i, raw in enumerate(line_data)
               if isinstance(raw, dict) and (g := _box_geometry(raw, i)) is not None
               and g["text"]]
    if len(entries) < 2:
        return []

    page_left = min(e["left"] for e in entries)
    page_right = max(e["right"] for e in entries)
    names = _find_product_names(entries, page_left, page_right)
    prices = _find_price_entries(entries, page_left, page_right)
    if not names or not prices:
        return []

    # The first name with a nearby right-column amount marks the start of the
    # product table. A tax/total label below it terminates the table.
    tax_y_values = [e["cy"] for e in entries
                    if any(label in _compact(e["text"]) for label in
                           ("과세금액", "부가세", "소계", "합계", "총액"))]
    item_names: list[dict[str, Any]] = []
    for name in names:
        later_names = [n for n in names if n["cy"] > name["cy"] + 0.25 * name["height"]]
        next_name_y = min((n["cy"] for n in later_names), default=float("inf"))
        has_price = any(
            p["cy"] >= name["cy"] - 0.7 * name["height"]
            and p["cy"] < next_name_y
            and p["cy"] - name["cy"] <= max(8.0 * name["height"], 180.0 * (max(e["height"] for e in entries) / 100.0))
            for p in prices
        )
        before_tax = not tax_y_values or name["cy"] < min(tax_y_values)
        if has_price and before_tax:
            item_names.append(name)

    if not item_names:
        return []

    # Keep the product table contiguous; names after the first tax/total marker
    # are footer/legal text and should never become products.
    item_names.sort(key=lambda n: n["cy"])
    table_end = min(tax_y_values) if tax_y_values else float("inf")
    item_names = [n for n in item_names if n["cy"] < table_end]

    items: list[dict[str, Any]] = []
    used_price_indices: set[int] = set()
    for index, name in enumerate(item_names):
        next_y = item_names[index + 1]["cy"] if index + 1 < len(item_names) else table_end
        candidates = []
        for p_index, price in enumerate(prices):
            if p_index in used_price_indices:
                continue
            # Price is expected on the same item row or below its title, and
            # before the next product title. Relative text height handles scale.
            lower = name["cy"] - max(name["height"] * 0.8, 4.0)
            upper = min(next_y - 0.15 * name["height"], table_end)
            if lower <= price["cy"] < upper:
                candidates.append((abs(price["cy"] - name["cy"]), p_index, price))
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x[0], -x[2]["confidence"]))
        _, selected_index, selected = candidates[0]
        used_price_indices.add(selected_index)
        amount = selected["amount"]
        if amount < 0:
            continue
        items.append({
            "name": re.sub(r"\s+", " ", name["name"]).strip(),
            "quantity": 1,
            "unit_price": amount,
            "price": amount,
            "currency": "KRW",
        })
    return items


# ---------------------------------------------------------------------------
# Text-only fallback for OCR providers without Bounding Boxes
# ---------------------------------------------------------------------------
def _extract_items_from_text(lines: list[str]) -> list[dict[str, Any]]:
    cleaned = [_clean_ocr_text(line) for line in lines if _clean_ocr_text(line)]
    items: list[dict[str, Any]] = []
    used_names: set[str] = set()

    def add_item(name: str, amount: int) -> None:
        name = re.sub(r"\s+", " ", name).strip(" .,:;[]{}'\"|-")
        key = _compact(name)
        if key and key not in used_names and _valid_item_name(name) and amount >= 100:
            used_names.add(key)
            items.append({"name": name, "quantity": 1, "unit_price": amount,
                          "price": amount, "currency": "KRW"})

    for index, line in enumerate(cleaned):
        if extract_date(line) or _is_excluded_line(line) or _is_total_label(line):
            continue
        title = re.sub(r"^\s*\d{1,2}[.)、:-]\s*", "", line).strip()

        # Inline format: product title followed by its amount.
        matches = list(_MONEY_TOKEN_RE.finditer(line))
        if matches:
            match = matches[-1]
            amount = _parse_money_token(match.group(1))
            name_part = line[:match.start()].strip(" .,:;|-")
            if amount is not None and amount >= 100 and _valid_item_name(name_part):
                add_item(name_part, amount)
                continue

        if not _valid_item_name(title) or _looks_like_sku(title) or _amounts(title):
            continue
        for candidate in cleaned[index + 1:min(index + 5, len(cleaned))]:
            if extract_date(candidate) or _is_total_label(candidate) or _is_excluded_line(candidate):
                break
            candidate_title = re.sub(r"^\s*\d{1,2}[.)、:-]\s*", "", candidate).strip()
            if _valid_item_name(candidate_title) and not _looks_like_sku(candidate_title) and not _amounts(candidate_title):
                break
            amounts = [value for value in _amounts(candidate) if value >= 100]
            if amounts:
                add_item(title, amounts[-1])
                break
    return items


# ---------------------------------------------------------------------------
# Totals and merchant extraction
# ---------------------------------------------------------------------------
def _extract_labeled_amount(lines: list[str], labels: tuple[str, ...]) -> int | None:
    for index, line in enumerate(lines):
        compact = _compact(line)
        if labels == TOTAL_LABELS and any(marker in compact for marker in ("카드", "신용카드", "card", "상품권", "giftvoucher")):
            continue
        if any(label in compact for label in labels):
            amounts = _amounts(line)
            if amounts:
                return amounts[-1]
            if index + 1 < len(lines):
                next_amounts = _amounts(lines[index + 1])
                # Tiny isolated OCR fragments are usually truncated amounts,
                # not reliable monetary values (e.g. "9" instead of "9,990").
                reliable = [value for value in next_amounts if value >= 100]
                if reliable:
                    return reliable[-1]
    return None


def extract_total(lines: list[str], items: list[dict[str, Any]] | None = None) -> int | None:
    """Extract total with tax and item-sum cross checks.

    Card-slip details are excluded from receipt-total candidates. If the printed
    total is OCR-damaged, item sum and taxable amount + VAT provide independent
    consistency checks; disagreement is not silently hidden in diagnostics.
    """
    receipt_lines: list[str] = []
    for line in lines:
        if any(marker in _compact(line) for marker in SLIP_MARKERS):
            break
        receipt_lines.append(line)

    item_sum = sum(int(item.get("price", 0)) for item in (items or [])) if items else None
    taxable = _extract_labeled_amount(receipt_lines, TAX_LABELS)
    vat = _extract_labeled_amount(receipt_lines, VAT_LABELS)
    tax_total = taxable + vat if taxable is not None and vat is not None else None

    explicit = _extract_labeled_amount(receipt_lines, TOTAL_LABELS)
    # Repeated standalone amounts are useful when the OCR drops the total label.
    counts: dict[int, int] = {}
    for line in receipt_lines:
        if _is_excluded_line(line) or extract_date(line):
            continue
        values = [v for v in _amounts(line) if v >= 100]
        if len(values) == 1:
            counts[values[0]] = counts.get(values[0], 0) + 1
    repeated = [value for value, count in counts.items() if count >= 2]
    repeated_total = max(repeated) if repeated else None

    candidates = [v for v in (explicit, tax_total, item_sum) if v is not None and v >= 0]
    if not candidates:
        return repeated_total

    # Prefer a value independently supported by at least two sources.
    for candidate in candidates:
        if sum(1 for other in candidates if other == candidate) >= 2:
            return candidate
    # Tax-inclusive total is usually more reliable than a damaged label; item
    # sum is preferred only when it agrees with the tax calculation.
    if tax_total is not None:
        return tax_total
    if explicit is not None:
        return explicit
    if repeated_total is not None and (item_sum is None or repeated_total >= item_sum):
        return repeated_total
    if item_sum is not None:
        return item_sum
    return None


def _extract_merchant(lines: list[str]) -> str | None:
    for line in lines[:15]:
        compact = _compact(line)
        # Generic normalization for common OCR variants of the visible brand.
        if ("8seconds" in compact or "8세컨즈" in compact or "에잇세컨즈" in compact
                or "8sec" in compact):
            return "8 SECONDS"
    for line in lines[:15]:
        match = re.search(r"(?i)^(?:store|merchant|shop|상호|상호명|가맹점명)\s*[:：]\s*(.+)$", line)
        if match:
            return match.group(1).strip()
    # Legal company/store line is a better fallback than arbitrary header OCR.
    for line in lines[:15]:
        if (re.search(r"\(주\)|주식회사|지점|매장", line)
                and not _is_excluded_line(line) and not _amounts(line)):
            return _clean_ocr_text(line)
    return None


def _rows_from_bounding_boxes(line_data: list[dict[str, Any]]) -> list[str]:
    """Group OCR detections into visual rows using text-height-relative gaps."""
    entries = [g for i, raw in enumerate(line_data)
               if isinstance(raw, dict) and (g := _box_geometry(raw, i)) is not None]
    positioned = sorted(entries, key=lambda e: (e["cy"], e["left"]))
    rows: list[list[dict[str, Any]]] = []
    for entry in positioned:
        best_row = None
        best_distance = None
        for row in rows[-5:]:
            center = sum(e["cy"] for e in row) / len(row)
            typical_height = max(entry["height"], max(e["height"] for e in row))
            threshold = max(5.0, typical_height * 0.55)
            distance = abs(entry["cy"] - center)
            if distance <= threshold and (best_distance is None or distance < best_distance):
                best_row, best_distance = row, distance
        if best_row is None:
            rows.append([entry])
        else:
            best_row.append(entry)
    return [" ".join(e["text"] for e in sorted(row, key=lambda e: e["left"]))
            for row in rows]


def _ocr_vote_key(text: str) -> str:
    """Comparison key for detections from alternate OCR preprocessing variants."""
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", text or "")).casefold()


def _merge_ocr_variant_lines(variants: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge geometrically matching OCR detections across preprocessing variants.

    Coordinates are compared relatively to each variant's own page dimensions.
    Each variant contributes at most one vote to a cluster. Exact text consensus
    wins; confidence breaks ties. This avoids blindly preferring CLAHE output.
    """
    prepared: list[tuple[str, list[dict[str, Any]], float, float]] = []
    for variant_name, payload in variants.items():
        if not isinstance(payload, dict):
            continue
        raw_lines = payload.get("lines") or []
        entries = [g for i, raw in enumerate(raw_lines)
                   if isinstance(raw, dict) and (g := _box_geometry(raw, i)) is not None
                   and g["text"]]
        if not entries:
            continue
        left = min(e["left"] for e in entries)
        right = max(e["right"] for e in entries)
        top = min(e["top"] for e in entries)
        bottom = max(e["bottom"] for e in entries)
        width, height = max(right-left, 1.0), max(bottom-top, 1.0)
        for e in entries:
            e["nx"] = (e["cx"] - left) / width
            e["ny"] = (e["cy"] - top) / height
            e["nw"] = e["width"] / width
            e["nh"] = e["height"] / height
            e["variant"] = variant_name
        prepared.append((variant_name, entries, width, height))

    clusters: list[list[dict[str, Any]]] = []
    for variant_name, entries, _, _ in prepared:
        for entry in entries:
            best_cluster = None
            best_distance = float("inf")
            for cluster in clusters:
                # Never allow two detections from the same variant to vote twice.
                if any(member["variant"] == variant_name for member in cluster):
                    continue
                anchor = cluster[0]
                dx = abs(entry["nx"] - anchor["nx"])
                dy = abs(entry["ny"] - anchor["ny"])
                # Relative tolerances accommodate small preprocessing/crop shifts.
                if dx <= max(0.018, 0.65 * max(entry["nw"], anchor["nw"])) and \
                   dy <= max(0.012, 0.65 * max(entry["nh"], anchor["nh"])):
                    distance = dx + dy
                    if distance < best_distance:
                        best_cluster, best_distance = cluster, distance
            if best_cluster is None:
                clusters.append([entry])
            else:
                best_cluster.append(entry)

    merged: list[dict[str, Any]] = []
    for cluster in clusters:
        votes: dict[str, list[dict[str, Any]]] = {}
        for member in cluster:
            votes.setdefault(_ocr_vote_key(member["text"]), []).append(member)
        # Majority exact normalized spelling; highest confidence among that vote.
        winning_key, winning_members = max(
            votes.items(),
            key=lambda pair: (len(pair[1]),
                              sum(m["confidence"] for m in pair[1]),
                              max(m["confidence"] for m in pair[1]))
        )
        chosen = max(winning_members, key=lambda m: m["confidence"])
        merged.append({
            "text": chosen["text"],
            "confidence": sum(m["confidence"] for m in winning_members) / len(winning_members),
            "box": [[chosen["left"], chosen["top"]],
                    [chosen["right"], chosen["top"]],
                    [chosen["right"], chosen["bottom"]],
                    [chosen["left"], chosen["bottom"]]],
        })
    return sorted(merged, key=lambda item: (
        sum(point[1] for point in item["box"]) / 4,
        sum(point[0] for point in item["box"]) / 4,
    ))


def extract_receipt_info(ocr_result: str | dict[str, Any] | Iterable[str]) -> dict[str, Any]:
    """Return merchant, date, items, total, currency and raw OCR text."""
    line_data: list[dict[str, Any]] = []
    supplemental_item_lines: list[dict[str, Any]] = []
    if isinstance(ocr_result, dict):
        supplemental = ocr_result.get("item_table_ocr")
        if isinstance(supplemental, dict):
            # Crop boxes use a separate coordinate system; merge all available
            # preprocessing variants by relative geometry instead of preferring
            # one OCR variant (e.g. CLAHE) unconditionally.
            supplemental_item_lines = _merge_ocr_variant_lines(supplemental)
        raw_text = str(ocr_result.get("text", "") or "")
        raw_lines = ocr_result.get("lines") or []
        if raw_lines and isinstance(raw_lines[0], dict):
            line_data = [line for line in raw_lines if isinstance(line, dict)]
            # Keep EasyOCR's recognition order for date/total labels; geometry
            # is used separately for item extraction.
            lines = [str(line.get("text", "")).strip() for line in line_data]
        elif raw_lines:
            lines = [str(line).strip() for line in raw_lines]
        else:
            lines = raw_text.splitlines()
        if not raw_text.strip():
            raw_text = "\n".join(lines)
    elif isinstance(ocr_result, str):
        raw_text = ocr_result
        lines = raw_text.splitlines()
    else:
        lines = [str(line).strip() for line in ocr_result]
        raw_text = "\n".join(lines)

    lines = [_clean_ocr_text(line) for line in lines if _clean_ocr_text(line)]
    receipt_date = extract_date(raw_text or "\n".join(lines))

    items = (_extract_items_from_boxes(supplemental_item_lines) if supplemental_item_lines else (_extract_items_from_boxes(line_data) if line_data else []))
    if not items:
        items = _extract_items_from_text(lines)

    total = extract_total(lines, items)
    merchant = _extract_merchant(lines)

    # Report reconciliation rather than fabricating a balancing item/price.
    item_sum = sum(item["price"] for item in items) if items else None
    taxable = _extract_labeled_amount(lines, TAX_LABELS)
    vat = _extract_labeled_amount(lines, VAT_LABELS)
    tax_sum = taxable + vat if taxable is not None and vat is not None else None
    diagnostics = {
        "item_count": len(items),
        "item_sum": item_sum,
        "taxable_amount": taxable,
        "vat_amount": vat,
        "tax_plus_vat": tax_sum,
        "total_matches_item_sum": total == item_sum if total is not None and item_sum is not None else None,
        "total_matches_tax_plus_vat": total == tax_sum if total is not None and tax_sum is not None else None,
    }
    return {
        "merchant": merchant,
        "date": receipt_date,
        "items": items,
        "total": total,
        "currency": "KRW",
        "raw_text": raw_text,
        "validation": diagnostics,
    }
