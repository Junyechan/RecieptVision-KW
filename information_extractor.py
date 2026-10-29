"""Extract common fields from OCR text on Korean receipts."""
from __future__ import annotations

import re
from typing import Any, Iterable


DATE_PATTERNS = (
    re.compile(
        r"(?<!\d)(20\d{2})[./년-]\s*(\d{1,2})[./월-]\s*(\d{1,2})일?(?!\d)"
    ),
    re.compile(r"(?<!\d)(\d{2})[./-]\s*(\d{1,2})[./-]\s*(\d{1,2})(?!\d)"),
)

TOTAL_WORDS = (
    "합계", "총액", "총합계", "결제금액", "받을금액",
    "총 결제", "total", "amount due", "grand total",
)
SUBTOTAL_WORDS = ("소계", "subtotal")
IGNORE_AMOUNT_WORDS = (
    "부가세", "vat", "tax", "할인", "discount", "거스름돈",
    "받은금액", "현금", "카드", "승인", "전화", "사업자", "번호",
)

# Amounts such as 1,500, 1500원, or ₩1500.
MONEY_RE = re.compile(
    r"(?<![\w])(?:₩\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)(?:\s*원)?"
)


def _normalize_date(
    match: re.Match,
    two_digit_year: bool = False,
) -> str:
    year, month, day = [int(part) for part in match.groups()]

    if two_digit_year:
        year += 2000 if year < 70 else 1900

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""

    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_date(text: str) -> str | None:
    for index, pattern in enumerate(DATE_PATTERNS):
        for match in pattern.finditer(text):
            value = _normalize_date(
                match,
                two_digit_year=(index == 1 and len(match.group(1)) == 2),
            )
            if value:
                return value

    return None


def _amounts(line: str) -> list[int]:
    values = []

    for match in MONEY_RE.finditer(line):
        raw = match.group(1).replace(",", "")

        try:
            value = int(float(raw))
        except ValueError:
            continue

        if value >= 0:
            values.append(value)

    return values


def extract_total(lines: list[str]) -> int | None:
    # Prefer amounts appearing on a line labelled as the final total.
    for index, line in enumerate(lines):
        lower = line.lower()

        if (
            any(word in lower for word in TOTAL_WORDS)
            and not any(word in lower for word in SUBTOTAL_WORDS)
        ):
            amounts = _amounts(line)

            if amounts:
                return amounts[-1]

            # Some OCR engines separate the total label and amount.
            if index + 1 < len(lines):
                next_amounts = _amounts(lines[index + 1])
                if next_amounts:
                    return next_amounts[-1]

    return None


def extract_items(
    lines: list[str],
    total: int | None = None,
) -> list[dict[str, Any]]:
    """Extract item names and prices, including names on the previous line."""
    items = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue

        if any(word in lower for word in TOTAL_WORDS + SUBTOTAL_WORDS):
            continue

        if any(word in lower for word in IGNORE_AMOUNT_WORDS):
            continue

        if any(pattern.search(stripped) for pattern in DATE_PATTERNS):
            continue

        # Match quantity and unit price, e.g. 2x1500, 2 x 1500, or 2×1,500.
        quantity_match = re.search(
            r"(?<!\d)(\d+)\s*[xX×]\s*([0-9][0-9,]*)(?!\d)",
            stripped,
        )

        if quantity_match:
            quantity = int(quantity_match.group(1))
            unit_price = int(quantity_match.group(2).replace(",", ""))

            # Remove the quantity-price expression to obtain any inline name.
            name = (
                stripped[:quantity_match.start()]
                + " "
                + stripped[quantity_match.end():]
            )
            name = re.sub(r"\s+", " ", name).strip(" .:-")

            # OCR may put the item name on the previous line.
            if not name and index > 0:
                previous = lines[index - 1].strip()

                if (
                    previous
                    and not _amounts(previous)
                    and not any(
                        word in previous.lower()
                        for word in TOTAL_WORDS + SUBTOTAL_WORDS
                    )
                    and not any(
                        pattern.search(previous)
                        for pattern in DATE_PATTERNS
                    )
                ):
                    name = previous

            if name and re.search(r"[A-Za-z가-힣]", name):
                items.append({
                    "name": name,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "price": quantity * unit_price,
                    "currency": "KRW",
                })

            continue

        # Handle a simple item line such as "Apple 1500".
        amounts = _amounts(stripped)

        if not amounts:
            continue

        price = amounts[-1]

        if total is not None and price == total and len(amounts) == 1:
            continue

        name = MONEY_RE.sub(" ", stripped)
        name = re.sub(r"[|:：\-–—]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip(" .\t")

        if len(name) < 2 or not re.search(r"[A-Za-z가-힣]", name):
            continue

        items.append({
            "name": name,
            "quantity": 1,
            "unit_price": price,
            "price": price,
            "currency": "KRW",
        })

    return items


def extract_receipt_info(
    ocr_result: str | dict[str, Any] | Iterable[str],
) -> dict[str, Any]:
    """Convert OCR output into a structured receipt dictionary."""
    if isinstance(ocr_result, dict):
        raw_text = str(ocr_result.get("text", ""))
        line_data = ocr_result.get("lines", [])

        if line_data and isinstance(line_data[0], dict):
            lines = [
                str(item.get("text", "")).strip()
                for item in line_data
            ]
        elif line_data:
            lines = [str(item).strip() for item in line_data]
        else:
            lines = raw_text.splitlines()

    elif isinstance(ocr_result, str):
        raw_text = ocr_result
        lines = raw_text.splitlines()

    else:
        lines = [str(item).strip() for item in ocr_result]
        raw_text = "\n".join(lines)

    lines = [line.strip() for line in lines if line and line.strip()]

    date = extract_date(raw_text)
    total = extract_total(lines)
    items = extract_items(lines, total=total)

    # Prefer a clearly labelled store/merchant line.
    merchant = None

    for line in lines[:10]:
        match = re.search(
            r"(?i)^(?:store|merchant|shop|상호|상호명|가맹점명)"
            r"\s*[:：]\s*(.+)$",
            line.strip(),
        )

        if match:
            merchant = match.group(1).strip()
            break

    # Otherwise, skip generic receipt headings and select a plausible line.
    if merchant is None:
        generic_headings = {
            "receipt",
            "영수증",
            "purchase receipt",
            "sales receipt",
        }

        for line in lines[:8]:
            stripped = line.strip()

            if stripped.lower() in generic_headings:
                continue

            if any(pattern.search(stripped) for pattern in DATE_PATTERNS):
                continue

            if _amounts(stripped):
                continue

            if any(
                word in stripped.lower()
                for word in TOTAL_WORDS + SUBTOTAL_WORDS
            ):
                continue

            if re.search(r"[A-Za-z가-힣]", stripped) and len(stripped) >= 2:
                merchant = stripped
                break

    return {
        "merchant": merchant,
        "date": date,
        "items": items,
        "total": total,
        "currency": "KRW",
        "raw_text": raw_text,
    }