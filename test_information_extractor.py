import unittest
from information_extractor import extract_receipt_info


class InformationExtractorTests(unittest.TestCase):
    def test_extracts_date_total_and_items(self):
        text = "STARBUCKS\n2026-10-25\nAmericano 4,500\nLatte 5,000\nTOTAL 9,500 KRW"
        result = extract_receipt_info(text)
        self.assertEqual(result["date"], "2026-10-25")
        self.assertEqual(result["total"], 9500)
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["items"][0]["price"], 4500)

    def test_korean_date_and_total(self):
        text = "카페 별\n2026년 10월 25일\n아메리카노 4,500원\n합계 4,500원"
        result = extract_receipt_info(text)
        self.assertEqual(result["date"], "2026-10-25")
        self.assertEqual(result["total"], 4500)

    def test_apparel_receipt_separate_title_sku_and_price_rows(self):
        text = """8 seconds
2026-10-30
1. PANTS (LONG)
456X21MMW4XL OFS 1 49,900
49,900
2. SWEATER (PULLOVER)
456X51CCAHXL OAG 1 59,900
59,900
3. BAG (UNISEX/SPORTS)
1931D4WN1AF 1 100
100
정 상 109,900
부 가 세 9,990
결제금액 109,900
카드결제 99,900
상품권 10,000
TEL: 070-7799-0095
"""
        result = extract_receipt_info(text)
        self.assertEqual(result["date"], "2026-10-30")
        self.assertEqual(result["total"], 109900)
        self.assertEqual([item["name"] for item in result["items"]], [
            "PANTS (LONG)", "SWEATER (PULLOVER)", "BAG (UNISEX/SPORTS)"
        ])
        self.assertEqual([item["price"] for item in result["items"]], [49900, 59900, 100])

    def test_ocr_zero_and_semicolon_normalization(self):
        text = "Cafe\nAmericano 4;5OO\nTOTAL 4;5OO"
        result = extract_receipt_info(text)
        self.assertEqual(result["total"], 4500)
        self.assertEqual(result["items"][0]["price"], 4500)

    def test_total_fallback_when_ocr_misses_total_label(self):
        text = """PANTS (LONG)
49,900
SWEATER (PULLOVER)
59,900
109,900
109,900
99,900
10,000
부가세 9,990
카드 결제금액 99,900
"""
        result = extract_receipt_info(text)
        self.assertEqual(result["total"], 109900)

    def test_bounding_boxes_are_grouped_left_to_right(self):
        def box(x1, y1, x2, y2):
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ocr = {
            "text": "",
            "lines": [
                {"text": "49,900", "box": box(400, 100, 470, 120), "confidence": .9},
                {"text": "1. PANTS (LONG)", "box": box(20, 100, 180, 120), "confidence": .9},
                {"text": "2026-10-30", "box": box(20, 50, 140, 70), "confidence": .9},
                {"text": "TOTAL 49,900", "box": box(20, 160, 180, 180), "confidence": .9},
            ],
        }
        result = extract_receipt_info(ocr)
        self.assertEqual(result["date"], "2026-10-30")
        self.assertEqual(result["total"], 49900)
        self.assertEqual(result["items"][0]["name"], "PANTS (LONG)")
        self.assertEqual(result["items"][0]["price"], 49900)

    def test_empty_raw_text_falls_back_to_lines(self):
        result = extract_receipt_info({"text": "", "lines": ["2026-10-25", "TOTAL 1,500"]})
        self.assertEqual(result["date"], "2026-10-25")
        self.assertEqual(result["total"], 1500)


if __name__ == "__main__":
    unittest.main()
