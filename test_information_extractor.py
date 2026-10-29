"""Small deterministic tests for receipt text parsing (no receipt image needed)."""
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


if __name__ == "__main__":
    unittest.main()
