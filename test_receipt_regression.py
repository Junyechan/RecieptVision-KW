import json
import unittest
from pathlib import Path
from information_extractor import extract_receipt_info, _is_excluded_line, _parse_money_token

BASE = Path(__file__).resolve().parent

class ReceiptRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = json.loads((BASE / "outputs" / "clear_korean_receipt_final.json").read_text(encoding="utf-8"))
        cls.crop = json.loads((BASE / "outputs" / "items_ocr" / "items_ocr_comparison.json").read_text(encoding="utf-8"))
        ocr = dict(cls.full["ocr"])
        ocr["item_table_ocr"] = cls.crop
        cls.result = extract_receipt_info(ocr)

    def test_three_items_found(self):
        self.assertEqual(len(self.result["items"]), 3)

    def test_pants_price(self):
        self.assertEqual(self.result["items"][0]["price"], 49900)

    def test_sweater_price(self):
        self.assertEqual(self.result["items"][1]["price"], 59900)

    def test_bag_price(self):
        self.assertEqual(self.result["items"][2]["price"], 100)

    def test_total_from_item_sum(self):
        self.assertEqual(self.result["total"], 109900)

    def test_phone_number_excluded(self):
        self.assertTrue(_is_excluded_line(".070-7799-0095"))

    def test_ocr_money_normalization(self):
        self.assertEqual(_parse_money_token("49,g0C"), 49900)
        self.assertEqual(_parse_money_token("59,9OC"), 59900)

if __name__ == "__main__":
    unittest.main(verbosity=2)
