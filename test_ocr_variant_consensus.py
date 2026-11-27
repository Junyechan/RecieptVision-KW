"""Regression tests for multi-variant item-table OCR reconciliation."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from information_extractor import extract_receipt_info, _merge_ocr_variant_lines

BASE = Path(__file__).resolve().parent
FIXTURE = BASE / "tests" / "fixtures" / "items_ocr_comparison.json"


class OCRVariantConsensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_all_variants_are_merged(self):
        merged = _merge_ocr_variant_lines(self.variants)
        self.assertGreater(len(merged), 0)
        # The same spatial detection must not be duplicated for each variant.
        pants = [x for x in merged if "PANTS" in x["text"].upper()]
        self.assertEqual(len(pants), 1)

    def test_extracts_three_items(self):
        result = extract_receipt_info({
            "text": "", "lines": [], "item_table_ocr": self.variants
        })
        self.assertEqual(len(result["items"]), 3)

    def test_prices_are_correct(self):
        result = extract_receipt_info({
            "text": "", "lines": [], "item_table_ocr": self.variants
        })
        self.assertEqual([item["price"] for item in result["items"]],
                         [49900, 59900, 100])

    def test_item_sum_is_109900(self):
        result = extract_receipt_info({
            "text": "", "lines": [], "item_table_ocr": self.variants
        })
        self.assertEqual(result["validation"]["item_sum"], 109900)
        self.assertEqual(result["total"], 109900)
        self.assertTrue(result["validation"]["total_matches_item_sum"])

    def test_consensus_does_not_blindly_choose_clahe(self):
        result = extract_receipt_info({
            "text": "", "lines": [], "item_table_ocr": self.variants
        })
        names = [item["name"] for item in result["items"]]
        self.assertIn("SWEATER(PU LUVER)", names)
        self.assertNotIn("SMEATER(PULOVER)", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
