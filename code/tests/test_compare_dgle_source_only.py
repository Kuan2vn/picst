"""Check that the published row is aligned with our class order and metrics."""

import unittest

from scripts.compare_dgle_source_only import REFERENCE, compare


class CompareDGLESourceOnlyTest(unittest.TestCase):
    def test_published_rows_recover_reported_means(self):
        for task, ref in REFERENCE.items():
            with self.subTest(task=task):
                iou = [value / 100 for value in ref["iou_percent"]]
                metrics = {
                    "iou": iou, "miou": sum(iou) / len(iou),
                    "valid_pixels": 123,
                }
                if task == "vh2pd":
                    metrics["miou5"] = sum(iou[:5]) / 5
                report = compare({
                    "task": task, "split": ref["split"],
                    "source_checkpoint_sha256": "a" * 64,
                    "metrics": metrics,
                })
                self.assertEqual([row["class"] for row in report["classes"]], ref["classes"])
                self.assertTrue(all(row["delta_percentage_points"] == 0 for row in report["classes"]))
                self.assertLess(abs(report["scores"]["miou"]["delta_percentage_points"]), 0.01)

    def test_wrong_eval_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "split"):
            compare({"task": "r2u", "split": "loveda_urban_train"})

    def test_checkpoint_hash_mismatch_is_rejected(self):
        ref = REFERENCE["r2u"]
        iou = [value / 100 for value in ref["iou_percent"]]
        evaluation = {"task": "r2u", "split": ref["split"],
                      "source_checkpoint_sha256": "a" * 64,
                      "metrics": {"iou": iou, "miou": sum(iou) / 7,
                                  "valid_pixels": 123}}
        protocol = {"task": "r2u", "stage": "eval", "split": ref["split"],
                    "files_sha256": {"source_checkpoint": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            compare(evaluation, protocol)


if __name__ == "__main__":
    unittest.main()
