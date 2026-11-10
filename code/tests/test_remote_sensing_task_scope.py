"""Check task isolation and target-label visibility without ISPRS data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from scripts import verify_dgle_remote_sensing as verifier


class TaskScopeTests(unittest.TestCase):
    def fixture(self, root):
        expected = dict(verifier.EXPECTED)
        counts = {}
        for split in ("loveda_rural_train", "loveda_urban_train", "loveda_urban_val"):
            name = split[len("loveda_"):]
            labeled = name != "urban_train"
            expected[split] = (1, labeled, (2, 2), set(range(7)) | {255} if labeled else None)
            directory = root / "loveda" / name
            directory.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(directory / "image.png")
            label = ""
            if labeled:
                Image.new("L", (2, 2), 0).save(directory / "mask.png")
                label = f"loveda/{name}/mask.png"
            (root / "manifests").mkdir(exist_ok=True)
            with (root / "manifests" / (split + ".csv")).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image", "label"])
                writer.writerow([f"loveda/{name}/image.png", label])
            counts[name] = 1
        (root / "metadata.json").write_text(json.dumps({"counts": {"loveda": counts}}))
        return expected

    def test_r2u_without_any_isprs_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(verifier, "EXPECTED", self.fixture(root)):
                verifier.verify(root, task="r2u")

    def test_target_label_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.fixture(root)
            (root / "loveda/urban_train/labels").mkdir()
            with patch.object(verifier, "EXPECTED", expected):
                with self.assertRaisesRegex(RuntimeError, "target train label directory"):
                    verifier.verify(root, task="r2u")

    def test_invalid_class_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.fixture(root)
            Image.new("L", (2, 2), 7).save(root / "loveda/rural_train/mask.png")
            with patch.object(verifier, "EXPECTED", expected):
                with self.assertRaisesRegex(RuntimeError, "unexpected label"):
                    verifier.verify(root, task="r2u")


if __name__ == "__main__":
    unittest.main()
