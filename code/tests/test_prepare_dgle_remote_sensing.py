"""Small archive regression for labeled evaluation and unlabeled adaptation."""

import csv
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_dgle_remote_sensing as prepare  # noqa: E402


def tiff_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (512, 512), color).save(stream, format="TIFF")
    return stream.getvalue()


class IsprsPreparationTest(unittest.TestCase):
    def test_target_train_has_no_mask_and_eval_has_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "isprs" / "raw" / "Potsdam"
            raw.mkdir(parents=True)
            with zipfile.ZipFile(raw / "2_Ortho_RGB.zip", "w") as archive:
                for tile in ("6_10", "2_11"):
                    archive.writestr(f"top_potsdam_{tile}_RGB.tif", tiff_bytes((10, 20, 30)))
            with zipfile.ZipFile(raw / "5_Labels_all_noBoundary.zip", "w") as archive:
                archive.writestr("top_potsdam_2_11_label.tif", tiff_bytes((255, 255, 255)))
            with zipfile.ZipFile(raw / "5_Labels_all.zip", "w") as archive:
                archive.writestr("top_potsdam_2_11_label.tif", tiff_bytes((255, 0, 0)))
            output = root / "dgle"
            with patch.object(prepare, "PD_TRAIN", ("6_10",)):
                self.assertEqual(prepare.prepare_isprs(root, output, True, "pd_rgb_train"),
                                 {"pd_rgb_train": 1})
            with patch.object(prepare, "PD_EVAL", ("2_11",)):
                self.assertEqual(prepare.prepare_isprs(root, output, True, "pd_rgb_eval"),
                                 {"pd_rgb_eval": 1})
            with (output / "manifests" / "isprs_pd_rgb_train.csv").open(newline="") as stream:
                target_row = next(csv.DictReader(stream))
            self.assertEqual(target_row["label"], "")
            self.assertFalse((output / "isprs" / "pd_rgb_train" / "labels").exists())
            with (output / "manifests" / "isprs_pd_rgb_eval.csv").open(newline="") as stream:
                eval_row = next(csv.DictReader(stream))
            with Image.open(output / eval_row["label"]) as mask:
                self.assertEqual(mask.getpixel((0, 0)), 0)
            with patch.object(prepare, "PD_EVAL", ("2_11",)):
                prepare.prepare_isprs(root, output, True, "pd_rgb_eval", pd_eval_complete=True)
            with Image.open(output / eval_row["label"]) as mask:
                self.assertEqual(mask.getpixel((0, 0)), 5)


if __name__ == "__main__":
    unittest.main()
