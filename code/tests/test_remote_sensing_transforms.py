"""Verify geometric alignment and padding visibility without a GPU runtime."""

import unittest

import numpy as np
from PIL import Image

from scripts.remote_sensing_transforms import train_transform


class LowerBoundNoFlip:
    def uniform(self, low, high):
        return low

    def randint(self, low, high):
        return low

    def random(self):
        return 1.0


class RemoteSensingTransformTest(unittest.TestCase):
    def test_scaled_source_mask_and_padding_are_aligned(self):
        image = Image.new("RGB", (512, 512), (12, 34, 56))
        label = Image.new("L", (512, 512), 2)
        image, label, valid = train_transform(
            image, label, 512, "rs_standard", LowerBoundNoFlip(), (0.5, 1.5)
        )
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(label.getpixel((0, 0)), 2)
        self.assertEqual(label.getpixel((400, 400)), 255)
        self.assertEqual(valid.getpixel((0, 0)), 255)
        self.assertEqual(valid.getpixel((400, 400)), 0)
        self.assertEqual(np.count_nonzero(np.asarray(valid)), 256 * 256)

    def test_unlabeled_target_stays_unlabeled(self):
        image = Image.new("RGB", (512, 512), (12, 34, 56))
        _, label, valid = train_transform(
            image, None, 512, "rs_standard", LowerBoundNoFlip(), (0.5, 1.5)
        )
        self.assertIsNone(label)
        self.assertEqual(np.count_nonzero(np.asarray(valid)), 256 * 256)

    def test_default_isprs_scale_does_not_create_padding(self):
        image = Image.new("RGB", (512, 512), (12, 34, 56))
        _, _, valid = train_transform(image, None, 512, "rs_standard", LowerBoundNoFlip())
        self.assertEqual(np.count_nonzero(np.asarray(valid)), 512 * 512)


if __name__ == "__main__":
    unittest.main()
