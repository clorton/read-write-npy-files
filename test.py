#! /usr/bin/env python3

from pathlib import Path
import unittest

import numpy as np

WORK_DIR = Path(__file__).parent.absolute()


class NpyFileTests(unittest.TestCase):

    def test_one_d(self):

        one = np.load(WORK_DIR / "one_d.npy")
        self.assertEqual(one.shape, (32,))
        self.assertListEqual(list(one), list(range(32)))

        return

    def test_two_d(self):

        two = np.load(WORK_DIR / "two_d.npy")
        self.assertEqual(two.shape, (16, 16))
        self.assertListEqual(list(two[0,:]), list(range(16)))
        self.assertListEqual(list(two[:,0]), list(range(0, 256, 16)))

        return


if __name__ == "__main__":
    unittest.main()
