#!/usr/bin/env python3

"""itools-bayer-enctool_unittest.py: itools bayer enctools unittest.

# runme
# $ ./itools-bayer-enctool_unittest.py
"""

import importlib
import math
import os
import shlex
import string
import tempfile
import numpy as np
import unittest

itools_version = importlib.import_module("itools-version")
itools_bayer = importlib.import_module("itools-bayer")
itools_bayer_enctools = importlib.import_module("itools-bayer-enctools")


convertRg1g2bToYdgcocgTestCases = [
    {
        "name": "basic-4x4",
        "pix_fmt": "bayer_rggb8",
        "bayer_packed_image": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0xFF, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "bayer_y": np.array([[0x6F, 0x70], [0x70, 0x70]], dtype=np.uint16),
        "bayer_dg": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint16),
        "bayer_co": np.array([[0x00, 0x10], [0x20, 0x30]], dtype=np.uint16),
        "bayer_cg": np.array([[0x70, 0x70], [0x70, 0x70]], dtype=np.uint16),
    },
    {
        "name": "reverse-4x4",
        "pix_fmt": "bayer_rggb8",
        "bayer_packed_image": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0x00, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "bayer_y": np.array([[0x30, 0x70], [0x70, 0x70]], dtype=np.uint16),
        "bayer_dg": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint16),
        "bayer_co": np.array([[0x80, 0x10], [0x20, 0x30]], dtype=np.uint16),
        "bayer_cg": np.array([[0xB0, 0x70], [0x70, 0x70]], dtype=np.uint16),
    },
]


class MainTest(unittest.TestCase):
    def testConvertRg1g2bToYdgcocg(self):
        """convert_rg1g2b_to_ydgcocg test."""
        for test_case in convertRg1g2bToYdgcocgTestCases:
            print("...running %s" % test_case["name"])
            depth = itools_bayer.get_depth(test_case["pix_fmt"])
            # 1. run RG1G2B to YDgCoCg function
            bayer_packed_image = test_case["bayer_packed_image"]
            pix_fmt = itools_bayer_enctools.CV2_OPERATION_PIX_FMT_DICT[depth]
            bayer_image = itools_bayer.BayerImage.FromPacked(
                bayer_packed_image, pix_fmt
            )
            bayer_y, bayer_dg, bayer_co, bayer_cg = (
                itools_bayer_enctools.convert_rg1g2b_to_ydgcocg(
                    bayer_image,
                    depth,
                )
            )
            # check the values
            np.testing.assert_array_equal(test_case["bayer_y"], bayer_y)
            np.testing.assert_array_equal(test_case["bayer_dg"], bayer_dg)
            np.testing.assert_array_equal(test_case["bayer_co"], bayer_co)
            np.testing.assert_array_equal(test_case["bayer_cg"], bayer_cg)
            # 2. run YDgCoCg to RG1G2B function (reverse)
            bayer_image_prime = itools_bayer_enctools.convert_ydgcocg_to_rg1g2b(
                bayer_y, bayer_dg, bayer_co, bayer_cg, depth
            )
            absolute_tolerance = 1
            np.testing.assert_allclose(
                test_case["bayer_packed_image"],
                bayer_image_prime.GetPacked(),
                atol=absolute_tolerance,
                err_msg=f"error on forward case {test_case['name']}",
            )


if __name__ == "__main__":
    unittest.main()
