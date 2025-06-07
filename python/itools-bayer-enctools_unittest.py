#!/usr/bin/env python3

"""itools-bayer-enctool_unittest.py: itools bayer enctools unittest.

# runme
# $ ./itools-bayer-enctool_unittest.py
"""

import argparse
import importlib
import math
import os
import shlex
import string
import sys
import tempfile
import numpy as np
import unittest

itools_version = importlib.import_module("itools-version")
itools_bayer = importlib.import_module("itools-bayer")
itools_bayer_enctools = importlib.import_module("itools-bayer-enctools")


clipIntegerAndScaleTestCases = [
    {
        "name": "8-bits",
        "depth": 8,
        "arr": np.array(
            [
                [-0xFF, -0x80, -0x7F, 0x00, 0x7F, 0x80, 0xFF],
            ],
            dtype=np.int16,
        ),
        "clipped_arr": np.array(
            [
                [0x00, 0x40, 0x40, 0x80, 0xBF, 0xC0, 0xFF],
            ],
            dtype=np.uint8,
        ),
    },
    {
        "name": "10-bits",
        "depth": 10,
        "arr": np.array(
            [
                [-0x3FF, -0x200, -0x1FF, 0x00, 0x1FF, 0x200, 0x3FF],
            ],
            dtype=np.int32,
        ),
        "clipped_arr": np.array(
            [
                [0x000, 0x100, 0x100, 0x200, 0x2FF, 0x300, 0x3FF],
            ],
            dtype=np.uint16,
        ),
    },
    {
        "name": "16-bits",
        "depth": 16,
        "arr": np.array(
            [
                [-0xFFFF, -0x8000, -0x7FFF, 0x00, 0x7FFF, 0x8000, 0xFFFF],
            ],
            dtype=np.int32,
        ),
        "clipped_arr": np.array(
            [
                [0x0000, 0x4000, 0x4000, 0x8000, 0xBFFF, 0xC000, 0xFFFF],
            ],
            dtype=np.uint16,
        ),
    },
]

convertRg1g2bToYdgcocgTestCases = [
    {
        "name": "rggb8-forward-4x4",
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
        "name": "rggb8-reverse-4x4",
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
    {
        "name": "rggb10-forward-4x4",
        "pix_fmt": "SRGGB10",
        "bayer_packed_image": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0xFF, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint16,
        ),
        "bayer_y": np.array([[0x06F, 0x070], [0x070, 0x070]], dtype=np.uint16),
        "bayer_dg": np.array([[0x220, 0x230], [0x240, 0x250]], dtype=np.uint16),
        "bayer_co": np.array([[0x180, 0x190], [0x1A0, 0x1B0]], dtype=np.uint16),
        "bayer_cg": np.array([[0x1F0, 0x1F0], [0x1F0, 0x1F0]], dtype=np.uint16),
    },
    {
        "name": "rggb10-reverse-4x4",
        "pix_fmt": "SRGGB10",
        "bayer_packed_image": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0x00, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint16,
        ),
        "bayer_y": np.array([[0x030, 0x070], [0x070, 0x070]], dtype=np.uint16),
        "bayer_dg": np.array([[0x220, 0x230], [0x240, 0x250]], dtype=np.uint16),
        "bayer_co": np.array([[0x200, 0x190], [0x1A0, 0x1B0]], dtype=np.uint16),
        "bayer_cg": np.array([[0x230, 0x1F0], [0x1F0, 0x1F0]], dtype=np.uint16),
    },
    {
        "name": "rggb10-scaled-forward-4x4",
        "pix_fmt": "SRGGB10",
        "bayer_packed_image": np.array(
            [
                [0x000, 0x100, 0x040, 0x0C0],
                [0x200, 0x3FF, 0x240, 0x3C0],
                [0x080, 0x080, 0x0C0, 0x040],
                [0x280, 0x380, 0x2C0, 0x340],
            ],
            dtype=np.uint16,
        ),
        "bayer_y": np.array([[0x1BF, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
        "bayer_dg": np.array([[0x280, 0x2C0], [0x300, 0x340]], dtype=np.uint16),
        "bayer_co": np.array([[0x000, 0x040], [0x080, 0x0C0]], dtype=np.uint16),
        "bayer_cg": np.array([[0x1C0, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
    },
    {
        "name": "rggb10-scaled-reverse-4x4",
        "pix_fmt": "SRGGB10",
        "bayer_packed_image": np.array(
            [
                [0x000, 0x100, 0x040, 0x0C0],
                [0x200, 0x000, 0x240, 0x3C0],
                [0x080, 0x080, 0x0C0, 0x040],
                [0x280, 0x380, 0x2C0, 0x340],
            ],
            dtype=np.uint16,
        ),
        "bayer_y": np.array([[0x0C0, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
        "bayer_dg": np.array([[0x280, 0x2C0], [0x300, 0x340]], dtype=np.uint16),
        "bayer_co": np.array([[0x200, 0x040], [0x080, 0x0C0]], dtype=np.uint16),
        "bayer_cg": np.array([[0x2C0, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
    },
]


class MainTest(unittest.TestCase):
    def getTestCases(self, test_case_list):
        global EXPERIMENT_NAME
        global EXPERIMENT_LIST

        test_case_name_list = [test_case["name"] for test_case in test_case_list]
        if EXPERIMENT_LIST:
            print(f"experiment list: {test_case_name_list}")
            self.skipTest(f"experiment list: {test_case_name_list}")

        elif EXPERIMENT_NAME is not None:
            try:
                test_case = next(
                    test_case
                    for test_case in test_case_list
                    if test_case["name"] == EXPERIMENT_NAME
                )
            except StopIteration:
                raise AssertionError(
                    f'unknown experiment: "{EXPERIMENT_NAME}" list: {test_case_name_list}'
                )
            return [
                test_case,
            ]
        else:
            return test_case_list

    def testClipIntegerAndScale(self):
        """clip_integer_and_scale test."""
        for test_case in self.getTestCases(clipIntegerAndScaleTestCases):
            print("...running %s" % test_case["name"])
            arr = test_case["arr"]
            depth = test_case["depth"]
            expected_clipped_arr = test_case["clipped_arr"]
            # 1. run forward clipping
            clipped_arr = itools_bayer_enctools.clip_integer_and_scale(arr, depth)
            np.testing.assert_array_equal(
                expected_clipped_arr,
                clipped_arr,
                err_msg=f"error on forward case {test_case['name']}",
            )
            # 2. run backward clipping
            new_arr = itools_bayer_enctools.unclip_integer_and_unscale(
                clipped_arr, depth
            )
            absolute_tolerance = 1
            np.testing.assert_allclose(
                arr,
                new_arr,
                atol=absolute_tolerance,
                err_msg=f"error on forward case {test_case['name']}",
            )

    def testConvertRg1g2bToYdgcocg(self):
        """convert_rg1g2b_to_ydgcocg test."""
        for test_case in self.getTestCases(convertRg1g2bToYdgcocgTestCases):
            print("...running %s" % test_case["name"])
            depth = itools_bayer.get_depth(test_case["pix_fmt"])
            # 1. run RG1G2B to YDgCoCg function
            bayer_packed_image = test_case["bayer_packed_image"]
            pix_fmt = itools_bayer_enctools.CV2_OPERATION_PIX_FMT_DICT[depth]
            bayer_image = itools_bayer.BayerImage.FromPacked(
                bayer_packed_image, pix_fmt
            )
            bayer_ydgcocg_planar = itools_bayer_enctools.convert_rg1g2b_to_ydgcocg(
                bayer_image,
                depth,
            )
            # check the values
            np.testing.assert_array_equal(
                test_case["bayer_y"], bayer_ydgcocg_planar["y"]
            )
            np.testing.assert_array_equal(
                test_case["bayer_dg"], bayer_ydgcocg_planar["dg"]
            )
            np.testing.assert_array_equal(
                test_case["bayer_co"], bayer_ydgcocg_planar["co"]
            )
            np.testing.assert_array_equal(
                test_case["bayer_cg"], bayer_ydgcocg_planar["cg"]
            )
            # 2. run YDgCoCg to RG1G2B function (reverse)
            bayer_image_prime = itools_bayer_enctools.convert_ydgcocg_to_rg1g2b(
                bayer_ydgcocg_planar, depth
            )
            absolute_tolerance = 1
            np.testing.assert_allclose(
                test_case["bayer_packed_image"],
                bayer_image_prime.GetBayerPacked(),
                atol=absolute_tolerance,
                err_msg=f"error on forward case {test_case['name']}",
            )


if __name__ == "__main__":
    global EXPERIMENT_NAME
    global EXPERIMENT_LIST

    parser = argparse.ArgumentParser()
    parser.add_argument("test_name", nargs="?", help="Test to run")
    parser.add_argument("--experiment", type=str)
    parser.add_argument(
        "--experiment-list", action="store_true", help="Enable experiment listing"
    )
    args, unknown = parser.parse_known_args()
    EXPERIMENT_NAME = args.experiment
    EXPERIMENT_LIST = args.experiment_list
    # clean sys.argv before passing to unittest
    if args.test_name:
        sys.argv = [sys.argv[0], args.test_name] + unknown
    else:
        sys.argv = [sys.argv[0]] + unknown
    unittest.main()
