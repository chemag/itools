#!/usr/bin/env python3

"""itools-common_unittest.py: itools bayer enctools unittest.

# runme
# $ ./itools-common_unittest.py
"""

import importlib
import math
import os
import shlex
import string
import tempfile
import numpy as np
import unittest

itools_common = importlib.import_module("itools-common")


chromaSubsampleTestCases = [
    {
        "name": "simple",
        "arr": np.array(
            [
                [0x02, 0x04, 0x06, 0x08],
                [0x22, 0x24, 0x26, 0x28],
                [0x42, 0x44, 0x46, 0x48],
                [0x62, 0x64, 0x66, 0x68],
            ],
            dtype=np.uint8,
        ),
        "direct_420": np.array(
            [
                [0x13, 0x17],
                [0x53, 0x57],
            ],
            dtype=np.uint8,
        ),
        "reverse_420": np.array(
            [
                [0x13, 0x13, 0x17, 0x17],
                [0x13, 0x13, 0x17, 0x17],
                [0x53, 0x53, 0x57, 0x57],
                [0x53, 0x53, 0x57, 0x57],
            ],
            dtype=np.uint8,
        ),
        "direct_422": np.array(
            [
                [0x03, 0x07],
                [0x23, 0x27],
                [0x43, 0x47],
                [0x63, 0x67],
            ],
            dtype=np.uint8,
        ),
        "reverse_422": np.array(
            [
                [0x03, 0x03, 0x07, 0x07],
                [0x23, 0x23, 0x27, 0x27],
                [0x43, 0x43, 0x47, 0x47],
                [0x63, 0x63, 0x67, 0x67],
            ],
            dtype=np.uint8,
        ),
    },
    {
        "name": "odd-height",
        "arr": np.array(
            [
                [0x02, 0x04, 0x06, 0x08],
                [0x22, 0x24, 0x26, 0x28],
                [0x42, 0x44, 0x46, 0x48],
                [0x62, 0x64, 0x66, 0x68],
                [0x82, 0x84, 0x86, 0x88],
            ],
            dtype=np.uint8,
        ),
        "direct_420": np.array(
            [
                [0x13, 0x17],
                [0x53, 0x57],
                [0x83, 0x87],
            ],
            dtype=np.uint8,
        ),
        "reverse_420": np.array(
            [
                [0x13, 0x13, 0x17, 0x17],
                [0x13, 0x13, 0x17, 0x17],
                [0x53, 0x53, 0x57, 0x57],
                [0x53, 0x53, 0x57, 0x57],
                [0x83, 0x83, 0x87, 0x87],
            ],
            dtype=np.uint8,
        ),
        "direct_422": np.array(
            [
                [0x03, 0x07],
                [0x23, 0x27],
                [0x43, 0x47],
                [0x63, 0x67],
                [0x83, 0x87],
            ],
            dtype=np.uint8,
        ),
        "reverse_422": np.array(
            [
                [0x03, 0x03, 0x07, 0x07],
                [0x23, 0x23, 0x27, 0x27],
                [0x43, 0x43, 0x47, 0x47],
                [0x63, 0x63, 0x67, 0x67],
                [0x83, 0x83, 0x87, 0x87],
            ],
            dtype=np.uint8,
        ),
    },
    {
        "name": "odd-width",
        "arr": np.array(
            [
                [0x02, 0x04, 0x06, 0x08, 0x0A],
                [0x22, 0x24, 0x26, 0x28, 0x2A],
                [0x42, 0x44, 0x46, 0x48, 0x4A],
                [0x62, 0x64, 0x66, 0x68, 0x6A],
            ],
            dtype=np.uint8,
        ),
        "direct_420": np.array(
            [
                [0x13, 0x17, 0x1A],
                [0x53, 0x57, 0x5A],
            ],
            dtype=np.uint8,
        ),
        "reverse_420": np.array(
            [
                [0x13, 0x13, 0x17, 0x17, 0x1A],
                [0x13, 0x13, 0x17, 0x17, 0x1A],
                [0x53, 0x53, 0x57, 0x57, 0x5A],
                [0x53, 0x53, 0x57, 0x57, 0x5A],
            ],
            dtype=np.uint8,
        ),
        "direct_422": np.array(
            [
                [0x03, 0x07, 0x0A],
                [0x23, 0x27, 0x2A],
                [0x43, 0x47, 0x4A],
                [0x63, 0x67, 0x6A],
            ],
            dtype=np.uint8,
        ),
        "reverse_422": np.array(
            [
                [0x03, 0x03, 0x07, 0x07, 0x0A],
                [0x23, 0x23, 0x27, 0x27, 0x2A],
                [0x43, 0x43, 0x47, 0x47, 0x4A],
                [0x63, 0x63, 0x67, 0x67, 0x6A],
            ],
            dtype=np.uint8,
        ),
    },
    {
        "name": "odd-both",
        "arr": np.array(
            [
                [0x02, 0x04, 0x06, 0x08, 0x0A],
                [0x22, 0x24, 0x26, 0x28, 0x2A],
                [0x42, 0x44, 0x46, 0x48, 0x4A],
                [0x62, 0x64, 0x66, 0x68, 0x6A],
                [0x82, 0x84, 0x86, 0x88, 0x8A],
            ],
            dtype=np.uint8,
        ),
        "direct_420": np.array(
            [
                [0x13, 0x17, 0x1A],
                [0x53, 0x57, 0x5A],
                [0x83, 0x87, 0x8A],
            ],
            dtype=np.uint8,
        ),
        "reverse_420": np.array(
            [
                [0x13, 0x13, 0x17, 0x17, 0x1A],
                [0x13, 0x13, 0x17, 0x17, 0x1A],
                [0x53, 0x53, 0x57, 0x57, 0x5A],
                [0x53, 0x53, 0x57, 0x57, 0x5A],
                [0x83, 0x83, 0x87, 0x87, 0x8A],
            ],
            dtype=np.uint8,
        ),
        "direct_422": np.array(
            [
                [0x03, 0x07, 0x0A],
                [0x23, 0x27, 0x2A],
                [0x43, 0x47, 0x4A],
                [0x63, 0x67, 0x6A],
                [0x83, 0x87, 0x8A],
            ],
            dtype=np.uint8,
        ),
        "reverse_422": np.array(
            [
                [0x03, 0x03, 0x07, 0x07, 0x0A],
                [0x23, 0x23, 0x27, 0x27, 0x2A],
                [0x43, 0x43, 0x47, 0x47, 0x4A],
                [0x63, 0x63, 0x67, 0x67, 0x6A],
                [0x83, 0x83, 0x87, 0x87, 0x8A],
            ],
            dtype=np.uint8,
        ),
    },
]


class MainTest(unittest.TestCase):
    def testChromaSubsample(self):
        """clip_integer_and_scale test."""
        for test_case in chromaSubsampleTestCases:
            print("...running %s" % test_case["name"])
            # 1. run direct 420 subsampling
            arr = test_case["arr"]
            expected_subsampled_420_arr = test_case["direct_420"]
            subsampled_420_arr = itools_common.chroma_subsample_direct(arr, "420")
            np.testing.assert_array_equal(
                expected_subsampled_420_arr,
                subsampled_420_arr,
                err_msg=f"error on direct 420 subsampling case {test_case['name']}",
            )
            # 2. run reverse 420 subsampling
            arr = test_case["direct_420"]
            expected_reverse_subsampled_420_arr = test_case["reverse_420"]
            in_luma_matrix = test_case["arr"]
            reverse_subsampled_420_arr = itools_common.chroma_subsample_reverse(
                in_luma_matrix, arr, "420"
            )
            np.testing.assert_array_equal(
                expected_reverse_subsampled_420_arr,
                reverse_subsampled_420_arr,
                err_msg=f"error on reverse 420 subsampling case {test_case['name']}",
            )
            # 3. run direct 422 subsampling
            arr = test_case["arr"]
            expected_subsampled_422_arr = test_case["direct_422"]
            subsampled_422_arr = itools_common.chroma_subsample_direct(arr, "422")
            np.testing.assert_array_equal(
                expected_subsampled_422_arr,
                subsampled_422_arr,
                err_msg=f"error on direct 422 subsampling case {test_case['name']}",
            )
            # 4. run reverse 422 subsampling
            arr = test_case["direct_422"]
            expected_reverse_subsampled_422_arr = test_case["reverse_422"]
            in_luma_matrix = test_case["arr"]
            reverse_subsampled_422_arr = itools_common.chroma_subsample_reverse(
                in_luma_matrix, arr, "422"
            )
            np.testing.assert_array_equal(
                expected_reverse_subsampled_422_arr,
                reverse_subsampled_422_arr,
                err_msg=f"error on reverse 422 subsampling case {test_case['name']}",
            )


if __name__ == "__main__":
    unittest.main()
