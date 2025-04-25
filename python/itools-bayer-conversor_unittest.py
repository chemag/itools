#!/usr/bin/env python3

"""itools-bayer-conversor_unittest.py: itools bayer unittest.

# runme
# $ ./itools-bayer-conversor_unittest.py
"""

import importlib
import math
import numpy as np
import os
import shlex
import string
import tempfile
import unittest

itools_bayer_conversor = importlib.import_module("itools-bayer-conversor")


processImageTestCases = [
    {
        "name": "basic-8x8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rgbg8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
                [[5, 7], [13, 15]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x05\x02\x07\x04\x06\x01\x08\x03\x0d\x0a\x0f\x0c\x0e\x09\x10\x0b",
    },
    # bayer8->bayer16
    {
        "name": "basic-bayer_bggr8-bayer_bggr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
    },
    {
        "name": "basic-bayer_bggr8-bayer_gbrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_gbrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x01\x04\x03\x06\x05\x08\x07\x0a\x09\x0c\x0b\x0e\x0d\x10\x0f",
    },
    {
        "name": "basic-bayer_bggr8-bayer_gbrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_gbrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x01\x04\x03\x06\x05\x08\x07\x0a\x09\x0c\x0b\x0e\x0d\x10\x0f",
    },
    {
        "name": "basic-bayer_bggr8-bayer_grbg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_grbg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x06\x04\x08\x01\x05\x03\x07\x0a\x0e\x0c\x10\x09\x0d\x0b\x0f",
    },
    {
        "name": "basic-bayer_bggr8-bayer_ggbr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_ggbr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x05\x04\x07\x01\x06\x03\x08\x0a\x0d\x0c\x0f\x09\x0e\x0b\x10",
    },
    {
        "name": "basic-bayer_bggr8-bayer_ggrb8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_ggrb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x05\x04\x07\x06\x01\x08\x03\x0a\x0d\x0c\x0f\x0e\x09\x10\x0b",
    },
    {
        "name": "basic-bayer_bggr8-bayer_rgbg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_rgbg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x06\x02\x08\x04\x01\x05\x03\x07\x0e\x0a\x10\x0c\x09\x0d\x0b\x0f",
    },
    {
        "name": "basic-bayer_bggr8-bayer_bgrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bgrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
                [[1, 3], [9, 11]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x06\x05\x08\x07\x09\x0a\x0b\x0c\x0e\x0d\x10\x0f",
    },
    # bayer8->bayer16
    {
        "name": "basic-8x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rgbg8",
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[0x200, 0x400], [0xA00, 0xC00]],
                [[0x600, 0x800], [0xE00, 0x1000]],
                [[0x100, 0x300], [0x900, 0xB00]],
                [[0x500, 0x700], [0xD00, 0xF00]],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x00\x05\x00\x02\x00\x07\x00\x04\x00\x06\x00\x01\x00\x08\x00\x03\x00\x0d\x00\x0a\x00\x0f\x00\x0c\x00\x0e\x00\x09\x00\x10\x00\x0b",
    },
    {
        "name": "basic-8x16.be",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rgbg8",
        "o_pix_fmt": "bayer_bggr16be",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": np.array(
            [
                [[0x200, 0x400], [0xA00, 0xC00]],
                [[0x600, 0x800], [0xE00, 0x1000]],
                [[0x100, 0x300], [0x900, 0xB00]],
                [[0x500, 0x700], [0xD00, 0xF00]],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x05\x00\x02\x00\x07\x00\x04\x00\x06\x00\x01\x00\x08\x00\x03\x00\x0d\x00\x0a\x00\x0f\x00\x0c\x00\x0e\x00\x09\x00\x10\x00\x0b\x00",
    },
    # bayer10->bayer16 (extended)
    {
        "name": "basic-extended10x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10",  # SRGGB10
        "o_pix_fmt": "bayer_rggb16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x20",
        "bayer_planar": np.array(
            [
                [[0xC0, 0x1C0], [0x4C0, 0x5C0]],
                [[0x8240, 0x8340], [0x8640, 0x8740]],
                [[0x8040, 0x8140], [0x8440, 0x8540]],
                [[0x2C0, 0x3C0], [0x6C0, 0x7C0]],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x40\x80\xc0\x00\x40\x81\xc0\x01\x40\x82\xc0\x02\x40\x83\xc0\x03\x40\x84\xc0\x04\x40\x85\xc0\x05\x40\x86\xc0\x06\x40\x87\xc0\x07",
    },
    # bayer10->bayer16 (packed)
    {
        "name": "basic-packed10x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "pRAA",
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
        "bayer_planar": np.array(
            [
                [[0x240, 0x400], [0xCC0, 0xE00]],
                [[0x680, 0x800], [0x1000, 0x1240]],
                [[0x140, 0x300], [0xBC0, 0xD00]],
                [[0x780, 0x900], [0x1140, 0x1300]],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x80\x07\x40\x02\x00\x09\x00\x04\x80\x06\x40\x01\x00\x08\x00\x03\x40\x11\xc0\x0c\x00\x13\x00\x0e\x00\x10\xc0\x0b\x40\x12\x00\x0d",
    },
]


class MainTest(unittest.TestCase):
    def testProcessImage(self):
        """Simplest get_data test."""
        for test_case in processImageTestCases:
            print("...running %s" % test_case["name"])
            # prepare input file
            infile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer-conversor_unittest.", suffix=".bin"
            ).name
            with open(infile, "wb") as f:
                f.write(test_case["input"])
            # prepare output file(s)
            outfile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer-conversor_unittest.", suffix=".bin"
            ).name
            expected_output = test_case["output"]
            logfile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer-conversor_unittest.", suffix=".log"
            ).name
            logfd = open(logfile, "w")
            # prepare parameters
            i_pix_fmt = test_case["i_pix_fmt"]
            width = test_case["width"]
            height = test_case["height"]
            o_pix_fmt = test_case["o_pix_fmt"]
            debug = test_case["debug"]
            # run function
            bayer_planar = itools_bayer_conversor.process_image(
                infile, i_pix_fmt, width, height, outfile, o_pix_fmt, logfd, debug
            )
            # check the planar representation is correct
            absolute_tolerance = 1
            np.testing.assert_allclose(
                test_case["bayer_planar"], bayer_planar, atol=absolute_tolerance
            )
            # read output file
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            self.assertEqual(
                expected_output,
                output,
                f"error on test {test_case['name']}",
            )
            logfd.close()


if __name__ == "__main__":
    unittest.main()
