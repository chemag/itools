#!/usr/bin/env python3

"""itools-bayer_unittest.py: itools bayer unittest.

# runme
# $ ./itools-bayer_unittest.py
"""

import importlib
import math
import numpy as np
import os
import shlex
import string
import tempfile
import unittest

itools_bayer = importlib.import_module("itools-bayer")


processImageTestCases = [
    # (a) component order
    {
        "name": "basic-8x8.noop",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_ggrb8",
        "o_pix_fmt": "bayer_ggrb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
            [
                [[1, 3], [9, 11]],
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
    },
    {
        "name": "basic-8x8.01",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rgbg8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
    # (b) depth/pix_fmt changes
    # bayer8->bayer16
    {
        "name": "basic-bayer_bggr8-bayer_bggr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
        "name": "basic-8x16.le.readable",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "planar_order": "BGgR",
        "bayer_planar_image": np.array(
            [
                [[0x1, 0x3], [0x9, 0xB]],
                [[0x2, 0x4], [0xA, 0xC]],
                [[0x5, 0x7], [0xD, 0xF]],
                [[0x6, 0x8], [0xE, 0x10]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10\x00",
    },
    {
        "name": "basic-8x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
            [
                [[0x2, 0x4], [0xA, 0xC]],
                [[0x5, 0x7], [0xD, 0xF]],
                [[0x6, 0x8], [0xE, 0x10]],
                [[0x1, 0x3], [0x9, 0xB]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10\x00",
    },
    {
        "name": "basic-8x16.be",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16be",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
            [
                [[0x2, 0x4], [0xA, 0xC]],
                [[0x5, 0x7], [0xD, 0xF]],
                [[0x6, 0x8], [0xE, 0x10]],
                [[0x1, 0x3], [0x9, 0xB]],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10",
    },
    # bayer10->bayer16 (extended)
    {
        "name": "basic-extended10x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10",  # SRGGB10
        "o_pix_fmt": "bayer_rggb16le",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        # TODO(chema): fix this
        "planar_order": "RgGB",
        "bayer_planar_image": np.array(
            [
                [[0x40, 0x8140], [0x440, 0x8540]],
                [[0x240, 0x8340], [0x640, 0x8740]],
                [[0x40C0, 0xC1C0], [0x44C0, 0xC5C0]],
                [[0x42C0, 0xC3C0], [0x46C0, 0xC7C0]],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
    },
    # bayer10->bayer16 (packed)
    {
        "name": "basic-packed10x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "pRAA",  # SRGGB10P
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
        "planar_order": itools_bayer.DEFAULT_PLANAR_ORDER,
        "bayer_planar_image": np.array(
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
                prefix="itools-bayer_unittest.", suffix=".bin"
            ).name
            with open(infile, "wb") as f:
                f.write(test_case["input"])
            # prepare output file(s)
            outfile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.", suffix=".bin"
            ).name
            expected_output = test_case["output"]
            logfile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.", suffix=".log"
            ).name
            # prepare parameters
            i_pix_fmt = test_case["i_pix_fmt"]
            width = test_case["width"]
            height = test_case["height"]
            o_pix_fmt = test_case["o_pix_fmt"]
            planar_order = test_case["planar_order"]
            debug = test_case["debug"]

            # 1. run forward conversion
            bayer_image = itools_bayer.convert_image_planar_mode(
                infile,
                i_pix_fmt,
                width,
                height,
                outfile,
                o_pix_fmt,
                planar_order,
                debug,
            )
            # check the planar representation is correct
            absolute_tolerance = 1
            np.testing.assert_allclose(
                test_case["bayer_planar_image"],
                bayer_image.GetPlanar(planar_order),
                atol=absolute_tolerance,
                err_msg=f"error on forward case {test_case['name']}",
            )
            # read output file
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            self.assertEqual(
                expected_output,
                output,
                f"error on forward test {test_case['name']}",
            )

            # 2. run output loop conversion
            _ = itools_bayer.convert_image_planar_mode(
                outfile,
                o_pix_fmt,
                width,
                height,
                outfile,
                o_pix_fmt,
                planar_order,
                debug,
            )
            # read output file
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            self.assertEqual(
                test_case["output"],
                output,
                f"error on output loop test {test_case['name']}",
            )

            # 3. run input loop conversion
            _ = itools_bayer.convert_image_planar_mode(
                infile,
                i_pix_fmt,
                width,
                height,
                outfile,
                i_pix_fmt,
                planar_order,
                debug,
            )
            # read output file
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            self.assertEqual(
                test_case["input"],
                output,
                f"error on input loop test {test_case['name']}",
            )


if __name__ == "__main__":
    unittest.main()
