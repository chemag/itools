#!/usr/bin/env python3

"""itools-bayer-conversor_unittest.py: itools bayer unittest.

# runme
# $ ./itools-bayer-conversor_unittest.py
"""

import importlib
import math
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
        "output": b"\x05\x02\x07\x04\x06\x01\x08\x03\x0d\x0a\x0f\x0c\x0e\x09\x10\x0b",
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
            itools_bayer_conversor.process_image(
                infile, i_pix_fmt, width, height, outfile, o_pix_fmt, logfd, debug
            )
            # read output file
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            self.assertEqual(
                expected_output,
                output,
            )
            logfd.close()


if __name__ == "__main__":
    unittest.main()
