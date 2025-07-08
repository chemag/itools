#!/usr/bin/env python3

"""itools-bayer-y4m_unittest.py: itools bayer-y4m unittest.

# runme
# $ ./itools-bayer-y4m_unittest.py
"""

import argparse
import importlib
import math
import numpy as np
import os
import shlex
import string
import sys
import tempfile
import unittest

itools_bayer_y4m = importlib.import_module("itools-bayer-y4m")


readVideoY4MTestCases = [
    # simple copy
    {
        "name": "basic-8x8.copy",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "o_pix_fmt": "bayer_bggr8",
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
    },
    # bayer8->bayer16
    {
        "name": "basic-8x16.be.conversion",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "o_pix_fmt": "bayer_bggr16be",
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr16be\nFRAME\n\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00FRAME\n\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f\x00",
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

    @classmethod
    def comparePlanar(cls, planar, expected_planar, absolute_tolerance, label):
        assert set(expected_planar.keys()) == set(planar.keys()), "Broken planar output"
        for key in expected_planar:
            np.testing.assert_allclose(
                planar[key],
                expected_planar[key],
                atol=absolute_tolerance,
                err_msg=f"error on {label} case {key=}",
            )

    def testVideoY4M(self):
        """video reading test."""
        for test_case in self.getTestCases(readVideoY4MTestCases):
            print("...running %s" % test_case["name"])
            debug = test_case["debug"]
            # prepare input/output files
            infile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.infile.", suffix=".y4m"
            ).name
            with open(infile, "wb") as f:
                f.write(test_case["input"])
            outfile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.outfile.", suffix=".y4m"
            ).name
            # read y4m file
            bayer_video_reader = itools_bayer_y4m.BayerY4MReader.FromY4MFile(
                infile, debug
            )
            bayer_video_writer = None
            for expected_bayer_buffer in test_case["frames"]:
                # read the frame
                bayer_image = bayer_video_reader.GetFrame()
                bayer_buffer = bayer_image.GetBuffer()
                self.assertEqual(
                    bayer_buffer,
                    expected_bayer_buffer,
                    f"error on frame {test_case['name']}",
                )
                # write the frame
                if bayer_video_writer is None:
                    height = bayer_image.height
                    width = bayer_image.width
                    colorspace = bayer_video_reader.y4m_file_reader.colorspace
                    colorrange = bayer_video_reader.y4m_file_reader.input_colorrange
                    o_pix_fmt = test_case["o_pix_fmt"]
                    bayer_video_writer = itools_bayer_y4m.BayerY4MWriter.ToY4MFile(
                        outfile, height, width, colorspace, colorrange, o_pix_fmt, debug
                    )
                bayer_video_writer.AddFrame(bayer_image)
            # ensure no more frames to read
            bayer_image = bayer_video_reader.GetFrame()
            assert bayer_image is None, f"error: found added frames"
            del bayer_video_writer
            # ensure outfile is correct
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            expected_output = test_case["output"]
            self.assertEqual(
                output,
                expected_output,
                f"error on input write test {test_case['name']}",
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
