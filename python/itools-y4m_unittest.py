#!/usr/bin/env python3

"""itools-y4m_unittest.py: itools-y4m unittest.

# runme
# $ ./itools-y4m_unittest.py
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

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")
itools_unittest = importlib.import_module("itools-unittest")


DEFAULT_ABSOLUTE_TOLERANCE = 1

y4mReadWriteTestCases = [
    # simple copy
    {
        "name": "mono",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "num_frames": 2,
        "frames": (
            np.array(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [16, 17, 18, 19],
                    [20, 21, 22, 23],
                    [24, 25, 26, 27],
                    [28, 29, 30, 31],
                ],
                dtype=np.uint8,
            ),
        ),
    },
    # mono10
    {
        "name": "mono10",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "num_frames": 1,
        "frames": (
            np.array(
                [
                    [1, 259, 517, 775],
                    [9, 267, 525, 783],
                    [17, 275, 533, 791],
                    [25, 283, 541, 799],
                ],
                dtype=np.uint16,
            ),
        ),
    },
    # mono12
    {
        "name": "mono12",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono12 XCOLORRANGE=FULL\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "num_frames": 1,
        "frames": (
            np.array(
                [
                    [1, 259, 517, 775],
                    [9, 267, 525, 783],
                    [17, 275, 533, 791],
                    [25, 283, 541, 799],
                ],
                dtype=np.uint16,
            ),
        ),
    },
    # mono14
    {
        "name": "mono14",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono14 XCOLORRANGE=FULL\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "num_frames": 1,
        "frames": (
            np.array(
                [
                    [1, 259, 517, 775],
                    [9, 267, 525, 783],
                    [17, 275, 533, 791],
                    [25, 283, 541, 799],
                ],
                dtype=np.uint16,
            ),
        ),
    },
    # mono16
    {
        "name": "mono16",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL\nFRAME\n\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0fFRAME\n\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
        "num_frames": 2,
        "frames": (
            np.array(
                [
                    [0, 256, 512, 768],
                    [1024, 1280, 1536, 1792],
                    [2048, 2304, 2560, 2816],
                    [3072, 3328, 3584, 3840],
                ],
                dtype=np.uint16,
            ),
            np.array(
                [
                    [4096, 4352, 4608, 4864],
                    [5120, 5376, 5632, 5888],
                    [6144, 6400, 6656, 6912],
                    [7168, 7424, 7680, 7936],
                ],
                dtype=np.uint16,
            ),
        ),
    },
    {
        "name": "420",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 C420 XCOLORRANGE=FULL\nFRAME\n\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7eFRAME\n\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e",
        "num_frames": 2,
        "frames": (
            np.array(
                [
                    [[3, 126, 127], [2, 126, 127], [3, 125, 127], [4, 125, 127]],
                    [[5, 126, 127], [6, 126, 127], [7, 125, 127], [7, 125, 127]],
                    [[9, 125, 127], [10, 125, 127], [11, 126, 127], [12, 126, 127]],
                    [[12, 125, 127], [12, 125, 127], [14, 126, 127], [13, 126, 127]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[19, 126, 127], [18, 126, 127], [19, 125, 127], [20, 125, 127]],
                    [[21, 126, 127], [22, 126, 127], [23, 125, 127], [23, 125, 127]],
                    [[25, 125, 127], [26, 125, 127], [27, 126, 127], [28, 126, 127]],
                    [[28, 125, 127], [28, 125, 127], [30, 126, 127], [29, 126, 127]],
                ],
                dtype=np.uint8,
            ),
        ),
    },
    {
        "name": "422",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 C422 XCOLORRANGE=FULL\nFRAME\n\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7d\x82\x84\x84\x83\x84\x84\x84\x83FRAME\n\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7d\x82\x84\x84\x83\x84\x84\x84\x83",
        "num_frames": 2,
        "frames": (
            np.array(
                [
                    [[3, 130, 127], [2, 130, 127], [3, 132, 127], [4, 132, 127]],
                    [[5, 132, 127], [6, 132, 127], [7, 131, 127], [7, 131, 127]],
                    [[9, 132, 126], [10, 132, 126], [11, 132, 125], [12, 132, 125]],
                    [[12, 132, 125], [12, 132, 125], [14, 131, 125], [13, 131, 125]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[19, 130, 127], [18, 130, 127], [19, 132, 127], [20, 132, 127]],
                    [[21, 132, 127], [22, 132, 127], [23, 131, 127], [23, 131, 127]],
                    [[25, 132, 126], [26, 132, 126], [27, 132, 125], [28, 132, 125]],
                    [[28, 132, 125], [28, 132, 125], [30, 131, 125], [29, 131, 125]],
                ],
                dtype=np.uint8,
            ),
        ),
    },
    {
        "name": "444",
        "debug": 0,
        "contents": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 C444 XCOLORRANGE=FULL\nFRAME\n\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82FRAME\n\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
        "num_frames": 2,
        "frames": (
            np.array(
                [
                    [[3, 130, 127], [2, 132, 127], [3, 132, 127], [4, 131, 127]],
                    [[5, 132, 126], [6, 132, 125], [7, 132, 125], [7, 132, 126]],
                    [[9, 132, 126], [10, 132, 125], [11, 132, 125], [12, 131, 125]],
                    [[12, 129, 126], [12, 130, 126], [14, 129, 126], [13, 130, 127]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[19, 130, 127], [18, 132, 127], [19, 132, 127], [20, 131, 127]],
                    [[21, 132, 126], [22, 132, 125], [23, 132, 125], [23, 132, 126]],
                    [[25, 132, 126], [26, 132, 125], [27, 132, 125], [28, 131, 125]],
                    [[28, 129, 126], [28, 130, 126], [30, 129, 126], [29, 130, 127]],
                ],
                dtype=np.uint8,
            ),
        ),
    },
]


class MainTest(itools_unittest.TestCase):

    def testY4mReadWriteTestCases(self):
        """y4m read/write test."""
        function_name = "testY4mReadWriteTestCases"
        absolute_tolerance = 0

        for test_case in self.getTestCases(function_name, y4mReadWriteTestCases):
            print(f"...running \"{function_name}.{test_case['name']}\"")
            debug = test_case["debug"]

            # prepare input/output files
            infile = tempfile.NamedTemporaryFile(
                prefix="itools-y4m_unittest.infile.", suffix=".y4m"
            ).name
            with open(infile, "wb") as f:
                f.write(test_case["contents"])
            outfile = tempfile.NamedTemporaryFile(
                prefix="itools-y4m_unittest.outfile.", suffix=".y4m"
            ).name
            # open input file
            y4m_file_reader = itools_y4m.Y4MFileReader(
                infile, colorrange=None, debug=debug
            )
            # open output file
            y4m_height = y4m_file_reader.height
            y4m_width = y4m_file_reader.width
            y4m_colorspace = y4m_file_reader.colorspace
            y4m_colorrange = y4m_file_reader.input_colorrange
            y4m_file_writer = itools_y4m.Y4MFileWriter(
                y4m_height,
                y4m_width,
                y4m_colorspace,
                y4m_colorrange,
                outfile,
                debug=debug,
            )

            frame_id = 0
            while True:
                # read image
                outyvu = y4m_file_reader.read_frame()
                if outyvu is None:
                    break
                # check the frame
                expected_outyvu = test_case["frames"][frame_id]
                np.testing.assert_allclose(
                    outyvu,
                    expected_outyvu,
                    atol=absolute_tolerance,
                    err_msg=f"error on frame buffer test {test_case['name']}",
                )
                # write image
                y4m_file_writer.write_frame_cv2_yvu(outyvu)
                frame_id += 1
            del y4m_file_reader
            del y4m_file_writer
            # check output file
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            expected_output = test_case["contents"]
            self.assertEqual(
                output,
                expected_output,
                f"error on output file test {test_case['name']}",
            )


if __name__ == "__main__":
    itools_unittest.main(sys.argv)
