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

itools_common = importlib.import_module("itools-common")
itools_bayer = importlib.import_module("itools-bayer")
itools_bayer_y4m = importlib.import_module("itools-bayer-y4m")
itools_unittest = importlib.import_module("itools-unittest")


DEFAULT_ABSOLUTE_TOLERANCE = 1

readVideoY4MTestCases = [
    # simple copy
    {
        "name": "bayer_bggr8-bayer_bggr8",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "i_pix_fmt": "bayer_bggr8",
        "num_frames": 2,
        "i_frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "i_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
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
        "o_pix_fmt": "bayer_bggr8",
        "o_frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "o_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
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
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
    },
    # bayer8->bayer16
    {
        "name": "bayer_bggr8-bayer_bggr16le",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "i_pix_fmt": "bayer_bggr8",
        "num_frames": 2,
        "i_frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "i_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
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
        "o_pix_fmt": "bayer_bggr16le",
        "o_frames": (
            b"\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f",
            b"\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
        ),
        "o_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=np.uint16,
            )
            << 8,
            np.array(
                [
                    [16, 17, 18, 19],
                    [20, 21, 22, 23],
                    [24, 25, 26, 27],
                    [28, 29, 30, 31],
                ],
                dtype=np.uint16,
            )
            << 8,
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_bggr16le\nFRAME\n\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0fFRAME\n\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
    },
    # bayer16->bayer8
    {
        "name": "bayer_bggr16le-bayer_bggr8",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_bggr16le\nFRAME\n\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0fFRAME\n\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
        "i_pix_fmt": "bayer_bggr16le",
        "num_frames": 2,
        "i_frames": (
            b"\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f",
            b"\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
        ),
        "i_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=np.uint16,
            )
            << 8,
            np.array(
                [
                    [16, 17, 18, 19],
                    [20, 21, 22, 23],
                    [24, 25, 26, 27],
                    [28, 29, 30, 31],
                ],
                dtype=np.uint16,
            )
            << 8,
        ),
        "o_pix_fmt": "bayer_bggr8",
        "o_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
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
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
    },
    # bayer16->bayer16
    {
        "name": "bayer_bggr16le-bayer_bggr16le",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_bggr16le\nFRAME\n\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0fFRAME\n\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
        "i_pix_fmt": "bayer_bggr16le",
        "num_frames": 2,
        "i_frames": (
            b"\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f",
            b"\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
        ),
        "o_pix_fmt": "bayer_bggr16le",
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_bggr16le\nFRAME\n\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0fFRAME\n\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f",
    },
    # bayer10->bayer16 (expanded)
    {
        "name": "RG10-bayer_rggb16le",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "i_pix_fmt": "RG10",
        "num_frames": 1,
        "i_frames": (
            b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        ),
        "i_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            ),
        ),
        "o_pix_fmt": "bayer_rggb16le",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            )
            << 6,
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_rggb16le\nFRAME\n\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
    },
    # bayer16->bayer10 (expanded)
    {
        "name": "bayer_rggb16le-RG10",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_rggb16le\nFRAME\n\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        "i_pix_fmt": "bayer_rggb16le",
        "num_frames": 1,
        "i_frames": (
            b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        ),
        "o_pix_fmt": "RG10",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            ),
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
    },
    # bayer10->bayer10 (expanded-packed)
    {
        "name": "RG10-pRAA",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "i_pix_fmt": "RG10",
        "num_frames": 1,
        "i_frames": (
            b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        ),
        "o_pix_fmt": "pRAA",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            ),
        ),
        "output": b"YUV4MPEG2 W5 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=pRAA\nFRAME\n\x00\x40\x81\xc1\xdd\x02\x42\x83\xc3\xdd\x04\x44\x85\xc5\xdd\x06\x46\x87\xc7\xdd",
    },
    # bayer10->bayer10 (expanded-planar)
    {
        "name": "RG10-RG10.planar",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "i_pix_fmt": "RG10",
        "num_frames": 1,
        "i_frames": (
            b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        ),
        "o_pix_fmt": "RG10.planar",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            ),
        ),
        "output": b"YUV4MPEG2 W2 H8 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10.planar\nFRAME\n\x01\x00\x05\x02\x11\x00\x15\x02\x03\x01\x07\x03\x13\x01\x17\x03\x09\x00\x0d\x02\x19\x00\x1d\x02\x0b\x01\x0f\x03\x1b\x01\x1f\x03",
    },
    # bayer16->bayer12 (expanded)
    {
        "name": "bayer_rggb16le-RG12",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_rggb16le\nFRAME\n\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        "i_pix_fmt": "bayer_rggb16le",
        "num_frames": 1,
        "i_frames": (
            b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        ),
        "o_pix_fmt": "RG12",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            )
            << 2,
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono12 XCOLORRANGE=FULL XEXTCS=RG12\nFRAME\n\x04\x00\x0c\x04\x14\x08\x1c\x0c\x24\x00\x2c\x04\x34\x08\x3c\x0c\x44\x00\x4c\x04\x54\x08\x5c\x0c\x64\x00\x6c\x04\x74\x08\x7c\x0c",
    },
    # bayer16->bayer12 (packed)
    {
        "name": "bayer_rggb16le-pRCC",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_rggb16le\nFRAME\n\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        "i_pix_fmt": "bayer_rggb16le",
        "num_frames": 1,
        "i_frames": (
            b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        ),
        "o_pix_fmt": "pRCC",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            )
            << 2,
        ),
        "output": b"YUV4MPEG2 W6 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=pRCC\nFRAME\n\x00\x40\xc4\x81\xc1\xc4\x02\x42\xc4\x83\xc3\xc4\x04\x44\xc4\x85\xc5\xc4\x06\x46\xc4\x87\xc7\xc4",
    },
    # bayer16->bayer14 (expanded)
    {
        "name": "bayer_rggb16le-RG14",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_rggb16le\nFRAME\n\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        "i_pix_fmt": "bayer_rggb16le",
        "num_frames": 1,
        "i_frames": (
            b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        ),
        "o_pix_fmt": "RG14",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            )
            << 4,
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono14 XCOLORRANGE=FULL XEXTCS=RG14\nFRAME\n\x10\x00\x30\x10\x50\x20\x70\x30\x90\x00\xb0\x10\xd0\x20\xf0\x30\x10\x01\x30\x11\x50\x21\x70\x31\x90\x01\xb0\x11\xd0\x21\xf0\x31",
    },
    # bayer16->bayer14 (packed)
    {
        "name": "bayer_rggb16le-pREE",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono16 XCOLORRANGE=FULL XEXTCS=bayer_rggb16le\nFRAME\n\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        "i_pix_fmt": "bayer_rggb16le",
        "num_frames": 1,
        "i_frames": (
            b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
        ),
        "o_pix_fmt": "pREE",
        "o_bayer_packed": (
            np.array(
                [
                    [0x001, 0x103, 0x205, 0x307],
                    [0x009, 0x10B, 0x20D, 0x30F],
                    [0x011, 0x113, 0x215, 0x317],
                    [0x019, 0x11B, 0x21D, 0x31F],
                ],
                dtype=np.uint16,
            )
            << 4,
        ),
        "output": b"YUV4MPEG2 W7 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=pREE\nFRAME\n\x00\x40\x81\xc1\x10\x0c\xc1\x02\x42\x83\xc3\x10\x0c\xc1\x04\x44\x85\xc5\x10\x0c\xc1\x06\x46\x87\xc7\x10\x0c\xc1",
    },
    # bayer10->ydgcocg10.packed (expanded-packed)
    {
        "name": "RG10-ydgcocg10.packed",
        # there is a small conversion error
        "avoid_backwards": True,
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x00\x00\x00\x01\x40\x00\xC0\x00\x00\x02\xff\x03\x40\x02\xC0\x03\x80\x00\x80\x00\xC0\x00\x40\x00\x80\x02\x80\x03\xC0\x02\x40\x03",
        "i_pix_fmt": "RG10",
        "i_bayer_packed": (
            np.array(
                [
                    [0x000, 0x100, 0x040, 0x0C0],
                    [0x200, 0x3FF, 0x240, 0x3C0],
                    [0x080, 0x080, 0x0C0, 0x040],
                    [0x280, 0x380, 0x2C0, 0x340],
                ],
                dtype=np.uint16,
            ),
        ),
        "num_frames": 1,
        "i_frames": (
            b"\x00\x00\x00\x01\x40\x00\xc0\x00\x00\x02\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
        ),
        "o_pix_fmt": "ydgcocg10.packed",
        "o_frames": (
            b"\xbf\x01\x80\x02\xc0\x01\xc0\x02\x00\x00\xc0\x01\x40\x00\xc0\x01\xc0\x01\x00\x03\xc0\x01\x40\x03\x80\x00\xc0\x01\xc0\x00\xc0\x01",
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=ydgcocg10.packed\nFRAME\n\xbf\x01\x80\x02\xc0\x01\xc0\x02\x00\x00\xc0\x01\x40\x00\xc0\x01\xc0\x01\x00\x03\xc0\x01\x40\x03\x80\x00\xc0\x01\xc0\x00\xc0\x01",
    },
    {
        "name": "ydgcocg10.packed-RG10",
        "absolute_tolerance": 2,
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=ydgcocg10.packed\nFRAME\n\xbf\x01\x80\x02\xc0\x01\xc0\x02\x00\x00\xc0\x01\x40\x00\xc0\x01\xc0\x01\x00\x03\xc0\x01\x40\x03\x80\x00\xc0\x01\xc0\x00\xc0\x01",
        "num_frames": 1,
        "i_pix_fmt": "ydgcocg10.packed",
        "i_frames": (
            b"\xbf\x01\x80\x02\xc0\x01\xc0\x02\x00\x00\xc0\x01\x40\x00\xc0\x01\xc0\x01\x00\x03\xc0\x01\x40\x03\x80\x00\xc0\x01\xc0\x00\xc0\x01",
        ),
        "o_pix_fmt": "RG10",
        "o_bayer_packed": (
            np.array(
                [
                    [0x000, 0x100, 0x040, 0x0C0],
                    [0x200, 0x3FF, 0x240, 0x3C0],
                    [0x080, 0x080, 0x0C0, 0x040],
                    [0x280, 0x380, 0x2C0, 0x340],
                ],
                dtype=np.uint16,
            ),
        ),
        "o_frames": (
            b"\x00\x00\xff\x00\x40\x00\xc0\x00\xff\x01\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x00\x00\xff\x00\x40\x00\xc0\x00\xff\x01\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
    },
    # bayer10->ydgcocg10.planar (expanded-planar)
    {
        "name": "RG10-ydgcocg10.planar",
        # there is a small conversion error
        "avoid_backwards": True,
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x00\x00\x00\x01\x40\x00\xC0\x00\x00\x02\xff\x03\x40\x02\xC0\x03\x80\x00\x80\x00\xC0\x00\x40\x00\x80\x02\x80\x03\xC0\x02\x40\x03",
        "i_pix_fmt": "RG10",
        "i_bayer_packed": (
            np.array(
                [
                    [0x000, 0x100, 0x040, 0x0C0],
                    [0x200, 0x3FF, 0x240, 0x3C0],
                    [0x080, 0x080, 0x0C0, 0x040],
                    [0x280, 0x380, 0x2C0, 0x340],
                ],
                dtype=np.uint16,
            ),
        ),
        "num_frames": 1,
        "i_frames": (
            b"\x00\x00\x00\x01\x40\x00\xc0\x00\x00\x02\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
        ),
        "o_pix_fmt": "ydgcocg10.planar",
        "o_frames": (
            b"\xbf\x01\xc0\x01\xc0\x01\xc0\x01\x80\x02\xc0\x02\x00\x03\x40\x03\x00\x00\x40\x00\x80\x00\xc0\x00\xc0\x01\xc0\x01\xc0\x01\xc0\x01",
        ),
        "output": b"YUV4MPEG2 W2 H8 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=ydgcocg10.planar\nFRAME\n\xbf\x01\xc0\x01\xc0\x01\xc0\x01\x80\x02\xc0\x02\x00\x03\x40\x03\x00\x00\x40\x00\x80\x00\xc0\x00\xc0\x01\xc0\x01\xc0\x01\xc0\x01",
    },
    {
        "name": "ydgcocg10.planar-RG10",
        # there is a small conversion error
        "avoid_backwards": True,
        "debug": 0,
        "input": b"YUV4MPEG2 W2 H8 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=ydgcocg10.planar\nFRAME\n\xbf\x01\xc0\x01\xc0\x01\xc0\x01\x80\x02\xc0\x02\x00\x03\x40\x03\x00\x00\x40\x00\x80\x00\xc0\x00\xc0\x01\xc0\x01\xc0\x01\xc0\x01",
        "num_frames": 1,
        "i_pix_fmt": "ydgcocg10.planar",
        "i_frames": (
            b"\xbf\x01\xc0\x01\xc0\x01\xc0\x01\x80\x02\xc0\x02\x00\x03\x40\x03\x00\x00\x40\x00\x80\x00\xc0\x00\xc0\x01\xc0\x01\xc0\x01\xc0\x01",
        ),
        "o_pix_fmt": "RG10",
        "o_bayer_packed": (
            np.array(
                [
                    [0x000, 0x100, 0x040, 0x0C0],
                    [0x200, 0x3FF, 0x240, 0x3C0],
                    [0x080, 0x080, 0x0C0, 0x040],
                    [0x280, 0x380, 0x2C0, 0x340],
                ],
                dtype=np.uint16,
            ),
        ),
        "o_frames": (
            b"\x00\x00\xff\x00\x40\x00\xc0\x00\xff\x01\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
        ),
        "output": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x00\x00\xff\x00\x40\x00\xc0\x00\xff\x01\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
    },
    # bayer_bggr8->rgb8.planar
    {
        "name": "bayer_bggr8-rgb8.planar",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "i_pix_fmt": "bayer_bggr8",
        "num_frames": 2,
        "i_frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "i_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
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
        "o_pix_fmt": "rgb8.planar",
        "o_frames": (
            b"\x05\x06\x07\x07\x09\x0a\x0b\x0b\x0d\x0e\x0f\x0f\x0d\x0e\x0f\x0f\x02\x01\x02\x03\x04\x05\x06\x06\x08\x09\x0a\x0b\x0c\x0c\x0e\x0c\x00\x00\x01\x02\x00\x00\x01\x02\x04\x04\x05\x06\x08\x08\x09\x0a",
            b"\x15\x16\x17\x17\x19\x1a\x1b\x1b\x1d\x1e\x1f\x1f\x1d\x1e\x1f\x1f\x12\x11\x12\x13\x14\x15\x16\x16\x18\x19\x1a\x1b\x1c\x1c\x1e\x1c\x10\x10\x11\x12\x10\x10\x11\x12\x14\x14\x15\x16\x18\x18\x19\x1a",
        ),
        "output": b"YUV4MPEG2 W4 H12 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=rgb8.planar\nFRAME\n\x05\x06\x07\x07\x09\x0a\x0b\x0b\x0d\x0e\x0f\x0f\x0d\x0e\x0f\x0f\x02\x01\x02\x03\x04\x05\x06\x06\x08\x09\x0a\x0b\x0c\x0c\x0e\x0c\x00\x00\x01\x02\x00\x00\x01\x02\x04\x04\x05\x06\x08\x08\x09\x0aFRAME\n\x15\x16\x17\x17\x19\x1a\x1b\x1b\x1d\x1e\x1f\x1f\x1d\x1e\x1f\x1f\x12\x11\x12\x13\x14\x15\x16\x16\x18\x19\x1a\x1b\x1c\x1c\x1e\x1c\x10\x10\x11\x12\x10\x10\x11\x12\x14\x14\x15\x16\x18\x18\x19\x1a",
    },
    # RG10->rgb10.planar
    {
        "name": "RG10-rgb10.planar",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=RG10\nFRAME\n\x00\x00\x00\x01\x40\x00\xC0\x00\x00\x02\xff\x03\x40\x02\xC0\x03\x80\x00\x80\x00\xC0\x00\x40\x00\x80\x02\x80\x03\xC0\x02\x40\x03",
        "i_pix_fmt": "RG10",
        "i_bayer_packed": (
            np.array(
                [
                    [0x000, 0x100, 0x040, 0x0C0],
                    [0x200, 0x3FF, 0x240, 0x3C0],
                    [0x080, 0x080, 0x0C0, 0x040],
                    [0x280, 0x380, 0x2C0, 0x340],
                ],
                dtype=np.uint16,
            ),
        ),
        "num_frames": 1,
        "i_frames": (
            b"\x00\x00\x00\x01\x40\x00\xc0\x00\x00\x02\xff\x03\x40\x02\xc0\x03\x80\x00\x80\x00\xc0\x00\x40\x00\x80\x02\x80\x03\xc0\x02\x40\x03",
        ),
        "o_pix_fmt": "rgb10.planar",
        "o_frames": (
            b"\x00\x00\x20\x00\x40\x00\x40\x00\x40\x00\x60\x00\x80\x00\x80\x00\x80\x00\xa0\x00\xc0\x00\xc0\x00\x80\x00\xa0\x00\xc0\x00\xc0\x00\x80\x01\x00\x01\x26\x01\xc0\x00\x00\x02\x70\x01\x40\x02\xd9\x00\xe6\x01\x80\x00\x70\x01\x40\x00\x80\x02\x33\x02\xc0\x02\x80\x01\xff\x03\xff\x03\xdf\x03\xc0\x03\xff\x03\xff\x03\xdf\x03\xc0\x03\xbf\x03\xbf\x03\x9f\x03\x80\x03\x80\x03\x80\x03\x60\x03\x40\x03",
        ),
        "output": b"YUV4MPEG2 W4 H12 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=rgb10.planar\nFRAME\n\x00\x00\x20\x00\x40\x00\x40\x00\x40\x00\x60\x00\x80\x00\x80\x00\x80\x00\xa0\x00\xc0\x00\xc0\x00\x80\x00\xa0\x00\xc0\x00\xc0\x00\x80\x01\x00\x01\x26\x01\xc0\x00\x00\x02\x70\x01\x40\x02\xd9\x00\xe6\x01\x80\x00\x70\x01\x40\x00\x80\x02\x33\x02\xc0\x02\x80\x01\xff\x03\xff\x03\xdf\x03\xc0\x03\xff\x03\xff\x03\xdf\x03\xc0\x03\xbf\x03\xbf\x03\x9f\x03\x80\x03\x80\x03\x80\x03\x60\x03\x40\x03",
    },
    # rgb8.planar->yuv444p8.mono
    {
        "name": "rgb8.planar-yuv444p8.mono",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H12 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=rgb8.planar\nFRAME\n\x05\x06\x07\x07\x09\x0a\x0b\x0b\x0d\x0e\x0f\x0f\x0d\x0e\x0f\x0f\x02\x01\x02\x03\x04\x05\x06\x06\x08\x09\x0a\x0b\x0c\x0c\x0e\x0c\x00\x00\x01\x02\x00\x00\x01\x02\x04\x04\x05\x06\x08\x08\x09\x0aFRAME\n\x15\x16\x17\x17\x19\x1a\x1b\x1b\x1d\x1e\x1f\x1f\x1d\x1e\x1f\x1f\x12\x11\x12\x13\x14\x15\x16\x16\x18\x19\x1a\x1b\x1c\x1c\x1e\x1c\x10\x10\x11\x12\x10\x10\x11\x12\x14\x14\x15\x16\x18\x18\x19\x1a",
        "num_frames": 2,
        "i_pix_fmt": "rgb8.planar",
        "i_frames": (
            b"\x05\x06\x07\x07\x09\x0a\x0b\x0b\x0d\x0e\x0f\x0f\x0d\x0e\x0f\x0f\x02\x01\x02\x03\x04\x05\x06\x06\x08\x09\x0a\x0b\x0c\x0c\x0e\x0c\x00\x00\x01\x02\x00\x00\x01\x02\x04\x04\x05\x06\x08\x08\x09\x0a",
            b"\x15\x16\x17\x17\x19\x1a\x1b\x1b\x1d\x1e\x1f\x1f\x1d\x1e\x1f\x1f\x12\x11\x12\x13\x14\x15\x16\x16\x18\x19\x1a\x1b\x1c\x1c\x1e\x1c\x10\x10\x11\x12\x10\x10\x11\x12\x14\x14\x15\x16\x18\x18\x19\x1a",
        ),
        "i_bayer_packed": (
            np.array(
                [
                    [5, 1, 7, 3],
                    [4, 0, 6, 2],
                    [13, 9, 15, 11],
                    [12, 8, 14, 10],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [21, 17, 23, 19],
                    [20, 16, 22, 18],
                    [29, 25, 31, 27],
                    [28, 24, 30, 26],
                ],
                dtype=np.uint8,
            ),
        ),
        "i_rgb_planar": (
            {
                "r": np.array(
                    [
                        [0x05, 0x06, 0x07, 0x07],
                        [0x09, 0x0A, 0x0B, 0x0B],
                        [0x0D, 0x0E, 0x0F, 0x0F],
                        [0x0D, 0x0E, 0x0F, 0x0F],
                    ],
                    dtype=np.uint8,
                ),
                "g": np.array(
                    [
                        [0x02, 0x01, 0x02, 0x03],
                        [0x04, 0x05, 0x06, 0x06],
                        [0x08, 0x09, 0x0A, 0x0B],
                        [0x0C, 0x0C, 0x0E, 0x0C],
                    ],
                    dtype=np.uint8,
                ),
                "b": np.array(
                    [
                        [0x00, 0x00, 0x01, 0x02],
                        [0x00, 0x00, 0x01, 0x02],
                        [0x04, 0x04, 0x05, 0x06],
                        [0x08, 0x08, 0x09, 0x0A],
                    ],
                    dtype=np.uint8,
                ),
            },
            {
                "r": np.array(
                    [
                        [0x15, 0x16, 0x17, 0x17],
                        [0x19, 0x1A, 0x1B, 0x1B],
                        [0x1D, 0x1E, 0x1F, 0x1F],
                        [0x1D, 0x1E, 0x1F, 0x1F],
                    ],
                    dtype=np.uint8,
                ),
                "g": np.array(
                    [
                        [0x12, 0x11, 0x12, 0x13],
                        [0x14, 0x15, 0x16, 0x16],
                        [0x18, 0x19, 0x1A, 0x1B],
                        [0x1C, 0x1C, 0x1E, 0x1C],
                    ],
                    dtype=np.uint8,
                ),
                "b": np.array(
                    [
                        [0x10, 0x10, 0x11, 0x12],
                        [0x10, 0x10, 0x11, 0x12],
                        [0x14, 0x14, 0x15, 0x16],
                        [0x18, 0x18, 0x19, 0x1A],
                    ],
                    dtype=np.uint8,
                ),
            },
        ),
        "o_bayer_packed": (
            np.array(
                [
                    [5, 1, 7, 3],
                    [4, 0, 6, 2],
                    [13, 9, 15, 11],
                    [12, 8, 14, 10],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [21, 17, 23, 19],
                    [20, 16, 22, 18],
                    [29, 25, 31, 27],
                    [28, 24, 30, 26],
                ],
                dtype=np.uint8,
            ),
        ),
        "o_pix_fmt": "yuv444p8.mono",
        "o_frames": (
            b"\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
            b"\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
        ),
        "output": b"YUV4MPEG2 W4 H12 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=yuv444p8.mono\nFRAME\n\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82FRAME\n\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
    },
    # bayer_bggr8->yuv444p8.mono
    {
        "name": "bayer_bggr8-yuv444p8.mono",
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=bayer_bggr8\nFRAME\n\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0fFRAME\n\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "i_pix_fmt": "bayer_bggr8",
        "num_frames": 2,
        "i_frames": (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        ),
        "i_bayer_packed": (
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
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
        "i_rgb_planar": (
            {
                "r": np.array(
                    [
                        [5, 6, 7, 7],
                        [9, 10, 11, 11],
                        [13, 14, 15, 15],
                        [13, 14, 15, 15],
                    ],
                    dtype=np.uint8,
                ),
                "g": np.array(
                    [
                        [2, 1, 2, 3],
                        [4, 5, 6, 6],
                        [8, 9, 10, 11],
                        [12, 12, 14, 12],
                    ],
                    dtype=np.uint8,
                ),
                "b": np.array(
                    [
                        [0, 0, 1, 2],
                        [0, 0, 1, 2],
                        [4, 4, 5, 6],
                        [8, 8, 9, 10],
                    ],
                    dtype=np.uint8,
                ),
            },
            {
                "r": np.array(
                    [
                        [21, 22, 23, 23],
                        [25, 26, 27, 27],
                        [29, 30, 31, 31],
                        [29, 30, 31, 31],
                    ],
                    dtype=np.uint8,
                ),
                "g": np.array(
                    [
                        [18, 17, 18, 19],
                        [20, 21, 22, 22],
                        [24, 25, 26, 27],
                        [28, 28, 30, 28],
                    ],
                    dtype=np.uint8,
                ),
                "b": np.array(
                    [
                        [16, 16, 17, 18],
                        [16, 16, 17, 18],
                        [20, 20, 21, 22],
                        [24, 24, 25, 26],
                    ],
                    dtype=np.uint8,
                ),
            },
        ),
        "o_pix_fmt": "yuv444p8.mono",
        "o_frames": (
            b"\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
            b"\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
        ),
        "output": b"YUV4MPEG2 W4 H12 F25:1 Ip A0:0 Cmono XCOLORRANGE=FULL XEXTCS=yuv444p8.mono\nFRAME\n\x03\x02\x03\x04\x05\x06\x07\x07\x09\x0a\x0b\x0c\x0c\x0c\x0e\x0d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82FRAME\n\x13\x12\x13\x14\x15\x16\x17\x17\x19\x1a\x1b\x1c\x1c\x1c\x1e\x1d\x7f\x7f\x7f\x7f\x7e\x7d\x7d\x7e\x7e\x7d\x7d\x7d\x7e\x7e\x7e\x7f\x82\x84\x84\x83\x84\x84\x84\x84\x84\x84\x84\x83\x81\x82\x81\x82",
    },
    {
        "name": "ydgcocg10.packed-yuv444p10.mono",
        "absolute_tolerance": 4,
        "debug": 0,
        "input": b"YUV4MPEG2 W4 H4 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=ydgcocg10.packed\nFRAME\n\xbf\x01\x80\x02\xc0\x01\xc0\x02\x00\x00\xc0\x01\x40\x00\xc0\x01\xc0\x01\x00\x03\xc0\x01\x40\x03\x80\x00\xc0\x01\xc0\x00\xc0\x01",
        "num_frames": 1,
        "i_pix_fmt": "ydgcocg10.packed",
        "i_frames": (
            b"\xbf\x01\x80\x02\xc0\x01\xc0\x02\x00\x00\xc0\x01\x40\x00\xc0\x01\xc0\x01\x00\x03\xc0\x01\x40\x03\x80\x00\xc0\x01\xc0\x00\xc0\x01",
        ),
        "i_bayer_planar": (
            {
                "R": np.array([[0, 64], [128, 192]], dtype=np.uint16),
                "G": np.array([[255, 192], [128, 64]], dtype=np.uint16),
                "g": np.array([[511, 576], [640, 704]], dtype=np.uint16),
                "B": np.array([[1023, 960], [896, 832]], dtype=np.uint16),
            },
        ),
        "i_rgb_planar": (
            {
                "r": np.array(
                    [
                        [0, 32, 64, 64],
                        [64, 96, 128, 128],
                        [128, 160, 192, 192],
                        [128, 160, 192, 192],
                    ],
                    dtype=np.uint16,
                ),
                "g": np.array(
                    [
                        [383, 255, 294, 192],
                        [511, 367, 576, 217],
                        [486, 128, 368, 64],
                        [640, 563, 704, 384],
                    ],
                    dtype=np.uint16,
                ),
                "b": np.array(
                    [
                        [1023, 1023, 991, 960],
                        [1023, 1023, 991, 960],
                        [959, 959, 927, 896],
                        [896, 896, 864, 832],
                    ],
                    dtype=np.uint16,
                ),
            },
        ),
        "o_pix_fmt": "yuv444p10.mono",
        "o_yuv_planar": (
            {
                "y": np.array(
                    [
                        [341, 275, 304, 241],
                        [435, 360, 489, 275],
                        [432, 232, 379, 197],
                        [516, 480, 569, 377],
                    ],
                    dtype=np.uint16,
                ),
                "u": np.array(
                    [
                        [847, 879, 849, 865],
                        [800, 837, 758, 848],
                        [770, 869, 781, 855],
                        [698, 716, 657, 735],
                    ],
                    dtype=np.uint16,
                ),
                "v": np.array(
                    [
                        [212, 298, 300, 356],
                        [186, 279, 195, 383],
                        [244, 448, 347, 507],
                        [171, 230, 181, 349],
                    ],
                    dtype=np.uint16,
                ),
            },
        ),
        "o_frames": (
            b"\x55\x01\x13\x01\x30\x01\xf1\x00\xb3\x01\x68\x01\xe9\x01\x13\x01\xb0\x01\xe8\x00\x7b\x01\xc5\x00\x04\x02\xe0\x01\x39\x02\x79\x01\x4f\x03\x6f\x03\x51\x03\x61\x03\x20\x03\x45\x03\xf6\x02\x50\x03\x02\x03\x65\x03\x0d\x03\x57\x03\xba\x02\xcc\x02\x91\x02\xdf\x02\xd4\x00\x2a\x01\x2c\x01\x64\x01\xba\x00\x17\x01\xc3\x00\x7f\x01\xf4\x00\xc0\x01\x5b\x01\xfb\x01\xab\x00\xe6\x00\xb5\x00\x5d\x01",
        ),
        "output": b"YUV4MPEG2 W4 H12 F25:1 Ip A0:0 Cmono10 XCOLORRANGE=FULL XEXTCS=yuv444p10.mono\nFRAME\n\x55\x01\x13\x01\x30\x01\xf1\x00\xb3\x01\x68\x01\xe9\x01\x13\x01\xb0\x01\xe8\x00\x7b\x01\xc5\x00\x04\x02\xe0\x01\x39\x02\x79\x01\x4f\x03\x6f\x03\x51\x03\x61\x03\x20\x03\x45\x03\xf6\x02\x50\x03\x02\x03\x65\x03\x0d\x03\x57\x03\xba\x02\xcc\x02\x91\x02\xdf\x02\xd4\x00\x2a\x01\x2c\x01\x64\x01\xba\x00\x17\x01\xc3\x00\x7f\x01\xf4\x00\xc0\x01\x5b\x01\xfb\x01\xab\x00\xe6\x00\xb5\x00\x5d\x01",
    },
]


class MainTest(itools_unittest.TestCase):

    def testVideoY4M(self):
        """video reading test."""
        function_name = "testVideoY4M"

        for test_case in self.getTestCases(function_name, readVideoY4MTestCases):
            print(f"...running forward \"{function_name}.{test_case['name']}\"")
            self.doTestVideoY4M(
                test_case["name"],
                test_case.get("absolute_tolerance", DEFAULT_ABSOLUTE_TOLERANCE),
                test_case["input"],
                test_case["num_frames"],
                test_case.get("i_frames", None),
                test_case.get("i_bayer_packed", None),
                test_case.get("i_bayer_planar", None),
                test_case.get("i_rgb_packed", None),
                test_case.get("i_rgb_planar", None),
                test_case.get("i_yuv_planar", None),
                test_case["o_pix_fmt"],
                test_case.get("o_frames", None),
                test_case.get("o_bayer_packed", None),
                test_case.get("o_bayer_planar", None),
                test_case.get("o_rgb_packed", None),
                test_case.get("o_rgb_planar", None),
                test_case.get("o_yuv_planar", None),
                test_case["output"],
                test_case["debug"],
            )
            if test_case.get("avoid_backwards", False):
                continue
            print(f"...running backwards \"{function_name}.{test_case['name']}\"")
            self.doTestVideoY4M(
                test_case["name"],
                test_case.get("absolute_tolerance", DEFAULT_ABSOLUTE_TOLERANCE),
                test_case["output"],
                test_case["num_frames"],
                test_case.get("o_frames", None),
                test_case.get("o_bayer_packed", None),
                test_case.get("o_bayer_planar", None),
                test_case.get("o_rgb_packed", None),
                test_case.get("o_rgb_planar", None),
                test_case.get("o_yuv_planar", None),
                test_case["i_pix_fmt"],
                test_case.get("i_frames", None),
                test_case.get("i_bayer_packed", None),
                test_case.get("i_bayer_planar", None),
                test_case.get("i_rgb_packed", None),
                test_case.get("i_rgb_planar", None),
                test_case.get("i_yuv_planar", None),
                test_case["input"],
                test_case["debug"],
            )

    def doTestBuffer(
        self, expected_buffer, bayer_image, absolute_tolerance, test_name, label
    ):
        buffer = bayer_image.GetBuffer()
        self.compareBuffer(
            buffer,
            expected_buffer,
            absolute_tolerance,
            label,
        )

    def doTestBayerPacked(
        self, expected_bayer_packed, bayer_image, absolute_tolerance, test_name, label
    ):
        bayer_packed = bayer_image.GetBayerPacked()
        np.testing.assert_allclose(
            bayer_packed,
            expected_bayer_packed,
            atol=absolute_tolerance,
            err_msg=f"error on {label} bayer_packed case {test_name}",
        )

    def doTestBayerPlanar(
        self, expected_bayer_planar, bayer_image, absolute_tolerance, test_name, label
    ):
        bayer_planar = bayer_image.GetBayerPlanar()
        self.comparePlanar(
            bayer_planar,
            expected_bayer_planar,
            absolute_tolerance,
            label,
        )

    def doTestRGBPacked(
        self, expected_rgb_packed, bayer_image, absolute_tolerance, test_name, label
    ):
        rgb_packed = bayer_image.GetRGBPacked()
        np.testing.assert_allclose(
            rgb_packed,
            expected_rgb_packed,
            atol=absolute_tolerance,
            err_msg=f"error on {label} rgb_packed case {test_name}",
        )

    def doTestRGBPlanar(
        self, expected_rgb_planar, bayer_image, absolute_tolerance, test_name, label
    ):
        rgb_planar = bayer_image.GetRGBPlanar()
        self.comparePlanar(
            rgb_planar,
            expected_rgb_planar,
            absolute_tolerance,
            label,
        )

    def doTestYUVPlanar(
        self, expected_yuv_planar, bayer_image, absolute_tolerance, test_name, label
    ):
        yuv_planar = bayer_image.GetYUVPlanar()
        self.comparePlanar(
            yuv_planar,
            expected_yuv_planar,
            absolute_tolerance,
            label,
        )

    def doTestFull(
        self,
        frame_id,
        test_frames,
        test_bayer_packed,
        test_bayer_planar,
        test_rgb_packed,
        test_rgb_planar,
        test_yuv_planar,
        bayer_image,
        absolute_tolerance,
        test_name,
        label,
    ):
        if test_frames is not None:
            self.doTestBuffer(
                test_frames[frame_id],
                bayer_image,
                absolute_tolerance,
                test_name,
                label,
            )
        if test_bayer_packed is not None:
            self.doTestBayerPacked(
                test_bayer_packed[frame_id],
                bayer_image,
                absolute_tolerance,
                test_name,
                label,
            )
        if test_bayer_planar is not None:
            self.doTestBayerPlanar(
                test_bayer_planar[frame_id],
                bayer_image,
                absolute_tolerance,
                test_name,
                label,
            )
        if test_rgb_packed is not None:
            self.doTestRGBPacked(
                test_rgb_packed[frame_id],
                bayer_image,
                absolute_tolerance,
                test_name,
                label,
            )
        if test_rgb_planar is not None:
            self.doTestRGBPlanar(
                test_rgb_planar[frame_id],
                bayer_image,
                absolute_tolerance,
                test_name,
                label,
            )
        if test_yuv_planar is not None:
            self.doTestYUVPlanar(
                test_yuv_planar[frame_id],
                bayer_image,
                absolute_tolerance,
                test_name,
                label,
            )

    def doTestY4MOutputFile(
        self, expected_output, output, absolute_tolerance, test_name
    ):
        if absolute_tolerance == 0:
            self.assertEqual(
                output,
                expected_output,
                f"error on output file test {test_name}",
            )
            return
        header, *frames = output.split(b"FRAME\n")
        expected_header, *expected_frames = expected_output.split(b"FRAME\n")
        assert len(frames) == len(
            expected_frames
        ), f"error on output file test {test_name}"
        self.assertEqual(
            header,
            expected_header,
            f"error on output file test {test_name}",
        )
        for i in range(len(frames)):
            self.compareBuffer(
                frames[i],
                expected_frames[i],
                absolute_tolerance,
                f"output file test {test_name}",
            )

    def doTestVideoY4M(
        self,
        test_name,
        absolute_tolerance,
        test_input,
        test_num_frames,
        test_i_frames,
        test_i_bayer_packed,
        test_i_bayer_planar,
        test_i_rgb_packed,
        test_i_rgb_planar,
        test_i_yuv_planar,
        test_o_pix_fmt,
        test_o_frames,
        test_o_bayer_packed,
        test_o_bayer_planar,
        test_o_rgb_packed,
        test_o_rgb_planar,
        test_o_yuv_planar,
        test_output,
        debug,
    ):
        # prepare input/output files
        infile = tempfile.NamedTemporaryFile(
            prefix="itools-bayer_unittest.infile.", suffix=".y4m"
        ).name
        with open(infile, "wb") as f:
            f.write(test_input)
        outfile = tempfile.NamedTemporaryFile(
            prefix="itools-bayer_unittest.outfile.", suffix=".y4m"
        ).name
        # read y4m file
        bayer_video_reader = itools_bayer_y4m.BayerY4MReader.FromY4MFile(infile, debug)
        bayer_video_writer = None
        num_frames = test_num_frames
        for frame_id in range(num_frames):
            # read the frame
            bayer_image = bayer_video_reader.GetFrame()
            # test the input
            self.doTestFull(
                frame_id,
                test_i_frames,
                test_i_bayer_packed,
                test_i_bayer_planar,
                test_i_rgb_packed,
                test_i_rgb_planar,
                test_i_yuv_planar,
                bayer_image,
                absolute_tolerance,
                test_name,
                "input",
            )

            # create the frame writer
            if bayer_video_writer is None:
                height = bayer_image.height
                width = bayer_image.width
                colorrange = bayer_video_reader.y4m_file_reader.input_colorrange
                o_pix_fmt = test_o_pix_fmt
                bayer_video_writer = itools_bayer_y4m.BayerY4MWriter.ToY4MFile(
                    outfile, height, width, colorrange, o_pix_fmt, debug
                )
            # write the frame
            bayer_image_out = bayer_video_writer.AddFrame(bayer_image)
            # test the output frame
            self.doTestFull(
                frame_id,
                test_o_frames,
                test_o_bayer_packed,
                test_o_bayer_planar,
                test_o_rgb_packed,
                test_o_rgb_planar,
                test_o_yuv_planar,
                bayer_image_out,
                absolute_tolerance,
                test_name,
                "output",
            )

        # ensure no more frames to read
        bayer_image = bayer_video_reader.GetFrame()
        assert bayer_image is None, f"error: found added frames"
        del bayer_video_writer
        # ensure outfile is correct
        with open(outfile, "rb") as f:
            output = f.read()
        # check the values
        expected_output = test_output
        self.doTestY4MOutputFile(expected_output, output, absolute_tolerance, test_name)


if __name__ == "__main__":
    itools_unittest.main(sys.argv)
