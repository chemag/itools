#!/usr/bin/env python3

"""itools-bayer_unittest.py: itools bayer unittest.

# runme
# $ ./itools-bayer_unittest.py
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

itools_bayer = importlib.import_module("itools-bayer")
itools_unittest = importlib.import_module("itools-unittest")


convertImageFormatTestCases = [
    # (a) component order
    {
        "name": "bayer_rggb8-bayer_rggb8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "bayer_rggb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "R": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "B": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
    },
    {
        "name": "bayer_rgbg8-bayer_bggr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rgbg8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "R": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "B": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "g": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x05\x02\x07\x04\x06\x01\x08\x03\x0d\x0a\x0f\x0c\x0e\x09\x10\x0b",
    },
    # (b) depth/pix_fmt changes
    {
        "name": "bayer_bggr8-bayer_bggr8.02",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
    },
    {
        "name": "bayer_bggr8-bayer_gbrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_gbrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x01\x04\x03\x06\x05\x08\x07\x0a\x09\x0c\x0b\x0e\x0d\x10\x0f",
    },
    {
        "name": "bayer_bggr8-bayer_gbrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_gbrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x01\x04\x03\x06\x05\x08\x07\x0a\x09\x0c\x0b\x0e\x0d\x10\x0f",
    },
    {
        "name": "bayer_bggr8-bayer_grbg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_grbg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x06\x04\x08\x01\x05\x03\x07\x0a\x0e\x0c\x10\x09\x0d\x0b\x0f",
    },
    {
        "name": "bayer_bggr8-bayer_ggbr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_ggbr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x05\x04\x07\x01\x06\x03\x08\x0a\x0d\x0c\x0f\x09\x0e\x0b\x10",
    },
    {
        "name": "bayer_bggr8-bayer_ggrb8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_ggrb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x02\x05\x04\x07\x06\x01\x08\x03\x0a\x0d\x0c\x0f\x0e\x09\x10\x0b",
    },
    {
        "name": "bayer_bggr8-bayer_rgbg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_rgbg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x06\x02\x08\x04\x01\x05\x03\x07\x0e\x0a\x10\x0c\x09\x0d\x0b\x0f",
    },
    {
        "name": "bayer_bggr8-bayer_bgrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bgrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x06\x05\x08\x07\x09\x0a\x0b\x0c\x0e\x0d\x10\x0f",
    },
    # bayer8->bayer16
    {
        "name": "bayer_bggr8-bayer_bggr16be",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16be",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10\x00",
    },
    {
        "name": "bayer_bggr8-bayer_bggr16be",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16be",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10\x00",
    },
    {
        "name": "bayer_bggr8-bayer_bggr16le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10",
    },
    # bayer10->bayer16 (extended)
    {
        "name": "RG10-bayer_rggb16le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10",  # SRGGB10
        "o_pix_fmt": "bayer_rggb16le",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "bayer_planar": {
            "R": np.array([[0x001, 0x205], [0x011, 0x215]], dtype=np.uint16),
            "G": np.array([[0x103, 0x307], [0x113, 0x317]], dtype=np.uint16),
            "g": np.array([[0x009, 0x20D], [0x019, 0x21D]], dtype=np.uint16),
            "B": np.array([[0x10B, 0x30F], [0x11B, 0x31F]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x001, 0x103, 0x205, 0x307],
                [0x009, 0x10B, 0x20D, 0x30F],
                [0x011, 0x113, 0x215, 0x317],
                [0x019, 0x11B, 0x21D, 0x31F],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x40\x00\xc0\x40\x40\x81\xc0\xc1\x40\x02\xc0\x42\x40\x83\xc0\xc3\x40\x04\xc0\x44\x40\x85\xc0\xc5\x40\x06\xc0\x46\x40\x87\xc0\xc7",
    },
    # bayer10 packed read/write
    {
        "name": "pRAA-pRAA",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "pRAA",  # SRGGB10P
        "o_pix_fmt": "pRAA",  # SRGGB10P
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
        "bayer_planar": {
            "R": np.array([[0x005, 0x00C], [0x02F, 0x034]], dtype=np.uint16),
            "G": np.array([[0x009, 0x010], [0x033, 0x038]], dtype=np.uint16),
            "g": np.array([[0x01A, 0x020], [0x040, 0x049]], dtype=np.uint16),
            "B": np.array([[0x01E, 0x024], [0x045, 0x04C]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x005, 0x009, 0x00C, 0x010],
                [0x01A, 0x01E, 0x020, 0x024],
                [0x02F, 0x033, 0x034, 0x038],
                [0x040, 0x045, 0x049, 0x04C],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
    },
    # bayer10->bayer16 (packed)
    {
        "name": "pRAA-bayer_bggr16le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "pRAA",  # SRGGB10P
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
        "bayer_planar": {
            "R": np.array([[0x005, 0x00C], [0x02F, 0x034]], dtype=np.uint16),
            "G": np.array([[0x009, 0x010], [0x033, 0x038]], dtype=np.uint16),
            "g": np.array([[0x01A, 0x020], [0x040, 0x049]], dtype=np.uint16),
            "B": np.array([[0x01E, 0x024], [0x045, 0x04C]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x005, 0x009, 0x00C, 0x010],
                [0x01A, 0x01E, 0x020, 0x024],
                [0x02F, 0x033, 0x034, 0x038],
                [0x040, 0x045, 0x049, 0x04C],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x80\x07\x40\x02\x00\x09\x00\x04\x80\x06\x40\x01\x00\x08\x00\x03\x40\x11\xc0\x0c\x00\x13\x00\x0e\x00\x10\xc0\x0b\x40\x12\x00\x0d",
    },
    # (c) planar bayer
    {
        "name": "bayer_rggb8.planar-bayer_rggb8.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8.planar",
        "o_pix_fmt": "bayer_rggb8.planar",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "R": np.array([[1, 2], [3, 4]], dtype=np.uint8),
            "G": np.array([[5, 6], [7, 8]], dtype=np.uint8),
            "g": np.array([[9, 10], [11, 12]], dtype=np.uint8),
            "B": np.array([[13, 14], [15, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 5, 2, 6],
                [9, 13, 10, 14],
                [3, 7, 4, 8],
                [11, 15, 12, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
    },
    {
        "name": "bayer_rggb8.planar-bayer_rggb8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8.planar",
        "o_pix_fmt": "bayer_rggb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "R": np.array([[1, 2], [3, 4]], dtype=np.uint8),
            "G": np.array([[5, 6], [7, 8]], dtype=np.uint8),
            "g": np.array([[9, 10], [11, 12]], dtype=np.uint8),
            "B": np.array([[13, 14], [15, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 5, 2, 6],
                [9, 13, 10, 14],
                [3, 7, 4, 8],
                [11, 15, 12, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x05\x02\x06\x09\x0d\x0a\x0e\x03\x07\x04\x08\x0b\x0f\x0c\x10",
    },
    {
        "name": "bayer_rggb8-bayer_rggb8.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "bayer_rggb8.planar",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar": {
            "R": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "B": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x01\x03\x09\x0b\x02\x04\x0a\x0c\x05\x07\x0d\x0f\x06\x08\x0e\x10",
    },
    # bayer10 (extended)
    {
        "name": "RG10-RG10",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10",
        "o_pix_fmt": "RG10",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "bayer_planar": {
            "R": np.array([[0x001, 0x205], [0x011, 0x215]], dtype=np.uint16),
            "G": np.array([[0x103, 0x307], [0x113, 0x317]], dtype=np.uint16),
            "g": np.array([[0x009, 0x20D], [0x019, 0x21D]], dtype=np.uint16),
            "B": np.array([[0x10B, 0x30F], [0x11B, 0x31F]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x001, 0x103, 0x205, 0x307],
                [0x009, 0x10B, 0x20D, 0x30F],
                [0x011, 0x113, 0x215, 0x317],
                [0x019, 0x11B, 0x21D, 0x31F],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
    },
    {
        "name": "RG10-RG10.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10",
        "o_pix_fmt": "RG10.planar",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "bayer_planar": {
            "R": np.array([[0x001, 0x205], [0x011, 0x215]], dtype=np.uint16),
            "G": np.array([[0x103, 0x307], [0x113, 0x317]], dtype=np.uint16),
            "g": np.array([[0x009, 0x20D], [0x019, 0x21D]], dtype=np.uint16),
            "B": np.array([[0x10B, 0x30F], [0x11B, 0x31F]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x001, 0x103, 0x205, 0x307],
                [0x009, 0x10B, 0x20D, 0x30F],
                [0x011, 0x113, 0x215, 0x317],
                [0x019, 0x11B, 0x21D, 0x31F],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x01\x00\x05\x02\x11\x00\x15\x02\x03\x01\x07\x03\x13\x01\x17\x03\x09\x00\x0d\x02\x19\x00\x1d\x02\x0b\x01\x0f\x03\x1b\x01\x1f\x03",
    },
    {
        "name": "RG10.planar-RG10",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10.planar",
        "o_pix_fmt": "RG10",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "bayer_planar": {
            "R": np.array([[0x001, 0x103], [0x205, 0x307]], dtype=np.uint16),
            "G": np.array([[0x009, 0x10B], [0x20D, 0x30F]], dtype=np.uint16),
            "g": np.array([[0x011, 0x113], [0x215, 0x317]], dtype=np.uint16),
            "B": np.array([[0x019, 0x11B], [0x21D, 0x31F]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x001, 0x009, 0x103, 0x10B],
                [0x011, 0x019, 0x113, 0x11B],
                [0x205, 0x20D, 0x307, 0x30F],
                [0x215, 0x21D, 0x317, 0x31F],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x01\x00\x09\x00\x03\x01\x0b\x01\x11\x00\x19\x00\x13\x01\x1b\x01\x05\x02\x0d\x02\x07\x03\x0f\x03\x15\x02\x1d\x02\x17\x03\x1f\x03",
    },
    {
        "name": "RG10.planar-RG10.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10.planar",
        "o_pix_fmt": "RG10.planar",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "bayer_planar": {
            "R": np.array([[0x001, 0x103], [0x205, 0x307]], dtype=np.uint16),
            "G": np.array([[0x009, 0x10B], [0x20D, 0x30F]], dtype=np.uint16),
            "g": np.array([[0x011, 0x113], [0x215, 0x317]], dtype=np.uint16),
            "B": np.array([[0x019, 0x11B], [0x21D, 0x31F]], dtype=np.uint16),
        },
        "bayer_packed": np.array(
            [
                [0x001, 0x009, 0x103, 0x10B],
                [0x011, 0x019, 0x113, 0x11B],
                [0x205, 0x20D, 0x307, 0x30F],
                [0x215, 0x21D, 0x317, 0x31F],
            ],
            dtype=np.uint16,
        ),
        "output": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
    },
    # (d) YDgCoCg conversions
    # (d.1) 8-bit
    {
        "name": "ydgcocg8.planar-ydgcocg8.planar-1",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "ydgcocg8.planar",
        "o_pix_fmt": "ydgcocg8.planar",
        "debug": 0,
        "input": b"\x6f\x70\x70\x70\xA0\xB0\xC0\xD0\x00\x10\x20\x30\x70\x70\x70\x70",
        "ydgcocg_planar": {
            "Y": np.array([[0x6F, 0x70], [0x70, 0x70]], dtype=np.uint8),
            "D": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint8),
            "C": np.array([[0x00, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "c": np.array([[0x70, 0x70], [0x70, 0x70]], dtype=np.uint8),
        },
        "ydgcocg_packed": np.array(
            [
                [0x6F, 0xA0, 0x70, 0xB0],
                [0x00, 0x70, 0x10, 0x70],
                [0x70, 0xC0, 0x70, 0xD0],
                [0x20, 0x70, 0x30, 0x70],
            ],
            dtype=np.uint8,
        ),
        "bayer_planar": {
            "R": np.array([[0x00, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "G": np.array([[0x40, 0x30], [0x20, 0x10]], dtype=np.uint8),
            "g": np.array([[0x80, 0x90], [0xA0, 0xB0]], dtype=np.uint8),
            "B": np.array([[0xFF, 0xF0], [0xE0, 0xD0]], dtype=np.uint8),
        },
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0xFF, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x6f\x70\x70\x70\xA0\xB0\xC0\xD0\x00\x10\x20\x30\x70\x70\x70\x70",
    },
    {
        "name": "bayer_rggb8-ydgcocg8.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "ydgcocg8.planar",
        "debug": 0,
        "input": b"\x00\x40\x10\x30\x80\xFF\x90\xF0\x20\x20\x30\x10\xA0\xE0\xB0\xD0",
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0xFF, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "bayer_planar": {
            "R": np.array([[0x00, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "G": np.array([[0x40, 0x30], [0x20, 0x10]], dtype=np.uint8),
            "g": np.array([[0x80, 0x90], [0xA0, 0xB0]], dtype=np.uint8),
            "B": np.array([[0xFF, 0xF0], [0xE0, 0xD0]], dtype=np.uint8),
        },
        "ydgcocg_planar": {
            "Y": np.array([[0x6F, 0x70], [0x70, 0x70]], dtype=np.uint8),
            "D": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint8),
            "C": np.array([[0x00, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "c": np.array([[0x70, 0x70], [0x70, 0x70]], dtype=np.uint8),
        },
        "output": b"\x6f\x70\x70\x70\xA0\xB0\xC0\xD0\x00\x10\x20\x30\x70\x70\x70\x70",
    },
    {
        "name": "bayer_rggb8-ydgcocg8.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "ydgcocg8.planar",
        "debug": 0,
        "input": b"\x00\x40\x10\x30\x80\x00\x90\xF0\x20\x20\x30\x10\xA0\xE0\xB0\xD0",
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0x00, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x30, 0x70], [0x70, 0x70]], dtype=np.uint8),
            "D": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint8),
            "C": np.array([[0x80, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "c": np.array([[0xB0, 0x70], [0x70, 0x70]], dtype=np.uint8),
        },
        "ydgcocg_packed": np.array(
            [
                [0x30, 0xA0, 0x70, 0xB0],
                [0x80, 0xB0, 0x10, 0x70],
                [0x70, 0xC0, 0x70, 0xD0],
                [0x20, 0x70, 0x30, 0x70],
            ],
            dtype=np.uint8,
        ),
        "output": b"\x30\x70\x70\x70\xA0\xB0\xC0\xD0\x80\x10\x20\x30\xB0\x70\x70\x70",
    },
    {
        "name": "bayer_rggb8.planar-ydgcocg8.planar-maxmin",
        "width": 16,
        "height": 4,
        "i_pix_fmt": "bayer_rggb8.planar",
        "o_pix_fmt": "ydgcocg8.planar",
        "debug": 0,
        "input": b"\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF",
        "bayer_planar": {
            "R": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "G": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
                    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "g": np.array(
                [
                    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF],
                    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "B": np.array(
                [
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                ],
                dtype=np.uint8,
            ),
        },
        "ydgcocg_planar": {
            "Y": np.array(
                [
                    [0x00, 0x3F, 0x3F, 0x7E, 0x3F, 0x7E, 0x7E, 0xBD],
                    [0x3F, 0x7E, 0x7E, 0xBD, 0x7E, 0xBD, 0xBD, 0xFC],
                ],
                dtype=np.uint8,
            ),
            "D": np.array(
                [
                    [0x80, 0x80, 0xFF, 0xFF, 0x00, 0x00, 0x80, 0x80],
                    [0x80, 0x80, 0xFF, 0xFF, 0x00, 0x00, 0x80, 0x80],
                ],
                dtype=np.uint8,
            ),
            "C": np.array(
                [
                    [0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00],
                    [0xFF, 0x80, 0xFF, 0x80, 0xFF, 0x80, 0xFF, 0x80],
                ],
                dtype=np.uint8,
            ),
            "c": np.array(
                [
                    [0x80, 0x40, 0xBF, 0x80, 0xBF, 0x80, 0xFF, 0xBF],
                    [0x40, 0x01, 0x80, 0x40, 0x80, 0x40, 0xBF, 0x80],
                ],
                dtype=np.uint8,
            ),
        },
        "output": b"\x00\x3f\x3f\x7e\x3f\x7e\x7e\xbd\x3f\x7e\x7e\xbd\x7e\xbd\xbd\xfc\x80\x80\xff\xff\x00\x00\x80\x80\x80\x80\xff\xff\x00\x00\x80\x80\x80\x00\x80\x00\x80\x00\x80\x00\xff\x80\xff\x80\xff\x80\xff\x80\x80\x40\xbf\x80\xbf\x80\xff\xbf\x40\x01\x80\x40\x80\x40\xbf\x80",
    },
    {
        "name": "ydgcocg8.planar-bayer_rggb8.planar-maxmin",
        "width": 16,
        "height": 4,
        "i_pix_fmt": "ydgcocg8.planar",
        "o_pix_fmt": "bayer_rggb8.planar",
        "debug": 0,
        "input": b"\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF",
        "ydgcocg_planar": {
            "Y": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "D": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
                    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "C": np.array(
                [
                    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF],
                    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "c": np.array(
                [
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                ],
                dtype=np.uint8,
            ),
        },
        "bayer_planar": {
            "R": np.array(
                [
                    [0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00],
                    [0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "G": np.array(
                [
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00],
                    [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "g": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x00, 0xFE],
                    [0x00, 0xFE, 0x00, 0xFE, 0xFE, 0xFF, 0xFE, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "B": np.array(
                [
                    [0xFF, 0x01, 0x01, 0x00, 0xFF, 0x01, 0x01, 0x00],
                    [0xFF, 0xFF, 0xFF, 0x01, 0xFF, 0xFF, 0xFF, 0x01],
                ],
                dtype=np.uint8,
            ),
        },
        "output": b"\x00\x00\xff\x00\x00\x00\xff\x00\xff\x00\xff\xff\xff\x00\xff\xff\x00\xff\x00\xff\x00\x00\x00\x00\xff\xff\xff\xff\x00\xff\x00\xff\x00\x00\x00\x00\x00\xfe\x00\xfe\x00\xfe\x00\xfe\xfe\xff\xfe\xff\xff\x01\x01\x00\xff\x01\x01\x00\xff\xff\xff\x01\xff\xff\xff\x01",
    },
    # (d.2) 10-bit
    {
        "name": "ydgcocg10.planar-ydgcocg10.planar",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "ydgcocg10.planar",
        "o_pix_fmt": "ydgcocg10.planar",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "ydgcocg_planar": {
            "Y": np.array([[0x001, 0x103], [0x205, 0x307]], dtype=np.uint16),
            "D": np.array([[0x009, 0x10B], [0x20D, 0x30F]], dtype=np.uint16),
            "C": np.array([[0x011, 0x113], [0x215, 0x317]], dtype=np.uint16),
            "c": np.array([[0x019, 0x11B], [0x21D, 0x31F]], dtype=np.uint16),
        },
        "bayer_planar": {
            "R": np.array([[0x000, 0x0FB], [0x1FD, 0x2FF]], dtype=np.uint16),
            "G": np.array([[0x011, 0x113], [0x215, 0x317]], dtype=np.uint16),
            "g": np.array([[0x000, 0x000], [0x22F, 0x3FF]], dtype=np.uint16),
            "B": np.array([[0x3D7, 0x2D5], [0x1D3, 0x0D1]], dtype=np.uint16),
        },
        "output": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
    },
    {
        "name": "ydgcocg8.planar-ydgcocg10.planar",
        "width": 16,
        "height": 4,
        "i_pix_fmt": "ydgcocg8.planar",
        "o_pix_fmt": "ydgcocg10.planar",
        "debug": 0,
        "input": b"\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF",
        "ydgcocg_planar": {
            "Y": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "D": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
                    [0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "C": np.array(
                [
                    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF],
                    [0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "c": np.array(
                [
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                    [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF],
                ],
                dtype=np.uint8,
            ),
        },
        "output": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfc\x03\xfc\x03\xfc\x03\xfc\x03\xfc\x03\xfc\x03\xfc\x03\xfc\x03\x00\x00\x00\x00\x00\x00\x00\x00\xfc\x03\xfc\x03\xfc\x03\xfc\x03\x00\x00\x00\x00\x00\x00\x00\x00\xfc\x03\xfc\x03\xfc\x03\xfc\x03\x00\x00\x00\x00\xfc\x03\xfc\x03\x00\x00\x00\x00\xfc\x03\xfc\x03\x00\x00\x00\x00\xfc\x03\xfc\x03\x00\x00\x00\x00\xfc\x03\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03\x00\x00\xfc\x03",
    },
    {
        "name": "ydgcocg10.planar-ydgcocg8.planar-maxmin",
        "width": 16,
        "height": 4,
        "i_pix_fmt": "ydgcocg10.planar",
        "o_pix_fmt": "ydgcocg8.planar",
        "debug": 0,
        "input": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x03\xff\x03\xff\x03\xff\x03\xff\x03\xff\x03\xff\x03\xff\x03\x00\x00\x00\x00\x00\x00\x00\x00\xff\x03\xff\x03\xff\x03\xff\x03\x00\x00\x00\x00\x00\x00\x00\x00\xff\x03\xff\x03\xff\x03\xff\x03\x00\x00\x00\x00\xff\x03\xff\x03\x00\x00\x00\x00\xff\x03\xff\x03\x00\x00\x00\x00\xff\x03\xff\x03\x00\x00\x00\x00\xff\x03\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03\x00\x00\xff\x03",
        "ydgcocg_planar": {
            "Y": np.array(
                [
                    [0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF, 0x3FF, 0x3FF, 0x3FF, 0x3FF],
                ],
                dtype=np.uint16,
            ),
            "D": np.array(
                [
                    [0x000, 0x000, 0x000, 0x000, 0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x000, 0x000, 0x000, 0x000, 0x3FF, 0x3FF, 0x3FF, 0x3FF],
                ],
                dtype=np.uint16,
            ),
            "C": np.array(
                [
                    [0x000, 0x000, 0x3FF, 0x3FF, 0x000, 0x000, 0x3FF, 0x3FF],
                    [0x000, 0x000, 0x3FF, 0x3FF, 0x000, 0x000, 0x3FF, 0x3FF],
                ],
                dtype=np.uint16,
            ),
            "c": np.array(
                [
                    [0x000, 0x3FF, 0x000, 0x3FF, 0x000, 0x3FF, 0x000, 0x3FF],
                    [0x000, 0x3FF, 0x000, 0x3FF, 0x000, 0x3FF, 0x000, 0x3FF],
                ],
                dtype=np.uint16,
            ),
        },
        "output": b"\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\x00\xFF\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF\x00\xFF",
    },
]


processColorConversions = [
    # (a) 8-bit, red/purple image
    {
        "name": "red-purple.8bits.4x4",
        "width": 4,
        "height": 4,
        "pix_fmt": "bayer_rggb8",
        "debug": 0,
        "input": b"\xFF\x00\xFF\x00\x00\x00\x00\x80\xFF\x00\xFF\x00\x00\x80\x00\xFF",
        "bayer_planar": {
            "R": np.array([[0xFF, 0xFF], [0xFF, 0xFF]], dtype=np.uint8),
            "G": np.array([[0x00, 0x00], [0x00, 0x00]], dtype=np.uint8),
            "g": np.array([[0x00, 0x00], [0x00, 0x00]], dtype=np.uint8),
            "B": np.array([[0x00, 0x80], [0x80, 0xFF]], dtype=np.uint8),
        },
        "rgb_planar": {
            "r": np.array(
                [
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    [0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
            "g": np.array(
                [
                    [0x00, 0x00, 0x00, 0x00],
                    [0x00, 0x00, 0x00, 0x00],
                    [0x00, 0x00, 0x00, 0x00],
                    [0x00, 0x00, 0x00, 0x00],
                ],
                dtype=np.uint8,
            ),
            "b": np.array(
                [
                    [0x00, 0x00, 0x40, 0x80],
                    [0x00, 0x00, 0x40, 0x80],
                    [0x40, 0x40, 0x7F, 0xBF],
                    [0x80, 0x80, 0xBF, 0xFF],
                ],
                dtype=np.uint8,
            ),
        },
        "yuv_planar": {
            "y": np.array(
                [
                    [0x4C, 0x4C, 0x54, 0x5B],
                    [0x4C, 0x4C, 0x54, 0x5B],
                    [0x54, 0x54, 0x5B, 0x62],
                    [0x5B, 0x5B, 0x62, 0x69],
                ],
                dtype=np.uint8,
            ),
            "u": np.array(
                [
                    [0x5B, 0x5B, 0x76, 0x92],
                    [0x5B, 0x5B, 0x76, 0x92],
                    [0x76, 0x76, 0x92, 0xAE],
                    [0x92, 0x92, 0xAE, 0xCA],
                ],
                dtype=np.uint8,
            ),
            "v": np.array(
                [
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    [0xFF, 0xFF, 0xFF, 0xFF],
                ],
                dtype=np.uint8,
            ),
        },
    },
    # (b) 10-bit, red/purple image
    {
        "name": "red-purple.10bits.4x4",
        "width": 4,
        "height": 4,
        "pix_fmt": "bayer_rggb10",
        "debug": 0,
        "input": b"\xFF\x03\x00\x00\xFF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\xFF\x03\x00\x00\xFF\x03\x00\x00\x00\x00\x00\x02\x00\x00\xFF\x03",
        "bayer_planar": {
            "R": np.array([[0x3FF, 0x3FF], [0x3FF, 0x3FF]], dtype=np.uint16),
            "G": np.array([[0x000, 0x000], [0x000, 0x000]], dtype=np.uint16),
            "g": np.array([[0x000, 0x000], [0x000, 0x000]], dtype=np.uint16),
            "B": np.array([[0x000, 0x200], [0x200, 0x3FF]], dtype=np.uint16),
        },
        "rgb_planar": {
            "r": np.array(
                [
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                ],
                dtype=np.uint16,
            ),
            "g": np.array(
                [
                    [0x000, 0x000, 0x000, 0x000],
                    [0x000, 0x000, 0x000, 0x000],
                    [0x000, 0x000, 0x000, 0x000],
                    [0x000, 0x000, 0x000, 0x000],
                ],
                dtype=np.uint16,
            ),
            "b": np.array(
                [
                    [0x000, 0x000, 0x100, 0x200],
                    [0x000, 0x000, 0x100, 0x200],
                    [0x100, 0x100, 0x1FF, 0x2FF],
                    [0x200, 0x200, 0x2FF, 0x3FF],
                ],
                dtype=np.uint16,
            ),
        },
        "yuv_planar": {
            "y": np.array(
                [
                    [0x131, 0x131, 0x14F, 0x16C],
                    [0x131, 0x131, 0x14F, 0x16C],
                    [0x14F, 0x14F, 0x16C, 0x189],
                    [0x16C, 0x16C, 0x189, 0x1A6],
                ],
                dtype=np.uint16,
            ),
            "u": np.array(
                [
                    [0x169, 0x169, 0x1D9, 0x248],
                    [0x169, 0x169, 0x1D9, 0x248],
                    [0x1D9, 0x1D9, 0x248, 0x2B7],
                    [0x248, 0x248, 0x2B7, 0x327],
                ],
                dtype=np.uint16,
            ),
            "v": np.array(
                [
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                    [0x3FF, 0x3FF, 0x3FF, 0x3FF],
                ],
                dtype=np.uint16,
            ),
        },
    },
]

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
        "debug": 0,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "ydgcocg8.planar",
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0xFF, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x6F, 0x70], [0x70, 0x70]], dtype=np.uint8),
            "D": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint8),
            "C": np.array([[0x00, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "c": np.array([[0x70, 0x70], [0x70, 0x70]], dtype=np.uint8),
        },
    },
    {
        "name": "rggb8-reverse-4x4",
        "debug": 0,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "ydgcocg8.planar",
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0x00, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint8,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x30, 0x70], [0x70, 0x70]], dtype=np.uint8),
            "D": np.array([[0xA0, 0xB0], [0xC0, 0xD0]], dtype=np.uint8),
            "C": np.array([[0x80, 0x10], [0x20, 0x30]], dtype=np.uint8),
            "c": np.array([[0xB0, 0x70], [0x70, 0x70]], dtype=np.uint8),
        },
    },
    {
        "name": "rggb10-forward-4x4",
        "debug": 0,
        "i_pix_fmt": "SRGGB10",
        "o_pix_fmt": "ydgcocg10.planar",
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0xFF, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint16,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x06F, 0x070], [0x070, 0x070]], dtype=np.uint16),
            "D": np.array([[0x220, 0x230], [0x240, 0x250]], dtype=np.uint16),
            "C": np.array([[0x180, 0x190], [0x1A0, 0x1B0]], dtype=np.uint16),
            "c": np.array([[0x1F0, 0x1F0], [0x1F0, 0x1F0]], dtype=np.uint16),
        },
    },
    {
        "name": "rggb10-reverse-4x4",
        "debug": 0,
        "i_pix_fmt": "SRGGB10",
        "o_pix_fmt": "ydgcocg10.planar",
        "bayer_packed": np.array(
            [
                [0x00, 0x40, 0x10, 0x30],
                [0x80, 0x00, 0x90, 0xF0],
                [0x20, 0x20, 0x30, 0x10],
                [0xA0, 0xE0, 0xB0, 0xD0],
            ],
            dtype=np.uint16,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x030, 0x070], [0x070, 0x070]], dtype=np.uint16),
            "D": np.array([[0x220, 0x230], [0x240, 0x250]], dtype=np.uint16),
            "C": np.array([[0x200, 0x190], [0x1A0, 0x1B0]], dtype=np.uint16),
            "c": np.array([[0x230, 0x1F0], [0x1F0, 0x1F0]], dtype=np.uint16),
        },
    },
    {
        "name": "rggb10-scaled-forward-4x4",
        "debug": 0,
        "i_pix_fmt": "SRGGB10",
        "o_pix_fmt": "ydgcocg10.planar",
        "bayer_packed": np.array(
            [
                [0x000, 0x100, 0x040, 0x0C0],
                [0x200, 0x3FF, 0x240, 0x3C0],
                [0x080, 0x080, 0x0C0, 0x040],
                [0x280, 0x380, 0x2C0, 0x340],
            ],
            dtype=np.uint16,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x1BF, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
            "D": np.array([[0x280, 0x2C0], [0x300, 0x340]], dtype=np.uint16),
            "C": np.array([[0x000, 0x040], [0x080, 0x0C0]], dtype=np.uint16),
            "c": np.array([[0x1C0, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
        },
    },
    {
        "name": "rggb10-scaled-reverse-4x4",
        "debug": 0,
        "i_pix_fmt": "SRGGB10",
        "o_pix_fmt": "ydgcocg10.planar",
        "bayer_packed": np.array(
            [
                [0x000, 0x100, 0x040, 0x0C0],
                [0x200, 0x000, 0x240, 0x3C0],
                [0x080, 0x080, 0x0C0, 0x040],
                [0x280, 0x380, 0x2C0, 0x340],
            ],
            dtype=np.uint16,
        ),
        "ydgcocg_planar": {
            "Y": np.array([[0x0C0, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
            "D": np.array([[0x280, 0x2C0], [0x300, 0x340]], dtype=np.uint16),
            "C": np.array([[0x200, 0x040], [0x080, 0x0C0]], dtype=np.uint16),
            "c": np.array([[0x2C0, 0x1C0], [0x1C0, 0x1C0]], dtype=np.uint16),
        },
    },
]


convertRg1g2bToRgbTestCases = [
    {
        "name": "bayer_rggb8-rggb8.planar.4x4",
        "debug": 0,
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "rgb8.planar",
        "bayer_planar": {
            "R": np.array([[0, 16], [32, 48]], dtype=np.uint8),
            "G": np.array([[64, 48], [32, 16]], dtype=np.uint8),
            "g": np.array([[128, 144], [160, 176]], dtype=np.uint8),
            "B": np.array([[255, 240], [224, 208]], dtype=np.uint8),
        },
        "rgb_planar": {
            "r": np.array(
                [
                    [0, 8, 16, 16],
                    [16, 24, 32, 32],
                    [32, 40, 48, 48],
                    [32, 40, 48, 48],
                ],
                dtype=np.uint8,
            ),
            "g": np.array(
                [
                    [96, 64, 73, 48],
                    [128, 92, 144, 54],
                    [121, 32, 92, 16],
                    [160, 140, 176, 96],
                ],
                dtype=np.uint8,
            ),
            "b": np.array(
                [
                    [255, 255, 247, 240],
                    [255, 255, 247, 240],
                    [239, 239, 231, 224],
                    [224, 224, 216, 208],
                ],
                dtype=np.uint8,
            ),
        },
    },
]


class MainTest(itools_unittest.TestCase):

    def testConvertImageFormat(self):
        """Simplest get_data test."""
        function_name = "testConvertImageFormat"
        for test_case in self.getTestCases(function_name, convertImageFormatTestCases):
            print(f"...running \"{function_name}.{test_case['name']}\"")
            # prepare input file
            infile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.infile.", suffix=".bin"
            ).name
            with open(infile, "wb") as f:
                f.write(test_case["input"])
            # prepare output file(s)
            outfile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.outfile.", suffix=".bin"
            ).name
            # prepare parameters
            i_pix_fmt = test_case["i_pix_fmt"]
            width = test_case["width"]
            height = test_case["height"]
            o_pix_fmt = test_case["o_pix_fmt"]
            debug = test_case["debug"]
            absolute_tolerance = 1

            # 1. read input image
            input_image = itools_bayer.BayerImage.FromFile(
                infile, i_pix_fmt, width, height, debug
            )

            # 2. check write back produces the same input file
            input_image.ToFile(outfile, debug)
            with open(outfile, "rb") as f:
                output = f.read()
            # check the values
            expected_output = test_case["input"]
            self.assertEqual(
                output,
                expected_output,
                f"error on input write test {test_case['name']}",
            )

            # 3. check the bayer_planar representation is correct
            if "bayer_planar" in test_case:
                expected_bayer_planar = test_case["bayer_planar"]
                bayer_planar = input_image.GetBayerPlanar()
                self.comparePlanar(
                    bayer_planar,
                    expected_bayer_planar,
                    absolute_tolerance,
                    f"bayer_planar {test_case['name']}",
                )

            # 4. check the bayer_packed representation is correct
            if "bayer_packed" in test_case:
                bayer_packed = input_image.GetBayerPacked()
                expected_bayer_packed = test_case["bayer_packed"]
                np.testing.assert_allclose(
                    bayer_packed,
                    expected_bayer_packed,
                    atol=absolute_tolerance,
                    err_msg=f"error on bayer_packed case {test_case['name']}",
                )

            # 5. check the ydgcocg plane is correct
            if "ydgcocg_planar" in test_case:
                expected_ydgcocg_planar = test_case["ydgcocg_planar"]
                ydgcocg_planar = input_image.GetYDgCoCgPlanar()
                self.comparePlanar(
                    ydgcocg_planar,
                    expected_ydgcocg_planar,
                    absolute_tolerance,
                    f"ydgcocg_planar {test_case['name']}",
                )

            # 6. check the ydgcocg plane is correct
            if "ydgcocg_packed" in test_case:
                expected_ydgcocg_packed = test_case["ydgcocg_packed"]
                ydgcocg_packed = input_image.GetYDgCoCgPacked()
                np.testing.assert_allclose(
                    ydgcocg_packed,
                    expected_ydgcocg_packed,
                    atol=absolute_tolerance,
                    err_msg=f"error on ydgcocg_packed case {test_case['name']}",
                )

            # 7. run forward conversion
            if "output" in test_case:
                output_image = input_image.Copy(o_pix_fmt, debug)
                # write it to a file
                output_image.ToFile(outfile, debug)
                with open(outfile, "rb") as f:
                    output = f.read()
                # check the values
                expected_output = test_case["output"]
                self.assertEqual(
                    output,
                    expected_output,
                    f"error on forward write test {test_case['name']}",
                )

    def testProcessColorConversions(self):
        """Test color conversions."""
        function_name = "testProcessColorConversions"
        for test_case in self.getTestCases(function_name, processColorConversions):
            print(f"...running \"{function_name}.{test_case['name']}\"")
            # prepare input file
            infile = tempfile.NamedTemporaryFile(
                prefix="itools-bayer_unittest.", suffix=".bin"
            ).name
            with open(infile, "wb") as f:
                f.write(test_case["input"])
            # prepare output file(s)
            # prepare parameters
            pix_fmt = test_case["pix_fmt"]
            width = test_case["width"]
            height = test_case["height"]
            debug = test_case["debug"]

            # 1. run forward conversion
            pix_fmt = itools_bayer.get_canonical_input_pix_fmt(pix_fmt)
            bayer_image = itools_bayer.BayerImage.FromFile(
                infile, pix_fmt, width, height, debug
            )

            # 2. check the Bayer planar representation is correct
            absolute_tolerance = 1
            expected_bayer_planar = test_case["bayer_planar"]
            bayer_planar = bayer_image.GetBayerPlanar()
            self.comparePlanar(
                bayer_planar,
                expected_bayer_planar,
                absolute_tolerance,
                test_case["name"],
            )

            # 3. check the RGB planar representation is correct
            expected_rgb_planar = test_case["rgb_planar"]
            rgb_planar = bayer_image.GetRGBPlanar()
            self.comparePlanar(
                rgb_planar, expected_rgb_planar, absolute_tolerance, test_case["name"]
            )

            # 4. check the YUV planar representation is correct
            expected_yuv_planar = test_case["yuv_planar"]
            yuv_planar = bayer_image.GetYUVPlanar()
            self.comparePlanar(
                yuv_planar, expected_yuv_planar, absolute_tolerance, test_case["name"]
            )

    def testClipIntegerAndScale(self):
        """clip_integer_and_scale test."""
        function_name = "testClipIntegerAndScale"
        for test_case in self.getTestCases(function_name, clipIntegerAndScaleTestCases):
            print(f"...running \"{function_name}.{test_case['name']}\"")
            arr = test_case["arr"]
            depth = test_case["depth"]
            expected_clipped_arr = test_case["clipped_arr"]
            # 1. run forward clipping
            clipped_arr = itools_bayer.clip_integer_and_scale(arr, depth)
            np.testing.assert_array_equal(
                clipped_arr,
                expected_clipped_arr,
                err_msg=f"error on forward case {test_case['name']}",
            )
            # 2. run backward clipping
            new_arr = itools_bayer.unclip_integer_and_unscale(clipped_arr, depth)
            absolute_tolerance = 1
            np.testing.assert_allclose(
                arr,
                new_arr,
                atol=absolute_tolerance,
                err_msg=f"error on forward case {test_case['name']}",
            )

    def testConvertRg1g2bToYdgcocg(self):
        """convert_rg1g2b_to_ydgcocg test."""
        function_name = "testConvertRg1g2bToYdgcocg"
        for test_case in self.getTestCases(
            function_name, convertRg1g2bToYdgcocgTestCases
        ):
            print(f"...running \"{function_name}.{test_case['name']}\"")
            i_pix_fmt = test_case["i_pix_fmt"]
            depth = itools_bayer.get_depth(i_pix_fmt)
            # 1. run RG1G2B to YDgCoCg function
            bayer_packed = test_case["bayer_packed"]
            debug = test_case["debug"]
            bayer_image = itools_bayer.BayerImage.FromBayerPacked(
                bayer_packed, i_pix_fmt
            )
            ydgcocg_planar = bayer_image.GetYDgCoCgPlanar()
            expected_ydgcocg_planar = test_case["ydgcocg_planar"]
            # check the values
            absolute_tolerance = 1
            self.comparePlanar(
                ydgcocg_planar,
                expected_ydgcocg_planar,
                absolute_tolerance,
                test_case["name"],
            )

            # 2. run YDgCoCg to RG1G2B function (reverse)
            o_pix_fmt = test_case["o_pix_fmt"]
            bayer_image_prime = itools_bayer.BayerImage.FromYDgCoCgPlanar(
                ydgcocg_planar, o_pix_fmt, debug
            )
            bayer_order = itools_bayer.get_order(i_pix_fmt)
            bayer_packed_prime = bayer_image_prime.GetBayerPacked(order=bayer_order)
            np.testing.assert_allclose(
                test_case["bayer_packed"],
                bayer_packed_prime,
                atol=absolute_tolerance,
                err_msg=f"error on forward case {test_case['name']}",
            )

    def testConvertRg1g2bToRgb(self):
        """convert_rg1g2b_to_rgb test."""
        function_name = "testConvertRg1g2bToRgb"
        for test_case in self.getTestCases(function_name, convertRg1g2bToRgbTestCases):
            print(f"...running \"{function_name}.{test_case['name']}\"")
            i_pix_fmt = test_case["i_pix_fmt"]
            depth = itools_bayer.get_depth(i_pix_fmt)
            # 1. run RG1G2B to RGB function
            bayer_planar = test_case["bayer_planar"]
            debug = test_case["debug"]
            bayer_image = itools_bayer.BayerImage.FromBayerPlanar(
                bayer_planar, i_pix_fmt
            )
            rgb_planar = bayer_image.GetRGBPlanar()
            expected_rgb_planar = test_case["rgb_planar"]
            # check the values
            absolute_tolerance = 1
            self.comparePlanar(
                rgb_planar,
                expected_rgb_planar,
                absolute_tolerance,
                test_case["name"],
            )

            # 2. run RGB to RG1G2B function (reverse)
            o_pix_fmt = test_case["o_pix_fmt"]
            bayer_image_prime = itools_bayer.BayerImage.FromRGBPlanar(
                rgb_planar, o_pix_fmt, debug
            )
            bayer_order = itools_bayer.get_order(i_pix_fmt)
            bayer_planar_prime = bayer_image_prime.GetBayerPlanar()
            expected_bayer_planar = test_case["bayer_planar"]
            self.comparePlanar(
                bayer_planar_prime,
                expected_bayer_planar,
                absolute_tolerance,
                f"error on forward case {test_case['name']}",
            )


if __name__ == "__main__":
    itools_unittest.main(sys.argv)
