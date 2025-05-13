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
        "i_pix_fmt": "bayer_rggb8",
        "o_pix_fmt": "bayer_rggb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "R": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "B": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-8x8.01",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_rgbg8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "R": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "B": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "g": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
    # bayer8->bayer16
    {
        "name": "basic-bayer_bggr8-bayer_bggr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_gbrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_gbrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_gbrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_gbrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_grbg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_grbg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_ggbr8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_ggbr8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_ggrb8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_ggrb8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_rgbg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_rgbg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-bayer_bggr8-bayer_bgrg8",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bgrg8",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-8x16.be.readable",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16be",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-8x16.be",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16be",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-8x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "bayer_bggr8",
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
        "bayer_planar_image": {
            "B": np.array([[1, 3], [9, 11]], dtype=np.uint8),
            "G": np.array([[2, 4], [10, 12]], dtype=np.uint8),
            "g": np.array([[5, 7], [13, 15]], dtype=np.uint8),
            "R": np.array([[6, 8], [14, 16]], dtype=np.uint8),
        },
        "bayer_packed_image": np.array(
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
        "name": "basic-extended10x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "RG10",  # SRGGB10
        "o_pix_fmt": "bayer_rggb16le",
        "debug": 0,
        "input": b"\x01\x00\x03\x01\x05\x02\x07\x03\x09\x00\x0b\x01\x0d\x02\x0f\x03\x11\x00\x13\x01\x15\x02\x17\x03\x19\x00\x1b\x01\x1d\x02\x1f\x03",
        "bayer_planar_image": {
            "R": np.array([[0x001, 0x205], [0x011, 0x215]], dtype=np.uint16),
            "G": np.array([[0x103, 0x307], [0x113, 0x317]], dtype=np.uint16),
            "g": np.array([[0x009, 0x20D], [0x019, 0x21D]], dtype=np.uint16),
            "B": np.array([[0x10B, 0x30F], [0x11B, 0x31F]], dtype=np.uint16),
        },
        "bayer_packed_image": np.array(
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
    # bayer10->bayer16 (packed)
    {
        "name": "basic-packed10x16.le",
        "width": 4,
        "height": 4,
        "i_pix_fmt": "pRAA",  # SRGGB10P
        "o_pix_fmt": "bayer_bggr16le",
        "debug": 0,
        "input": b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
        "bayer_planar_image": {
            "R": np.array([[0x005, 0x00C], [0x02F, 0x034]], dtype=np.uint16),
            "G": np.array([[0x009, 0x010], [0x033, 0x038]], dtype=np.uint16),
            "g": np.array([[0x01A, 0x020], [0x040, 0x049]], dtype=np.uint16),
            "B": np.array([[0x01E, 0x024], [0x045, 0x04C]], dtype=np.uint16),
        },
        "bayer_packed_image": np.array(
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
        "bayer_planar_image": {
            "R": np.array([[0xFF, 0xFF], [0xFF, 0xFF]], dtype=np.uint8),
            "G": np.array([[0x00, 0x00], [0x00, 0x00]], dtype=np.uint8),
            "g": np.array([[0x00, 0x00], [0x00, 0x00]], dtype=np.uint8),
            "B": np.array([[0x00, 0x80], [0x80, 0xFF]], dtype=np.uint8),
        },
        "rgb_planar_image": {
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
                    [0x00, 0x00, 0x40, 0x40],
                    [0x00, 0x00, 0x40, 0x40],
                    [0x40, 0x40, 0x80, 0x80],
                    [0x40, 0x40, 0x80, 0x80],
                ],
                dtype=np.uint8,
            ),
        },
        "yuv_planar_image": {
            "y": np.array(
                [
                    [0x4C, 0x4C, 0x54, 0x54],
                    [0x4C, 0x4C, 0x54, 0x54],
                    [0x54, 0x54, 0x5B, 0x5B],
                    [0x54, 0x54, 0x5B, 0x5B],
                ],
                dtype=np.uint8,
            ),
            "u": np.array(
                [
                    [0x5B, 0x5B, 0x76, 0x76],
                    [0x5B, 0x5B, 0x76, 0x76],
                    [0x76, 0x76, 0x92, 0x92],
                    [0x76, 0x76, 0x92, 0x92],
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
        "bayer_planar_image": {
            "R": np.array([[0x3FF, 0x3FF], [0x3FF, 0x3FF]], dtype=np.uint16),
            "G": np.array([[0x000, 0x000], [0x000, 0x000]], dtype=np.uint16),
            "g": np.array([[0x000, 0x000], [0x000, 0x000]], dtype=np.uint16),
            "B": np.array([[0x000, 0x200], [0x200, 0x3FF]], dtype=np.uint16),
        },
        "rgb_planar_image": {
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
                    [0x000, 0x000, 0x100, 0x100],
                    [0x000, 0x000, 0x100, 0x100],
                    [0x100, 0x100, 0x1FF, 0x1FF],
                    [0x100, 0x100, 0x1FF, 0x1FF],
                ],
                dtype=np.uint16,
            ),
        },
        "yuv_planar_image": {
            "y": np.array(
                [
                    [0x131, 0x131, 0x14F, 0x14F],
                    [0x131, 0x131, 0x14F, 0x14F],
                    [0x14F, 0x14F, 0x16C, 0x16C],
                    [0x14F, 0x14F, 0x16C, 0x16C],
                ],
                dtype=np.uint16,
            ),
            "u": np.array(
                [
                    [0x169, 0x169, 0x1D9, 0x1D9],
                    [0x169, 0x169, 0x1D9, 0x1D9],
                    [0x1D9, 0x1D9, 0x248, 0x248],
                    [0x1D9, 0x1D9, 0x248, 0x248],
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


class MainTest(unittest.TestCase):
    def testProcessColorConversions(self):
        """Test color conversions."""
        for test_case in processColorConversions:
            print("...running %s" % test_case["name"])
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
            # a. check if the dictionaries have the same keys
            expected_planar = test_case["bayer_planar_image"]
            planar = bayer_image.GetPlanar()
            assert set(expected_planar.keys()) == set(
                planar.keys()
            ), "Broken planar output"
            # b. check if the numpy arrays have the same values
            for key in expected_planar:
                np.testing.assert_allclose(
                    expected_planar[key],
                    planar[key],
                    atol=absolute_tolerance,
                    err_msg=f"error on forward case {key=} {test_case['name']}",
                )

            # 3. check the RGB planar representation is correct
            # a. check if the dictionaries have the same keys
            expected_planar = test_case["rgb_planar_image"]
            planar = bayer_image.GetRGBPlanar()
            assert set(expected_planar.keys()) == set(
                planar.keys()
            ), "Broken planar output"
            # b. check if the numpy arrays have the same values
            for key in expected_planar:
                np.testing.assert_allclose(
                    expected_planar[key],
                    planar[key],
                    atol=absolute_tolerance,
                    err_msg=f"error on forward case {key=} {test_case['name']}",
                )

            # 4. check the YUV planar representation is correct
            # a. check if the dictionaries have the same keys
            expected_planar = test_case["yuv_planar_image"]
            planar = bayer_image.GetYUVPlanar()
            assert set(expected_planar.keys()) == set(
                planar.keys()
            ), "Broken planar output"

            # b. check if the numpy arrays have the same values
            for key in expected_planar:
                np.testing.assert_allclose(
                    expected_planar[key],
                    planar[key],
                    atol=absolute_tolerance,
                    err_msg=f"error on forward case {key=} {test_case['name']}",
                )

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
            # prepare parameters
            i_pix_fmt = test_case["i_pix_fmt"]
            width = test_case["width"]
            height = test_case["height"]
            o_pix_fmt = test_case["o_pix_fmt"]
            debug = test_case["debug"]

            # 1. run forward conversion
            bayer_image = itools_bayer.convert_image_planar_mode(
                infile,
                i_pix_fmt,
                width,
                height,
                outfile,
                o_pix_fmt,
                debug,
            )
            # check the planar representation is correct
            absolute_tolerance = 1
            # a. check if the dictionaries have the same keys
            expected_planar = test_case["bayer_planar_image"]
            planar = bayer_image.GetPlanar()
            assert set(expected_planar.keys()) == set(
                planar.keys()
            ), "Broken planar output"
            # b. check if the numpy arrays have the same values
            for key in expected_planar:
                np.testing.assert_allclose(
                    expected_planar[key],
                    planar[key],
                    atol=absolute_tolerance,
                    err_msg=f"error on forward case {key=} {test_case['name']}",
                )

            # check the packed representation is correct
            np.testing.assert_allclose(
                test_case["bayer_packed_image"],
                bayer_image.GetPacked(),
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

            # 2. run output loop conversion (convert output to output)
            _ = itools_bayer.convert_image_planar_mode(
                outfile,
                o_pix_fmt,
                width,
                height,
                outfile,
                o_pix_fmt,
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

            # 3. run input loop conversion (convert input to input)
            _ = itools_bayer.convert_image_planar_mode(
                infile,
                i_pix_fmt,
                width,
                height,
                outfile,
                i_pix_fmt,
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
