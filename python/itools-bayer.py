#!/usr/bin/env python3

"""Module to convert (raw) Bayer (CFA) image pixel formats.

Supported formats:
* all ffmpeg formats
* all Linux V4L2 formats
* some MIPI-RAW formats.

Notes:
* ffmpeg only supports 8 Bayer formats (12 when considering that the 16-bit
  formats exist in both BE and LE flavors). We want to allow converting
  other Bayer formats to any of the ffmpeg ones. Main goal is to allow
  ffmpeg access to generic Bayer formats.
"""


import argparse
import cv2
import enum
import importlib
import numpy as np
import os
import sys

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")

__version__ = "0.1"

COLOR_COMPONENTS = set("RGgB")

# internal planar bayer image format is G1G2RB
DEFAULT_BAYER_PLANAR_ORDER = list("RGgB")
DEFAULT_YDGCOCG_PLANAR_ORDER = list("YDCc")

OPENCV_BAYER_FROM_CONVERSIONS = {
    "RGgB": cv2.COLOR_BayerRGGB2RGB,
    "GRBg": cv2.COLOR_BayerGRBG2RGB,
    "BGgR": cv2.COLOR_BayerBGGR2RGB,
    "GBRg": cv2.COLOR_BayerGBRG2RGB,
}

# dtype operation
# We use 2x dtype values
# * (1) st_dtype (storage dtype): This is uint8 for 8-bit Bayer, uint16 for
#   higher bit depths.
# * (2) op_dtype (operation dtype): This is int32 in all cases.
ST_DTYPE_8BIT = np.uint8
ST_DTYPE_16BIT = np.uint16
OP_DTYPE = np.int32

CV2_OPERATION_PIX_FMT_DICT = {
    8: "SRGGB8",
    10: "SRGGB10",
    12: "SRGGB12",
    14: "SRGGB14",
    16: "SRGGB16",
}


class LayoutType(enum.Enum):
    packed = 0
    planar = 1


class ComponentType(enum.Enum):
    bayer = 0
    ydgcocg = 1


# planar read/write functions
def rfun_planar(data, order, depth, width, height, debug):
    dtype = np.uint8 if depth == 8 else np.uint16
    element_size_bytes = 1 if depth == 8 else 2
    buffer_length = (width >> 1) * (height >> 1) * element_size_bytes
    planar = {
        order[0]: np.frombuffer(
            data[0 * buffer_length : 1 * buffer_length], dtype=dtype
        ).reshape((height >> 1, width >> 1)),
        order[1]: np.frombuffer(
            data[1 * buffer_length : 2 * buffer_length], dtype=dtype
        ).reshape((height >> 1, width >> 1)),
        order[2]: np.frombuffer(
            data[2 * buffer_length : 3 * buffer_length], dtype=dtype
        ).reshape((height >> 1, width >> 1)),
        order[3]: np.frombuffer(
            data[3 * buffer_length : 4 * buffer_length], dtype=dtype
        ).reshape((height >> 1, width >> 1)),
    }
    return planar


def wfun_planar(planar, order, debug):
    buffer = b""
    for plane_id in list(order):
        buffer += planar[plane_id].tobytes()
    return buffer


# component read/write functions


# 2 bytes -> 2 components
def rfun_8(data, debug):
    return data[0], data[1]


# 2 bytes -> 2 components
def wfun_8(c0, c1, debug):
    # make sure all the values are integers
    c0 = int(c0)
    c1 = int(c1)
    return int(c0).to_bytes(1, "big") + int(c1).to_bytes(1, "big")


# 4 bytes -> 2 components
def rfun_10_expanded_to_16(data, debug):
    # check the high 6 bits of both components are 0x0
    # 10-expanded-to-16 packs using
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    # |b7 |b6 |b5 |b4 |b3 |b2 |b1 |b0 | | 0 | 0 | 0 | 0 | 0 | 0 |b9 |b8 |
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    if (data[1] & 0xFC) != 0 or (data[3] & 0xFC) != 0:
        print("warn: upper 6 bits are not zero")
    c0 = (data[0]) | ((data[1] & 0x03) << 8)
    c1 = (data[2]) | ((data[3] & 0x03) << 8)
    return (c0, c1)


# 2 bytes -> 2 components
def wfun_10_expanded_to_16(c0, c1, debug):
    # make sure all the values are integers
    c0 = int(c0)
    c1 = int(c1)
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


# 5 bytes -> 4 components
def rfun_10_packed_expanded_to_16(data, debug):
    low = data[4]
    # 10-bit Bayer formats (packed) aka 4-in-5
    #   +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #   |A9 |A8 |A7 |A6 |A5 |A4 |A3 |A2 | |B9 |B8 |B7 |B6 |B5 |B4 |B3 |B2 |
    #   +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #   |C9 |C8 |C7 |C6 |C5 |C4 |C3 |C2 | |D9 |D8 |D7 |D6 |D5 |D4 |D3 |D2 |
    #   +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #   |D1 |D0 |C1 |C0 |B1 |B0 |A1 |A0 |
    #   +---+---+---+---+---+---+---+---+
    return (
        (data[0] << 2) | ((low >> 0) & 0x03),
        (data[1] << 2) | ((low >> 2) & 0x03),
        (data[2] << 2) | ((low >> 4) & 0x03),
        (data[3] << 2) | ((low >> 6) & 0x03),
    )


def wfun_10_packed_expanded_to_16(c0, c1, c2, c3, debug):
    # make sure all the values are integers
    c0 = int(c0)
    c1 = int(c1)
    c2 = int(c2)
    c3 = int(c3)
    main = ((c0 >> 2) << 24) | ((c1 >> 2) << 16) | ((c2 >> 2) << 8) | ((c3 >> 2) << 0)
    remaining = (
        ((c3 & 0x03) << 6)
        | ((c2 & 0x03) << 4)
        | ((c1 & 0x03) << 2)
        | ((c0 & 0x03) << 0)
    )
    return int(main).to_bytes(4, "big") + int(remaining).to_bytes(1, "big")


# 2 bytes -> 2 components
def rfun_10_alaw_expanded_to_16(data, debug):
    raise AssertionError("rfun_10_alaw_expanded_to_16: unimplemented")


def wfun_10_alaw_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_10_alaw_expanded_to_16: unimplemented")


# 2 bytes -> 2 components
def rfun_10_dpcm_expanded_to_16(data, debug):
    raise AssertionError("rfun_10_dpcm_expanded_to_16: unimplemented")


def wfun_10_dpcm_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_10_dpcm_expanded_to_16: unimplemented")


# 32 bytes -> 25 components
def rfun_10_ipu3_expanded_to_16(data, debug):
    raise AssertionError("rfun_10_ipu3_expanded_to_16: unimplemented")


def wfun_10_ipu3_expanded_to_16(carray, debug):
    raise AssertionError("wfun_10_ipu3_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_12_expanded_to_16(data, debug):
    # check the high 4 bits of both components are 0x0
    # 12-expanded-to-16 packs using
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    # |b7 |b6 |b5 |b4 |b3 |b2 |b1 |b0 | | 0 | 0 | 0 | 0 |b11|b10|b9 |b8 |
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    if (data[1] & 0xF0) != 0 or (data[3] & 0xF0) != 0:
        print("warn: upper 4 bits are not zero")
    return (
        data[0] | ((data[1] & 0x0F) << 8),
        data[2] | ((data[3] & 0x0F) << 8),
    )


def wfun_12_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_12_expanded_to_16: unimplemented")


# 3 bytes -> 2 components
def rfun_12_packed_expanded_to_16(data, debug):
    # 12-bit Bayer formats (packed) aka 4-in-5
    #   +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #   |A11|A10|A9 |A8 |A7 |A6 |A5 |A4 | |B11|B10|B9 |B8 |B7 |B6 |B5 |B4 |
    #   +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #   |B3 |B2 |B1 |B0 |A3 |A2 |A1 |A0 |
    #   +---+---+---+---+---+---+---+---+
    low = data[2]
    return (
        (data[0] << 4) | ((low >> 0) & 0x0F),
        (data[1] << 4) | ((low >> 4) & 0x0F),
    )


def wfun_12_packed_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_12_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_14_expanded_to_16(data, debug):
    # check the high 2 bits of both components are 0x0
    # 12-expanded-to-16 packs using
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    # |b7 |b6 |b5 |b4 |b3 |b2 |b1 |b0 | | 0 | 0 |b13|b12|b11|b10|b9 |b8 |
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    if (data[1] & 0xC0) != 0 or (data[3] & 0xC0) != 0:
        print("warn: upper 2 bits are not zero")
    return (
        data[0] | ((data[1] & 0x3F) << 8),
        data[2] | ((data[3] & 0x3F) << 8),
    )


def wfun_14_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_14_expanded_to_16: unimplemented")


# 7 bytes -> 4 components
def rfun_14_packed_expanded_to_16(data, debug):
    # 14-bit Bayer formats (packed) aka 4-in-7
    #  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #  |A13|A12|A11|A10|A9 |A8 |A7 |A6 | |B13|B12|B11|B10|B9 |B8 |B7 |B6 |
    #  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #  |C13|C12|C11|C10|C9 |C8 |C7 |C6 | |D13|D12|D11|D10|D9 |D8 |D7 |D6 |
    #  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #  |B1 |B0 |A5 |A4 |A3 |A2 |A1 |A0 | |C3 |C2 |C1 |C0 |B5 |B4 |B3 |B2 |
    #  +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    #  |D5 |D4 |D3 |D2 |D1 |D0 |C5 |C4 |
    #  +---+---+---+---+---+---+---+---+
    low0, low1, low2 = data[4:6]
    return (
        (data[0] << 6) | ((low0 >> 0) & 0x3F),
        (data[1] << 6) | ((low1 << 2) & 0x3C) | ((low0 >> 6) & 0x03),
        (data[2] << 6) | ((low2 << 4) & 0x30) | ((low1 >> 4) & 0x0F),
        (data[3] << 6) | ((low2 >> 2) & 0x3F),
    )


def wfun_14_packed_expanded_to_16(c0, c1, c2, c3, debug):
    raise AssertionError("wfun_14_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_16le(data, debug):
    return (
        (data[0] << 0) | (data[1] << 8),
        (data[2] << 0) | (data[3] << 8),
    )


def rfun_16be(data, debug):
    return (
        (data[1] << 0) | (data[0] << 8),
        (data[3] << 0) | (data[2] << 8),
    )


# 4 bytes -> 2 components
def wfun_16be(c0, c1, debug):
    # make sure all the values are integers
    c0 = int(c0)
    c1 = int(c1)
    return int(c0).to_bytes(2, "big") + int(c1).to_bytes(2, "big")


def wfun_16le(c0, c1, debug):
    # make sure all the values are integers
    c0 = int(c0)
    c1 = int(c1)
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


BAYER_FORMATS = {
    # 8-bit Bayer formats
    "bayer_bggr8": {
        "alias": (
            "BA81",
            "SBGGR8",
        ),
        # layout
        "layout": LayoutType.packed,
        # component order
        "order": "BGgR",
        # Some bayer formats use an intermediate "item" structure that gathers
        # components together. For example, the 10-bit Bayer formats (packed)
        # pack 4 components in 5 bytes. In this case, the item size will be
        # 5 bytes (blen=5), and it will contain 4 components (clen=4)
        # byte length: bytes per item read
        "blen": 2,
        # component length: components per item
        "clen": 2,
        # component depth (in bits)
        "depth": 8,
        # read function (component)
        "rfun": rfun_8,
        # write function (component)
        "wfun": wfun_8,
        # ffmpeg support
        "ffmpeg": True,
    },
    "bayer_rggb8": {
        "alias": (
            "RGGB",
            "SRGGB8",
        ),
        "layout": LayoutType.packed,
        "order": "RGgB",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_gbrg8": {
        "alias": (
            "GBRG",
            "SGBRG8",
        ),
        "layout": LayoutType.packed,
        "order": "GBRg",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_grbg8": {
        "alias": (
            "GRBG",
            "SGRBG8",
        ),
        "layout": LayoutType.packed,
        "order": "GRBg",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_ggbr8": {
        "layout": LayoutType.packed,
        "order": "GgBR",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_ggrb8": {
        "layout": LayoutType.packed,
        "order": "GgRB",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_rgbg8": {
        "layout": LayoutType.packed,
        "order": "RGBg",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_bgrg8": {
        "layout": LayoutType.packed,
        "order": "BGRg",
        "blen": 2,
        "clen": 2,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    # 10-bit Bayer formats expanded to 16 bits
    "RG10": {
        "alias": (
            "SRGGB10",
            "bayer_rggb10",
        ),
        "blen": 4,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "BA10": {
        "alias": (
            "SGRBG10",
            "bayer_grbg10",
        ),
        "blen": 4,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB10": {
        "alias": (
            "SGBRG10",
            "bayer_gbrg10",
        ),
        "blen": 4,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG10": {
        "alias": (
            "SBGGR10",
            "bayer_bggr10",
        ),
        "blen": 4,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats (packed)
    "pRAA": {
        "alias": ("SRGGB10P", "MIPI-RAW10-RGGB"),
        "blen": 5,
        "clen": 4,
        "depth": 10,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "pgAA": {
        "alias": ("SGRBG10P", "MIPI-RAW10-GRBG"),
        "blen": 5,
        "clen": 4,
        "depth": 10,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "pGAA": {
        "alias": ("SGBRG10P", "MIPI-RAW10-GBRG"),
        "blen": 5,
        "clen": 4,
        "depth": 10,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "pBAA": {
        "alias": ("SBGGR10P", "MIPI-RAW10-BGGR"),
        "blen": 5,
        "clen": 4,
        "depth": 10,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats compressed to 8 bits using a-law
    "aRA8": {
        "alias": ("SRGGB10ALAW8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "aBA8": {
        "alias": ("SBGGR10ALAW8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    "aGA8": {
        "alias": ("SGBRG10ALAW8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "agA8": {
        "alias": ("SGRBG10ALAW8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats compressed to 8 bits using dpcm
    "bRA8": {
        "alias": ("SRGGB10DPCM8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "bBA8": {
        "alias": ("SBGGR10DPCM8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    "bGA8": {
        "alias": ("SGBRG10DPCM8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BD10": {
        "alias": ("SGRBG10DPCM8",),
        "blen": 2,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats compressed a la Intel IPU3 driver
    "ip3r": {
        "alias": ("IPU3_SRGGB10",),
        "blen": 32,
        "clen": 25,
        "depth": 10,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "ip3b": {
        "alias": ("IPU3_SBGGR10",),
        "blen": 32,
        "clen": 25,
        "depth": 10,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    "ip3g": {
        "alias": ("IPU3_SGBRG10",),
        "blen": 32,
        "clen": 25,
        "depth": 10,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "ip3G": {
        "alias": ("IPU3_SGRBG10",),
        "blen": 32,
        "clen": 25,
        "depth": 10,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    # 12-bit Bayer formats expanded to 16 bits
    "RG12": {
        "alias": ("SRGGB12",),
        "blen": 4,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "BA12": {
        "alias": ("SGRBG12",),
        "blen": 4,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB12": {
        "alias": ("SGBRG12",),
        "blen": 4,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG12": {
        "alias": ("SBGGR12",),
        "blen": 4,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 12-bit Bayer formats (packed)
    "pRCC": {
        "alias": ("SRGGB12P",),
        "blen": 3,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "pgCC": {
        "alias": ("SGRBG12P",),
        "blen": 3,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "pGCC": {
        "alias": ("SGBRG12P",),
        "blen": 3,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "pBCC": {
        "alias": ("SBGGR12P",),
        "blen": 3,
        "clen": 2,
        "depth": 12,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 14-bit Bayer formats expanded to 16 bits
    "RG14": {
        "alias": ("SRGGB14",),
        "blen": 4,
        "clen": 2,
        "depth": 14,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "GR14": {
        "alias": ("SGRBG14",),
        "blen": 4,
        "clen": 2,
        "depth": 14,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB14": {
        "alias": ("SGBRG14",),
        "blen": 4,
        "clen": 2,
        "depth": 14,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG14": {
        "alias": ("SBGGR14",),
        "blen": 4,
        "clen": 2,
        "depth": 14,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 14-bit Bayer formats (packed)
    "pREE": {
        "alias": ("SRGGB14P",),
        "blen": 7,
        "clen": 4,
        "depth": 14,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "pgEE": {
        "alias": ("SGRBG14P",),
        "blen": 7,
        "clen": 4,
        "depth": 14,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "pGEE": {
        "alias": ("SGBRG14P",),
        "blen": 7,
        "clen": 4,
        "depth": 14,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "pBEE": {
        "alias": ("SBGGR14P",),
        "blen": 7,
        "clen": 4,
        "depth": 14,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 16-bit Bayer formats
    "bayer_bggr16le": {
        "alias": (
            "BA82",
            "BYR2",
            "SBGGR16",
        ),
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_rggb16le": {
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": True,
        "alias": (
            "RG16",
            "SRGGB16",
        ),
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_gbrg16le": {
        "alias": (
            "GB16",
            "SGBRG16",
        ),
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_grbg16le": {
        "alias": (
            "GR16",
            "SGRBG16",
        ),
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_bggr16be": {
        "layout": LayoutType.packed,
        "order": "BGgR",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_rggb16be": {
        "layout": LayoutType.packed,
        "order": "RGgB",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_gbrg16be": {
        "layout": LayoutType.packed,
        "order": "GBRg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_grbg16be": {
        "layout": LayoutType.packed,
        "order": "GRBg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "depth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    # planar
    "bayer_bggr8.planar": {
        "alias": (
            "BA81.planar",
            "SBGGR8.planar",
        ),
        "layout": LayoutType.planar,
        "order": "BGgR",
        "depth": 8,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "ffmpeg": False,
    },
    "bayer_rggb8.planar": {
        "alias": (
            "RGGB.planar",
            "SRGGB8.planar",
        ),
        "layout": LayoutType.planar,
        "order": "RGgB",
        "depth": 8,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "ffmpeg": False,
    },
    "bayer_gbrg8.planar": {
        "alias": (
            "GBRG.planar",
            "SGBRG8.planar",
        ),
        "layout": LayoutType.planar,
        "order": "GBRg",
        "depth": 8,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "ffmpeg": False,
    },
    "bayer_grbg8.planar": {
        "alias": (
            "GRBG.planar",
            "SGRBG8.planar",
        ),
        "layout": LayoutType.planar,
        "order": "GRBg",
        "depth": 8,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "ffmpeg": False,
    },
    "RG10.planar": {
        "alias": (
            "SRGGB10.planar",
            "bayer_rggb10.planar",
        ),
        "depth": 10,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "layout": LayoutType.planar,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "BA10.planar": {
        "alias": (
            "SGRBG10.planar",
            "bayer_grbg10.planar",
        ),
        "depth": 10,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "layout": LayoutType.planar,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB10.planar": {
        "alias": (
            "SGBRG10.planar",
            "bayer_gbrg10.planar",
        ),
        "depth": 10,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "layout": LayoutType.planar,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG10.planar": {
        "alias": (
            "SBGGR10.planar",
            "bayer_bggr10.planar",
        ),
        "depth": 10,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "layout": LayoutType.planar,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # non-RGGB Bayer
    "ydgcocg8.packed": {
        "layout": LayoutType.packed,
        "order": "YDCc",
        "blen": 1,
        "clen": 1,
        "depth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": False,
    },
    "ydgcocg8.planar": {
        "layout": LayoutType.planar,
        "order": "YDCc",
        "depth": 8,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "ffmpeg": False,
    },
    "ydgcocg10.packed": {
        "layout": LayoutType.packed,
        "order": "YDCc",
        "blen": 4,
        "clen": 2,
        "depth": 10,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "ffmpeg": False,
    },
    "ydgcocg10.planar": {
        "layout": LayoutType.planar,
        "order": "YDCc",
        "depth": 10,
        "rfun": rfun_planar,
        "wfun": wfun_planar,
        "ffmpeg": False,
    },
}

# calculate INPUT_FORMATS and OUTPUT_FORMATS
INPUT_FORMATS = {k: v for (k, v) in BAYER_FORMATS.items() if "rfun" in v}
OUTPUT_FORMATS = {k: v for (k, v) in BAYER_FORMATS.items() if "wfun" in v}
INPUT_CANONICAL_LIST = list(INPUT_FORMATS.keys())
INPUT_ALIAS_LIST = list(
    alias for v in INPUT_FORMATS.values() if "alias" in v for alias in v["alias"]
)
OUTPUT_CANONICAL_LIST = list(OUTPUT_FORMATS.keys())
OUTPUT_ALIAS_LIST = list(
    alias for v in OUTPUT_FORMATS.values() if "alias" in v for alias in v["alias"]
)

# full list of Bayer pixel formats
I_PIX_FMT_LIST = INPUT_CANONICAL_LIST + INPUT_ALIAS_LIST
O_PIX_FMT_LIST = OUTPUT_CANONICAL_LIST + OUTPUT_ALIAS_LIST


default_values = {
    "debug": 0,
    "dry_run": False,
    "i_pix_fmt": None,
    "o_pix_fmt": None,
    "width": 0,
    "height": 0,
    "infile": None,
    "outfile": None,
}


def get_canonical_input_pix_fmt(i_pix_fmt):
    # convert input pixel format to the canonical name
    if i_pix_fmt in INPUT_CANONICAL_LIST:
        return i_pix_fmt
    elif i_pix_fmt in INPUT_ALIAS_LIST:
        # find the canonical name
        for canonical, v in INPUT_FORMATS.items():
            if i_pix_fmt in v.get("alias", []):
                return canonical
    else:
        raise AssertionError(f"error: unknown input pix_fmt: {i_pix_fmt}")


def get_canonical_output_pix_fmt(o_pix_fmt):
    # convert output pixel format to the canonical name
    if o_pix_fmt in OUTPUT_CANONICAL_LIST:
        return o_pix_fmt
    elif o_pix_fmt in OUTPUT_ALIAS_LIST:
        # find the canonical name
        for canonical, v in OUTPUT_FORMATS.items():
            if o_pix_fmt in v.get("alias", []):
                o_pix_fmt = canonical
                break
        return o_pix_fmt
    else:
        raise AssertionError(f"error: unknown output pix_fmt: {o_pix_fmt}")


def get_depth(pix_fmt):
    pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
    return BAYER_FORMATS[pix_fmt]["depth"]


def get_order(pix_fmt):
    pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
    return BAYER_FORMATS[pix_fmt]["order"]


# 1. color conversions (Bayer-RGB)
# demosaic image
def bayer_packed_to_rgb_cv2_packed(bayer_packed, order, depth):
    # make sure that the packed representation is supported by by opencv
    assert (
        order in OPENCV_BAYER_FROM_CONVERSIONS
    ), f"error: invalid {pix_fmt=} which means {order=}. opencv only accepts order: {OPENCV_BAYER_FROM_CONVERSIONS.keys()}"
    # cv2 supports (among others) CV_8U and CV_16U types for color
    # conversions. These data types affect the conversions, and are
    # used when the input is an np.uint8 and np.uint16 numpy array,
    # respectively. This means that the color conversion for 10/12/14-bit
    # color need to first scale up to 16-bits.
    # use opencv to do the Bayer->RGB conversion from the packed
    if depth in (10, 12, 14):
        bayer_packed = bayer_packed << (16 - depth)
    # representation
    rgb_cv2_packed = cv2.cvtColor(bayer_packed, OPENCV_BAYER_FROM_CONVERSIONS[order])
    if depth in (10, 12, 14):
        rgb_cv2_packed = rgb_cv2_packed >> (16 - depth)
    return rgb_cv2_packed


# remosaic image
def rgb_cv2_packed_to_bayer_image(rgb_cv2_packed, pix_fmt):
    depth = get_depth(pix_fmt)
    st_dtype = np.uint8 if depth == 8 else np.uint16
    # 1. convert RGB to BGR
    bgr_image = cv2.cvtColor(rgb_cv2_packed, cv2.COLOR_RGB2BGR)
    height, width, _ = bgr_image.shape
    # 2. create a bayer_packed image
    bayer_packed = np.zeros((height, width), dtype=st_dtype)
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                bayer_packed[i, j] = bgr_image[i, j, 0]  # Blue
            else:
                if i % 2 == 0:
                    bayer_packed[i, j] = bgr_image[i, j, 1]  # Green
                else:
                    bayer_packed[i, j] = bgr_image[i, j, 2]  # Red
    return BayerImage.FromBayerPacked(bayer_packed, pix_fmt)


def rgb_planar_to_bayer_image(rgb_planar, pix_fmt):
    rgb_cv2_packed = rgb_planar_to_rgb_cv2_packed(rgb_planar)
    return rgb_cv2_packed_to_bayer_image(rgb_cv2_packed, pix_fmt)


# 2. color conversions (RGB-YUV)
def rgb_cv2_packed_to_yuv_cv2_packed(rgb_cv2_packed, depth):
    if depth in (10, 12, 14):
        rgb_cv2_packed = rgb_cv2_packed << (16 - depth)
    # use opencv to do the RGB->YUV conversion from the packed RGB
    # representation
    yuv_cv2_packed = cv2.cvtColor(rgb_cv2_packed, cv2.COLOR_RGB2YUV)
    if depth in (10, 12, 14):
        yuv_cv2_packed = yuv_cv2_packed >> (16 - depth)
    return yuv_cv2_packed


def rgb_planar_to_yuv_planar(rgb_planar, depth):
    rgb_cv2_packed = rgb_planar_to_rgb_cv2_packed(rgb_planar)
    yuv_cv2_packed = rgb_cv2_packed_to_yuv_cv2_packed(rgb_cv2_packed, depth)
    yuv_planar = yuv_cv2_packed_to_yuv_planar(yuv_cv2_packed)
    return yuv_planar


def yuv_planar_to_rgb_planar(yuv_planar, depth):
    yuv_cv2_packed = yuv_planar_to_yuv_cv2_packed(yuv_planar)
    rgb_cv2_packed = yuv_cv2_packed_to_rgb_cv2_packed(yuv_cv2_packed, depth)
    rgb_planar = rgb_cv2_packed_to_rgb_planar(rgb_cv2_packed)
    return rgb_planar


def yuv_cv2_packed_to_rgb_cv2_packed(yuv_cv2_packed, depth):
    if depth in (10, 12, 14):
        yuv_cv2_packed = yuv_cv2_packed << (16 - depth)
    # use opencv to do the YUV->RGB conversion from the packed YUV
    # representation
    rgb_cv2_packed = cv2.cvtColor(yuv_cv2_packed, cv2.COLOR_YUV2RGB)
    if depth in (10, 12, 14):
        rgb_cv2_packed = rgb_cv2_packed >> (16 - depth)
    return rgb_cv2_packed


# 3. color conversions (Bayer-YDgCoCg)
# Malvar Sullivan, "Progressive to Lossless Compression of Color Filter
# Array Images Using Macropixel Spectral Spatial Transformation", 2012
def rg1g2b_to_ydgcocg_planar(bayer_planar, depth):
    # 1. separate Bayer components
    bayer_r = bayer_planar["R"].astype(OP_DTYPE)
    bayer_g1 = bayer_planar["G"].astype(OP_DTYPE)
    bayer_g2 = bayer_planar["g"].astype(OP_DTYPE)
    bayer_b = bayer_planar["B"].astype(OP_DTYPE)

    # 2. do the color conversion
    # [ Y  ]   [ 1/4  1/4  1/4  1/4 ] [ G1 ]
    # [ Dg ] = [ -1    1    0    0  ] [ G4 ]
    # [ Co ]   [  0    0    1   -1  ] [ R2 ]
    # [ Cg ]   [ 1/2  1/2 -1/2 -1/2 ] [ B3 ]
    bayer_y = (bayer_g1 >> 2) + (bayer_g2 >> 2) + (bayer_r >> 2) + (bayer_b >> 2)
    bayer_dg = (-1) * bayer_g1 + (1) * bayer_g2
    bayer_co = (1) * bayer_r + (-1) * bayer_b
    bayer_cg = (bayer_g1 >> 1) + (bayer_g2 >> 1) - (bayer_r >> 1) - (bayer_b >> 1)

    # 3. clip matrices to storage type
    ydgcocg_planar = {
        "Y": clip_positive(bayer_y, depth),
        "D": clip_integer_and_scale(bayer_dg, depth),
        "C": clip_integer_and_scale(bayer_co, depth),
        "c": clip_integer_and_scale(bayer_cg, depth),
    }
    return ydgcocg_planar


def ydgcocg_to_rg1g2b_planar(ydgcocg_planar, depth):
    # 1. separate YDgCoCg components
    bayer_y = ydgcocg_planar["Y"]
    bayer_dg = ydgcocg_planar["D"]
    bayer_co = ydgcocg_planar["C"]
    bayer_cg = ydgcocg_planar["c"]

    # 2. unclip matrices from storage dtype
    bayer_y = unclip_positive(bayer_y, depth)
    bayer_dg = unclip_integer_and_unscale(bayer_dg, depth)
    bayer_co = unclip_integer_and_unscale(bayer_co, depth)
    bayer_cg = unclip_integer_and_unscale(bayer_cg, depth)

    # 3. do the color conversion
    # l = (1/4, 1/4, 1/4, 1/4, -1, 1, 0, 0, 0, 0, 1, -1, 1/2, 1/2, -1/2, -1/2)
    # matrix = np.array(l).reshape(4, 4)
    # np.linalg.inv(matrix)
    # array([[ 1. , -0.5, -0. ,  0.5],
    #        [ 1. ,  0.5,  0. ,  0.5],
    #        [ 1. ,  0. ,  0.5, -0.5],
    #        [ 1. ,  0. , -0.5, -0.5]])
    # [ G1 ]   [  1  -1/2   0   1/2 ] [ Y  ]
    # [ G4 ] = [  1   1/2   0   1/2 ] [ Dg ]
    # [ R2 ]   [  1    0   1/2 -1/2 ] [ Co ]
    # [ B3 ]   [  1    0  -1/2 -1/2 ] [ Cg ]
    bayer_g1 = bayer_y - (bayer_dg >> 1) + (bayer_cg >> 1)
    bayer_g2 = bayer_y + (bayer_dg >> 1) + (bayer_cg >> 1)
    bayer_r = bayer_y + (bayer_co >> 1) - (bayer_cg >> 1)
    bayer_b = bayer_y - (bayer_co >> 1) - (bayer_cg >> 1)

    # 4. round to storage dtype
    bayer_r = clip_positive(bayer_r, depth, check=False)
    bayer_g1 = clip_positive(bayer_g1, depth, check=False)
    bayer_g2 = clip_positive(bayer_g2, depth, check=False)
    bayer_b = clip_positive(bayer_b, depth, check=False)

    # 5. merge planes
    bayer_planar = {
        "R": bayer_r,
        "G": bayer_g1,
        "g": bayer_g2,
        "B": bayer_b,
    }
    return bayer_planar


# 4. packed-to-planar conversions
def rgb_cv2_packed_to_rgb_planar(rgb_cv2_packed):
    rgb_planar = {
        "r": rgb_cv2_packed[:, :, 0],
        "g": rgb_cv2_packed[:, :, 1],
        "b": rgb_cv2_packed[:, :, 2],
    }
    return rgb_planar


def rgb_planar_to_rgb_cv2_packed(rgb_planar):
    rgb_cv2_packed = np.dstack((rgb_planar["r"], rgb_planar["g"], rgb_planar["b"]))
    return rgb_cv2_packed


def yuv_cv2_packed_to_yuv_planar(yuv_cv2_packed):
    yuv_planar = {
        "y": yuv_cv2_packed[:, :, 0],
        "u": yuv_cv2_packed[:, :, 1],
        "v": yuv_cv2_packed[:, :, 2],
    }
    return yuv_planar


def yuv_planar_to_yuv_cv2_packed(yuv_planar):
    yuv_cv2_packed = np.dstack((yuv_planar["y"], yuv_planar["u"], yuv_planar["v"]))
    return yuv_cv2_packed


# 5. upsample/downsample conversions
def upsample_matrix(arr, shape):
    upsampled_array = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    height, width = shape
    return upsampled_array[:height, :width]


def yuv_subsample_planar(yuv_planar):
    yuv_subsampled_planar = yuv_planar.copy()
    yuv_subsampled_planar["u"] = yuv_subsampled_planar["u"][::2, ::2]
    yuv_subsampled_planar["v"] = yuv_subsampled_planar["v"][::2, ::2]
    return yuv_subsampled_planar


def yuv_upsample_planar(yuv_subsampled_planar):
    yuv_planar = yuv_subsampled_planar.copy()
    original_shape = yuv_planar["y"].shape
    yuv_planar["u"] = upsample_matrix(yuv_planar["u"], original_shape)
    yuv_planar["v"] = upsample_matrix(yuv_planar["v"], original_shape)
    return yuv_planar


# 6. matrix clip/unclip functions


# In our matrix operations, the OP_DTYPE is np.int32. After applying the
# matrix, we can have 2x situations:
#
# (a) positive value, in the [0, 2^{<depth>} - 1] range. This is what
# happens to the Y component, as it is calculated by shifting the R/G1/G2/B
# components to the right twice, and then add them.
#   +----+----+----+----+----+----+----+----+
#   |    |    |    |    |    |    |abcd|efgh| 8-bit
#   +----+----+----+----+----+----+----+----+
# In the 8-bit case, we clip the bits [a..h].
#   +----+----+----+----+----+----+----+----+
#   |    |    |    |    |    |  ab|cdef|ghij| 10-bit
#   +----+----+----+----+----+----+----+----+
# In the 10-bit case, we clip the bits [a..j].
#   +----+----+----+----+----+----+----+----+
#   |    |    |    |    |abcd|efgh|ijkl|mnop| 16-bit
#   +----+----+----+----+----+----+----+----+
# In the 10-bit case, we clip the bits [a..p].
def clip_positive(arr, depth, check=True):
    max_value = 2**depth - 1
    st_dtype = np.uint8 if depth == 8 else np.uint16
    # check for values outside the valid range
    if check and np.any((arr < 0) | (arr > max_value)):
        raise ValueError("Array contains values outside the valid range")
    # clip values to the closest integer and convert to storage dtype
    return np.clip(arr, 0, max_value).astype(st_dtype)


# (b) integer value, in the [-2^{<depth>}, 2^{<depth>} - 1] range. This is
# what happens to the Dg, Co, and Cg components, as they are calculated by
# adding/substracting the R/G1/G2/B components. Note that it is not possible
# to get away of the range.
#
# The solution in this case is to add a shift to center the value (positive
# or negative) around zero, and then throw the LSB in order to encode the
# right amount of bits.
#
# For example, for 8-bit components, the value of Dg/Co/Cg is in the
# [-255, 255] range. In order to encode it, we scale it to half (effectively
# throwing the LSB), and then add a shift. The value ends up in the range
# [0, 255].


def clip_integer_and_scale(arr, depth, check=True):
    max_value = 2**depth - 1
    min_value = -max_value
    shift = 2 ** (depth - 1)
    st_dtype = np.uint8 if depth == 8 else np.uint16
    # check for values outside the valid range
    if check and np.any((arr < min_value) | (arr > max_value)):
        raise ValueError("Array contains values outside the valid range")
    # scale and shift values to the storage dtype
    return ((arr >> 1) + shift).astype(st_dtype)


def unclip_positive(arr, depth):
    return arr.astype(OP_DTYPE)


def unclip_integer_and_unscale(arr, depth):
    shift = 2 ** (depth - 1)
    # unscale and convert
    return (arr.astype(OP_DTYPE) - shift) << 1


# main class
class BayerImage:

    @classmethod
    def IsBayer(cls, order):
        return set(order) == set("GgRB")

    @classmethod
    def IsYDgCoCg(cls, order):
        return set(order) == set("YDCc")

    @classmethod
    def GetComponentType(cls, order):
        if cls.IsBayer(order):
            return ComponentType.bayer
        elif cls.IsYDgCoCg(order):
            return ComponentType.ydgcocg
        else:
            raise AssertionError("error: invalid components: {order}")

    def __init__(
        self,
        infile,
        buffer,
        bayer_packed,
        bayer_planar,
        width,
        height,
        pix_fmt,
        debug=0,
    ):
        self.infile = infile
        self.buffer = buffer
        self.bayer_packed = bayer_packed
        self.bayer_planar = bayer_planar
        self.width = width
        self.height = height
        self.pix_fmt = pix_fmt
        pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
        self.layout = BAYER_FORMATS[pix_fmt]["layout"]
        self.depth = BAYER_FORMATS[pix_fmt]["depth"]
        self.order = BAYER_FORMATS[pix_fmt]["order"]
        self.component_type = self.GetComponentType(self.order)
        self.clen = BAYER_FORMATS[pix_fmt].get("clen", None)
        self.blen = BAYER_FORMATS[pix_fmt].get("blen", None)
        self.rfun = BAYER_FORMATS[pix_fmt].get("rfun", None)
        self.wfun = BAYER_FORMATS[pix_fmt].get("wfun", None)
        self.debug = debug
        self.rgb_cv2_packed = None
        self.rgb_planar = None
        self.yuv_cv2_packed = None
        self.yuv_planar = None
        self.ydgcocg_planar = None
        self.ydgcocg_packed = None

    # we have 5x representations
    #
    # YDgCoCgPacked <-                  -> BayerPacked
    #       ^         \               /        ^
    #       |          \            /          |
    #       |           -> Buffer<-            |
    #       |          /            \          |
    #       v         /               \        v
    # YDgCoCgPlanar <--------------------> BayerPlanar
    #
    # Note that the Bayer<->YDgCoCg conversion only occurs in planar
    # representation.

    # accessors
    def GetBuffer(self):
        if self.buffer is not None:
            pass
        elif self.IsBayer(self.order):
            if self.bayer_packed is not None:
                self.buffer = self.GetBufferFromBayerPacked()
            elif self.bayer_planar is not None:
                self.buffer = self.GetBufferFromBayerPlanar()
            else:
                raise AssertionError("error: invalid RGGB BayerImage")
        elif self.IsYDgCoCg(self.order):
            if self.ydgcocg_planar is not None:
                self.buffer = self.GetBufferFromYDgCoCgPlanar()
            elif self.ydgcocg_packed is not None:
                self.buffer = self.GetBufferFromYDgCoCgPacked()
            else:
                raise AssertionError("error: invalid YDgCoCg BayerImage")
        else:
            raise AssertionError("error: invalid BayerImage")
        return self.buffer

    def GetBayerPacked(self, order=None):
        if self.bayer_packed is not None:
            pass
        elif self.IsBayer(self.order):
            if self.buffer is not None:
                self.bayer_packed = self.GetBayerPackedFromBuffer()
            elif self.bayer_planar is not None:
                self.bayer_packed = self.GetBayerPackedFromBayerPlanar()
            else:
                raise AssertionError("error: invalid RGGB BayerImage")
        elif self.IsYDgCoCg(self.order):
            # BGGR<->YDgCoCg conversion always through planar
            self.ydgcocg_planar = self.GetYDgCoCgPlanar()
            self.bayer_planar = self.GetBayerPlanarFromYDgCoCgPlanar()
            self.bayer_packed = self.GetBayerPackedFromBayerPlanar(order)
        else:
            raise AssertionError("error: invalid BayerImage")
        return self.bayer_packed

    def GetBayerPlanar(self):
        if self.bayer_planar is not None:
            pass
        elif self.IsBayer(self.order):
            if self.buffer is not None:
                self.bayer_planar = self.GetBayerPlanarFromBuffer()
            elif self.bayer_packed is not None:
                self.bayer_planar = self.GetBayerPlanarFromBayerPacked()
            else:
                raise AssertionError("error: invalid RGGB BayerImage")
        elif self.IsYDgCoCg(self.order):
            # BGGR<->YDgCoCg conversion always through planar
            self.ydgcocg_planar = self.GetYDgCoCgPlanar()
            self.bayer_planar = self.GetBayerPlanarFromYDgCoCgPlanar()
        else:
            raise AssertionError("error: invalid BayerImage")
        return self.bayer_planar

    def GetYDgCoCgPlanar(self):
        if self.ydgcocg_planar is not None:
            return self.ydgcocg_planar
        elif self.IsYDgCoCg(self.order):
            if self.buffer is not None:
                self.ydgcocg_planar = self.GetYDgCoCgPlanarFromBuffer()
            elif self.ydgcocg_packed is not None:
                self.ydgcocg_planar = self.GetYDgCoCgPlanarFromYDgCoCgPacked()
            else:
                raise AssertionError("error: invalid YDgCoCg BayerImage")
        elif self.IsBayer(self.order):
            # BGGR<->YDgCoCg conversion always through planar
            self.bayer_planar = self.GetBayerPlanar()
            self.ydgcocg_planar = self.GetYDgCoCgPlanarFromBayerPlanar()
        else:
            raise AssertionError("error: invalid BayerImage")
        return self.ydgcocg_planar

    def GetYDgCoCgPacked(self):
        if self.ydgcocg_packed is not None:
            return self.ydgcocg_packed
        elif self.IsYDgCoCg(self.order):
            if self.buffer is not None:
                self.ydgcocg_packed = self.GetYDgCoCgPackedFromBuffer()
            elif self.ydgcocg_packed is not None:
                self.ydgcocg_packed = self.GetYDgCoCgPackedFromYDgCoCgPlanar()
            else:
                raise AssertionError("error: invalid YDgCoCg BayerImage")
        elif self.IsBayer(self.order):
            # BGGR<->YDgCoCg conversion always through planar
            self.bayer_planar = self.GetBayerPlanar()
            self.ydgcocg_planar = self.GetYDgCoCgPlanarFromBayerPlanar()
            self.ydgcocg_packed = self.GetYDgCoCgPackedFromYDgCoCgPlanar()
        else:
            raise AssertionError("error: invalid BayerImage")
        return self.ydgcocg_packed

    def GetRGBPlanar(self):
        if self.rgb_planar is not None:
            return self.rgb_planar
        rgb_cv2_packed = self.GetRGBCV2Packed()
        self.rgb_planar = rgb_cv2_packed_to_rgb_planar(rgb_cv2_packed)
        return self.rgb_planar

    def GetRGBCV2Packed(self):
        if self.rgb_cv2_packed is not None:
            return self.rgb_cv2_packed
        bayer_packed = self.GetBayerPacked()
        self.rgb_cv2_packed = bayer_packed_to_rgb_cv2_packed(
            bayer_packed, self.order, self.depth
        )
        return self.rgb_cv2_packed

    def GetYUVPlanar(self):
        if self.yuv_planar is not None:
            return self.yuv_planar
        yuv_cv2_packed = self.GetYUVCV2Packed()
        self.yuv_planar = yuv_cv2_packed_to_yuv_planar(yuv_cv2_packed)
        return self.yuv_planar

    def GetYUVCV2Packed(self):
        if self.yuv_cv2_packed is not None:
            return self.yuv_cv2_packed
        # cv2 supports (among others) CV_8U and CV_16U types for color
        # conversions. These data types affect the conversions, and are
        # used when the input is an np.uint8 and np.uint16 numpy array,
        # respectively. This means that the color conversion for 10/12/14-bit
        # color need to first scale up to 16-bits.
        rgb_cv2_packed = self.GetRGBCV2Packed()
        self.yuv_cv2_packed = rgb_cv2_packed_to_yuv_cv2_packed(
            rgb_cv2_packed, self.depth
        )
        return self.yuv_cv2_packed

    # converters: Generic packed/planar-Buffer
    def GetPackedFromBuffer(self):
        assert self.layout == LayoutType.packed, f"error: invalid call"
        # do the conversion directly
        # convert file buffer to packed
        # create bayer packed image
        dtype = np.uint16 if self.depth > 8 else np.uint8
        packed = np.zeros((self.height, self.width), dtype=dtype)
        # fill it up
        row = 0
        col = 0
        i = 0
        while True:
            if self.debug > 2:
                print(f"debug: {row=} {col=}")
            # 1. read components from the input
            components = ()
            while len(components) < self.clen:
                idata = self.buffer[i : i + self.blen]
                i += self.blen
                if not idata:
                    break
                components += self.rfun(idata, self.debug)
            if len(components) < self.clen:
                # end of input
                break
            if self.debug > 2:
                print(f"debug:  {components=}")
            # 2. convert component order
            for component in components:
                packed[row][col] = component
                if self.debug > 3:
                    print(f"debug: {row=} {col=}")
                col += 1
            # 3. update input row numbers
            if col == self.width:
                col = 0
                row += 1
        return packed

    def GetPlanarFromBuffer(self):
        assert self.layout == LayoutType.planar, f"error: invalid call"
        # do the conversion directly
        planar = rfun_planar(
            self.buffer, self.order, self.depth, self.width, self.height, self.debug
        )
        return planar

    def GetBufferFromPacked(self, packed):
        assert self.layout == LayoutType.packed, f"error: invalid call"
        row = 0
        col = 0
        buffer = bytearray()
        MAX_BUFFER = 500000
        while row < self.height:
            # 1. get components in order
            components = []
            for _ in range(self.clen):
                if self.debug > 2:
                    print(f"debug: {row=} {col=}")
                component = packed[row][col]
                components.append(component)
                col += 1
            if self.debug > 2:
                print(f"debug:  {components=}")
            # 2. write components to the output
            odata = self.wfun(*components[0 : self.clen], self.debug)
            buffer.extend(odata)
            # 3. update row numbers
            if col == self.width:
                col = 0
                row += 1
        self.buffer = bytes(buffer)
        return buffer

    def GetBufferFromPlanar(self, planar):
        assert self.layout == LayoutType.planar, f"error: invalid call"
        order = get_order(self.pix_fmt)
        self.buffer = wfun_planar(planar, order, self.debug)
        return self.buffer

    # converters: Bayer-Buffer
    def GetBayerPackedFromBuffer(self):
        assert self.IsBayer(self.order), f"error: invalid call"
        layout = BAYER_FORMATS[self.pix_fmt]["layout"]
        if self.layout == LayoutType.planar:
            # make sure we have the planar
            self.bayer_planar = self.GetBayerPlanarFromBuffer()
            # then get the packed
            self.bayer_packed = self.GetBayerPackedFromBayerPlanar()
            return self.bayer_packed
        # read the buffer
        self.bayer_packed = self.GetPackedFromBuffer()
        return self.bayer_packed

    def GetBayerPackedFromBayerPlanar(self, order=None):
        order = (
            order
            if order is not None
            else (
                self.order if self.IsBayer(self.order) else DEFAULT_BAYER_PLANAR_ORDER
            )
        )
        height, width = self.bayer_planar[order[0]].shape
        dtype = self.bayer_planar[order[0]].dtype
        bayer_packed = np.zeros((height * 2, width * 2), dtype=dtype)
        bayer_packed[0::2, 0::2] = self.bayer_planar[order[0]]
        bayer_packed[0::2, 1::2] = self.bayer_planar[order[1]]
        bayer_packed[1::2, 0::2] = self.bayer_planar[order[2]]
        bayer_packed[1::2, 1::2] = self.bayer_planar[order[3]]
        return bayer_packed

    def GetBayerPlanarFromBuffer(self):
        if self.layout == LayoutType.packed:
            # make sure we have the packed
            self.bayer_packed = self.GetBayerPackedFromBuffer()
            # then get the Planar
            self.bayer_planar = self.GetBayerPlanarFromBayerPacked()
            return self.bayer_planar
        # self.layout == LayoutType.planar:
        # do the conversion directly
        self.bayer_planar = self.GetPlanarFromBuffer()
        return self.bayer_planar

    def GetBayerPlanarFromBayerPacked(self):
        bayer_planar = {
            self.order[0]: self.bayer_packed[0::2, 0::2],
            self.order[1]: self.bayer_packed[0::2, 1::2],
            self.order[2]: self.bayer_packed[1::2, 0::2],
            self.order[3]: self.bayer_packed[1::2, 1::2],
        }
        return bayer_planar

    def GetBufferFromBayerPlanar(self):
        if self.layout == LayoutType.packed:
            # make sure we have the packed
            self.bayer_packed = self.GetBayerPackedFromBayerPlanar()
            # then get the Buffer
            self.buffer = self.GetBufferFromBayerPacked()
            return self.buffer
        # self.layout == LayoutType.planar:
        # do the conversion directly
        self.buffer = self.GetBufferFromPlanar(self.bayer_planar)
        return self.buffer

    def GetBufferFromBayerPacked(self):
        if self.layout == LayoutType.planar:
            # make sure we have the planar
            self.bayer_planar = self.GetBayerPlanarFromBayerPacked()
            # then get the Buffer
            self.buffer = self.GetBufferFromBayerPlanar()
            return self.buffer
        # self.layout == LayoutType.packed:
        # do the conversion directly
        self.buffer = self.GetBufferFromPacked(self.bayer_packed)
        return self.buffer

    # converters: YDgCoCg-Buffer
    def GetYDgCoCgPackedFromBuffer(self):
        assert self.IsYDgCoCg(self.order), f"error: invalid call"
        layout = BAYER_FORMATS[self.pix_fmt]["layout"]
        if self.layout == LayoutType.planar:
            # make sure we have the planar
            self.ydgcocg_planar = self.GetYDgCoCgPlanarFromBuffer()
            # then get the packed
            self.ydgcocg_packed = self.GetYDgCoCgPackedFromYDgCoCgPlanar()
            return self.ydgcocg_packed
        # read the buffer
        self.ydgcocg_packed = self.GetPackedFromBuffer()
        return self.ydgcocg_packed

    def GetYDgCoCgPackedFromYDgCoCgPlanar(self):
        order = (
            self.order if self.IsYDgCoCg(self.order) else DEFAULT_YDGCOCG_PLANAR_ORDER
        )
        height, width = self.ydgcocg_planar[order[0]].shape
        dtype = self.ydgcocg_planar[order[0]].dtype
        ydgcocg_packed = np.zeros((height * 2, width * 2), dtype=dtype)
        ydgcocg_packed[0::2, 0::2] = self.ydgcocg_planar[order[0]]
        ydgcocg_packed[0::2, 1::2] = self.ydgcocg_planar[order[1]]
        ydgcocg_packed[1::2, 0::2] = self.ydgcocg_planar[order[2]]
        ydgcocg_packed[1::2, 1::2] = self.ydgcocg_planar[order[3]]
        return ydgcocg_packed

    def GetYDgCoCgPlanarFromBuffer(self):
        if self.layout == LayoutType.packed:
            # make sure we have the packed
            self.ydgcocg_packed = self.GetYDgCoCgPackedFromBuffer()
            # then get the Planar
            self.ydgcocg_planar = self.GetYDgCoCgPlanarFromYDgCoCgPacked()
            return self.ydgcocg_planar
        # self.layout == LayoutType.planar:
        # do the conversion directly
        self.ydgcocg_planar = self.GetPlanarFromBuffer()
        return self.ydgcocg_planar

    def GetYDgCoCgPlanarFromYDgCoCgPacked(self):
        ydgcocg_planar = {
            self.order[0]: self.ydgcocg_packed[0::2, 0::2],
            self.order[1]: self.ydgcocg_packed[0::2, 1::2],
            self.order[2]: self.ydgcocg_packed[1::2, 0::2],
            self.order[3]: self.ydgcocg_packed[1::2, 1::2],
        }
        return ydgcocg_planar

    def GetBufferFromYDgCoCgPlanar(self):
        if self.layout == LayoutType.packed:
            # make sure we have the packed
            self.ydgcocg_packed = self.GetYDgCoCgPackedFromYDgCoCgPlanar()
            # then get the Buffer
            self.buffer = self.GetBufferFromYDgCoCgPacked()
            return self.buffer
        # self.layout == LayoutType.planar:
        # do the conversion directly
        self.buffer = self.GetBufferFromPlanar(self.ydgcocg_planar)
        return self.buffer

    def GetBufferFromYDgCoCgPacked(self):
        if self.layout == LayoutType.planar:
            # make sure we have the planar
            self.ydgcocg_planar = self.GetYDgCoCgPlanarFromYDgCoCgPacked()
            # then get the Buffer
            self.buffer = self.GetBufferFromYDgCoCgPlanar()
            return self.buffer
        # self.layout == LayoutType.packed:
        # do the conversion directly
        self.buffer = self.GetBufferFromPacked(self.ydgcocg_packed)
        return self.buffer

    # converters: Bayer-YDgCoCg
    def GetBayerPlanarFromYDgCoCgPlanar(self):
        self.bayer_planar = ydgcocg_to_rg1g2b_planar(self.ydgcocg_planar, self.depth)
        return self.bayer_planar

    def GetYDgCoCgPlanarFromBayerPlanar(self):
        self.ydgcocg_planar = rg1g2b_to_ydgcocg_planar(self.bayer_planar, self.depth)
        return self.ydgcocg_planar

    # other methods
    @classmethod
    def GetBayerPlanarOrder(cls, planar_order):
        assert (
            set(planar_order) == COLOR_COMPONENTS
        ), f"error: invalid Bayer components {planar_order}"
        return planar_order

    @classmethod
    def GetPlaneIds(cls, planar_order, row):
        planar_order = cls.GetBayerPlanarOrder(planar_order)
        plane_ids = planar_order[0:2] if row % 2 == 0 else planar_order[2:4]
        return plane_ids

    def ToY4MFile(self, outfile, debug):
        colorspace = "mono" if self.depth == 8 else "mono10"
        width = self.width >> 1
        height = self.height << 1
        outyvu = np.frombuffer(self.GetBuffer(), dtype="<u2").reshape((height, width))
        itools_y4m.write_y4m_image(
            outfile,
            outyvu,
            colorspace=colorspace,
            colorrange=itools_common.ColorRange.full,
            extcs=self.pix_fmt,
        )

    def ToFile(self, outfile, debug):
        if os.path.splitext(outfile)[1] == ".y4m":
            self.ToY4MFile(outfile, debug)
        else:
            with open(outfile, "wb") as fout:
                fout.write(self.GetBuffer())

    # create a new BayerImage from a depth-aware copy of another
    def Copy(self, o_pix_fmt, debug):
        # 1. select the input plane
        i_component_type = self.component_type
        if i_component_type == ComponentType.bayer:
            i_planar = self.GetBayerPlanar()
        elif i_component_type == ComponentType.ydgcocg:
            i_planar = self.GetYDgCoCgPlanar()
        # 2. convert planar to destination depth
        i_depth = get_depth(self.pix_fmt)
        o_depth = get_depth(o_pix_fmt)
        o_planar = self.CopyPlanar(i_planar, i_depth, o_depth)
        # 3. convert planar to color space
        o_order = get_order(o_pix_fmt)
        o_component_type = self.GetComponentType(o_order)
        if (
            i_component_type == ComponentType.bayer
            and o_component_type == ComponentType.ydgcocg
        ):
            o_planar = rg1g2b_to_ydgcocg_planar(o_planar, o_depth)
        elif (
            i_component_type == ComponentType.ydgcocg
            and o_component_type == ComponentType.bayer
        ):
            o_planar = ydgcocg_to_rg1g2b_planar(o_planar, o_depth)
        # 4. use the new plane to create the new image
        if o_component_type == ComponentType.bayer:
            image_copy = BayerImage.FromBayerPlanar(o_planar, o_pix_fmt, debug)
        elif o_component_type == ComponentType.ydgcocg:
            image_copy = BayerImage.FromYDgCoCgPlanar(o_planar, o_pix_fmt, debug)
        return image_copy

    # depth-aware planar copy
    def CopyPlanar(self, i_planar, i_depth, o_depth):
        o_planar = i_planar.copy()
        o_dtype = np.uint16 if o_depth > 8 else np.uint8
        if i_depth > o_depth:
            o_planar = {
                k: (v >> (i_depth - o_depth)).astype(o_dtype)
                for k, v in o_planar.items()
            }
        elif i_depth < o_depth:
            o_planar = {
                k: (v.astype(o_dtype) << (o_depth - i_depth))
                for k, v in o_planar.items()
            }
        return o_planar

    # factory methods
    @classmethod
    def FromY4MFile(cls, infile, debug=0):
        # check whether the image is self-describing
        frame, header, offset, status = itools_y4m.read_y4m_image(
            infile, output_colorrange=None, cleanup=0, logfd=sys.stdout, debug=debug
        )
        if header.colorspace in ("mono", "mono10"):
            # check that the image is annotated
            assert (
                "EXTCS" in header.comment
            ), f"error: monochrome image does not contain extended color space (EXTCS)"
            i_pix_fmt = header.comment["EXTCS"]
            i_pix_fmt = get_canonical_input_pix_fmt(i_pix_fmt)
            assert (
                i_pix_fmt in BAYER_FORMATS
            ), f"error: unknown extended color space: {i_pix_fmt}"
            # (height, width) plane
            frame = frame[:, :, 0]
            # process the frame as packed/planar
            layout = BAYER_FORMATS[i_pix_fmt]["layout"]
            bayer_formats = BAYER_FORMATS
            order = BAYER_FORMATS[i_pix_fmt]["order"]
            if layout == LayoutType.packed:
                if BayerImage.IsBayer(order):
                    return BayerImage.FromBayerPacked(frame, i_pix_fmt, debug)
                else:  # BayerImage.IsYDgCoCg(order):
                    raise AssertionError("error: unimplemented BayerImage.IsYDgCoCg()")
            elif layout == LayoutType.planar:
                height, width = frame.shape
                assert height % 4 == 0, f"error: invalid height: {height}"
                planar = dict(zip(order, np.split(frame, 4, axis=0)))
                if BayerImage.IsBayer(order):
                    return BayerImage.FromBayerPlanar(planar, i_pix_fmt, debug)
                else:  # BayerImage.IsYDgCoCg(order):
                    return BayerImage.FromYDgCoCgPlanar(planar, i_pix_fmt, debug)
        elif header.colorspace in ("yuv420", "yuv420p10", "yuv444", "yuv444p10"):
            raise AssertionError("error: unimplemented yuv/rgb reading")
        raise AssertionError(f"error: invalid y4m colorspace: {header.colorspace}")

    @classmethod
    def FromFile(cls, infile, pix_fmt, width, height, debug=0):
        # check whether the image is self-describing
        if os.path.splitext(infile)[1] == ".y4m":
            return cls.FromY4MFile(infile, debug)
        # check image resolution
        assert width % 2 == 0, f"error: only accept images with even width {width=}"
        assert height % 2 == 0, f"error: only accept images with even height {height=}"
        # check image pix_fmt
        pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
        # get format info
        layout = BAYER_FORMATS[pix_fmt]["layout"]
        if layout == LayoutType.packed:
            clen = BAYER_FORMATS[pix_fmt]["clen"]
            blen = BAYER_FORMATS[pix_fmt]["blen"]
            # make sure the width is OK
            # for Bayer pixel formats, only the width is important
            assert (
                width % clen == 0
            ), f"error: invalid width ({width}) as clen: {clen} for {pix_fmt}"
            expected_size = int((width * height * blen) / clen)
        elif layout == LayoutType.planar:
            depth = get_depth(pix_fmt)
            element_size_bytes = 1 if depth == 8 else 2
            expected_size = height * width * element_size_bytes

        # make sure the dimensions are OK
        file_size = os.stat(infile).st_size
        assert (
            expected_size == file_size
        ), f"error: invalid dimensions: {width}x{height}, {pix_fmt=}, {expected_size=}, {file_size=}"

        # read the file into a buffer
        with open(infile, "rb") as fin:
            buffer = fin.read(expected_size)

        return cls.FromBuffer(buffer, width, height, pix_fmt, infile, debug)

    @classmethod
    def FromBuffer(cls, buffer, width, height, pix_fmt, infile="", debug=0):
        return BayerImage(infile, buffer, None, None, width, height, pix_fmt, debug)

    @classmethod
    def FromBayerPlanar(cls, bayer_planar, pix_fmt, infile="", debug=0):
        # get format info
        height, width = (2 * dim for dim in bayer_planar["R"].shape)
        pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
        return BayerImage(
            infile, None, None, bayer_planar, width, height, pix_fmt, debug
        )

    @classmethod
    def FromBayerPacked(cls, bayer_packed, pix_fmt, debug=0):
        # get format info
        height, width = bayer_packed.shape
        pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
        bayer_image = BayerImage(
            "", None, bayer_packed, None, width, height, pix_fmt, debug
        )
        # ensure the image is correct
        order = get_order(pix_fmt)
        if cls.GetComponentType(order) == ComponentType.ydgcocg:
            # ensure there is a ydgcocg planar
            bayer_image.ydgcocg_planar = bayer_image.GetYDgCoCgPlanarFromBayerPlanar()
        return bayer_image

    @classmethod
    def FromYDgCoCgPlanar(cls, ydgcocg_planar, pix_fmt, debug=0):
        # get format info
        height, width = (2 * dim for dim in ydgcocg_planar["Y"].shape)
        pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
        bayer_image = BayerImage("", None, None, None, width, height, pix_fmt, debug)
        bayer_image.ydgcocg_planar = ydgcocg_planar
        # ensure the image is correct
        order = get_order(pix_fmt)
        if cls.GetComponentType(order) == ComponentType.bayer:
            # ensure there is a bayer planar
            bayer_image.bayer_planar = bayer_image.GetBayerPlanarFromYDgCoCgPlanar()
        return bayer_image


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    input_choices_str = " | ".join(I_PIX_FMT_LIST)
    parser.add_argument(
        "--i_pix_fmt",
        action="store",
        type=str,
        dest="i_pix_fmt",
        default=default_values["i_pix_fmt"],
        choices=I_PIX_FMT_LIST
        + [
            None,
        ],
        metavar=f"[{input_choices_str}]",
        help="input pixel format",
    )
    output_choices_str = " | ".join(O_PIX_FMT_LIST)
    parser.add_argument(
        "--o_pix_fmt",
        action="store",
        type=str,
        dest="o_pix_fmt",
        default=default_values["o_pix_fmt"],
        choices=O_PIX_FMT_LIST,
        metavar=f"[{output_choices_str}]",
        help="output pixel format",
    )
    # 2-parameter setter using argparse.Action
    parser.add_argument(
        "--width",
        action="store",
        type=int,
        dest="width",
        default=default_values["width"],
        metavar="WIDTH",
        help=("use WIDTH width (default: %i)" % default_values["width"]),
    )
    parser.add_argument(
        "--height",
        action="store",
        type=int,
        dest="height",
        default=default_values["height"],
        metavar="HEIGHT",
        help=("HEIGHT height (default: %i)" % default_values["height"]),
    )

    class VideoSizeAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.width, namespace.height = [int(v) for v in values[0].split("x")]

    parser.add_argument(
        "--video-size",
        action=VideoSizeAction,
        nargs=1,
        help="use <width>x<height>",
    )

    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        type=str,
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )

    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def convert_image_format(infile, i_pix_fmt, width, height, outfile, o_pix_fmt, debug):
    # 1. read input image file
    bayer_image = BayerImage.FromFile(infile, i_pix_fmt, width, height, debug)

    # 2. convert image to new pixel format
    bayer_image_copy = bayer_image.Copy(o_pix_fmt, debug)

    # 3. write converted image into output file
    bayer_image_copy.ToFile(outfile, debug)

    o_pix_fmt = get_canonical_output_pix_fmt(o_pix_fmt)
    ffmpeg_support = BAYER_FORMATS[o_pix_fmt]["ffmpeg"]
    if debug > 0 and ffmpeg_support:
        print(
            f"info: {itools_common.FFMPEG_SILENT} -f rawvideo -pixel_format {o_pix_fmt} "
            f"-s {width}x{height} -i {outfile} {outfile}.png"
        )
    return bayer_image


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get infile/outfile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")

    convert_image_format(
        options.infile,
        options.i_pix_fmt,
        options.width,
        options.height,
        options.outfile,
        options.o_pix_fmt,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
