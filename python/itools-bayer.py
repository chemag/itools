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
import importlib
import numpy as np
import os
import sys

itools_common = importlib.import_module("itools-common")


__version__ = "0.1"

COLOR_COMPONENTS = set("RGgB")

# internal planar bayer image format is G1G2RB
DEFAULT_PLANAR_ORDER = list("GgRB")


# planar read/write functions


# 2 bytes -> 2 components
def rfun_8(data, debug):
    return data[0], data[1]


# 2 bytes -> 2 components
def wfun_8(c0, c1, debug):
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
    c0 = ((data[0]) | ((data[1] & 0x03) << 8)) << 6
    c1 = ((data[2]) | ((data[3] & 0x03) << 8)) << 6
    return (c0, c1)


# 2 bytes -> 2 components
def wfun_10_expanded_to_16(c0, c1, debug):
    c0 >>= 6
    c1 >>= 6
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


# 5 bytes -> 4 components
def rfun_10_packed_expanded_to_16(data, debug):
    low = data[4]
    return (
        (data[0] << 8) | ((low & 0x03) << 6),
        (data[1] << 8) | ((low & 0x0C) << 4),
        (data[2] << 8) | ((low & 0x30) << 2),
        (data[3] << 8) | ((low & 0xC0) << 0),
    )


def wfun_10_packed_expanded_to_16(c0, c1, c2, c3, debug):
    main = ((c0 >> 8) << 24) | ((c1 >> 8) << 16) | ((c2 >> 8) << 8) | ((c3 >> 8) << 0)
    remaining = (
        (((c3 >> 6) & 0x03) << 6)
        | (((c2 >> 6) & 0x03) << 4)
        | (((c1 >> 6) & 0x03) << 2)
        | (((c0 >> 6) & 0x03) << 0)
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
    if (data[1] & 0xF0) != 0 or (data[3] & 0xF0) != 0:
        print("warn: upper 4 bits are not zero")
    return (
        (data[0] << 4) | ((data[1] & 0x0F) << 12),
        (data[2] << 4) | ((data[3] & 0x0F) << 12),
    )


def wfun_12_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_12_expanded_to_16: unimplemented")


# 3 bytes -> 2 components
def rfun_12_packed_expanded_to_16(data, debug):
    low = data[2]
    return (
        (data[0] << 8) | ((low & 0x0F) << 4),
        (data[1] << 8) | ((low & 0xF0) << 0),
    )


def wfun_12_packed_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_12_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_14_expanded_to_16(data, debug):
    # check the high 2 bits of both components are 0x0
    if (data[1] & 0xC0) != 0 or (data[3] & 0xC0) != 0:
        print("warn: upper 2 bits are not zero")
    return (
        (data[0] << 2) | ((data[1] & 0x3F) << 10),
        (data[2] << 2) | ((data[3] & 0x3F) << 10),
    )


def wfun_14_expanded_to_16(c0, c1, debug):
    raise AssertionError("wfun_14_expanded_to_16: unimplemented")


# 7 bytes -> 4 components
def rfun_14_packed_expanded_to_16(data, debug):
    low0, low1, low2 = data[4:6]
    return (
        (data[0] << 8) | ((low0 & 0x3F) << 2),
        (data[1] << 8) | ((low1 & 0x0F) << 2) | ((low0 & 0xC0) << 0),
        (data[2] << 8) | ((low2 & 0x03) << 2) | ((low1 & 0xF0) << 0),
        (data[3] << 8) | ((low2 & 0xFC) << 0),
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
    return int(c0).to_bytes(2, "big") + int(c1).to_bytes(2, "big")


def wfun_16le(c0, c1, debug):
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


BAYER_FORMATS = {
    # 8-bit Bayer formats
    "bayer_bggr8": {
        "alias": (
            "BA81",
            "SBGGR8",
        ),
        # component order
        "order": "BGgR",
        # byte length
        "blen": 2,
        # component length
        "clen": 2,
        # component depth (in bits)
        "cdepth": 8,
        # component read depth (in bits)
        "rdepth": 8,
        # read function (planar)
        "rfun": rfun_8,
        # write function (planar)
        "wfun": wfun_8,
        # ffmpeg support
        "ffmpeg": True,
    },
    "bayer_rggb8": {
        "alias": (
            "RGGB",
            "SRGGB8",
        ),
        "order": "RGgB",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rdepth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_gbrg8": {
        "alias": (
            "GBRG",
            "SGBRG8",
        ),
        "order": "GBRg",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rdepth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_grbg8": {
        "alias": (
            "GRBG",
            "SGRBG8",
        ),
        "order": "GRBg",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rdepth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
        "ffmpeg": True,
    },
    "bayer_ggbr8": {
        # component order
        "order": "GgBR",
        # byte length
        "blen": 2,
        # component length
        "clen": 2,
        # component depth (in bits)
        "cdepth": 8,
        # component read depth (in bits)
        "rdepth": 8,
        # read function
        "rfun": rfun_8,
        # write function
        "wfun": wfun_8,
        # ffmpeg support
        "ffmpeg": True,
    },
    "bayer_ggrb8": {
        # component order
        "order": "GgRB",
        # byte length
        "blen": 2,
        # component length
        "clen": 2,
        # component depth (in bits)
        "cdepth": 8,
        # component read depth (in bits)
        "rdepth": 8,
        # read function
        "rfun": rfun_8,
        # write function
        "wfun": wfun_8,
        # ffmpeg support
        "ffmpeg": True,
    },
    "bayer_rgbg8": {
        # component order
        "order": "RGBg",
        # byte length
        "blen": 2,
        # component length
        "clen": 2,
        # component depth (in bits)
        "cdepth": 8,
        # component read depth (in bits)
        "rdepth": 8,
        # read function
        "rfun": rfun_8,
        # write function
        "wfun": wfun_8,
        # ffmpeg support
        "ffmpeg": True,
    },
    "bayer_bgrg8": {
        # component order
        "order": "BGRg",
        # byte length
        "blen": 2,
        # component length
        "clen": 2,
        # component depth (in bits)
        "cdepth": 8,
        # component read depth (in bits)
        "rdepth": 8,
        # read function
        "rfun": rfun_8,
        # write function
        "wfun": wfun_8,
        # ffmpeg support
        "ffmpeg": True,
    },
    # 10-bit Bayer formats expanded to 16 bits
    "RG10": {
        "alias": ("SRGGB10",),
        "blen": 4,
        "clen": 2,
        # this is the depth of the rfun's output
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "BA10": {
        "alias": ("SGRBG10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB10": {
        "alias": ("SGBRG10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG10": {
        "alias": ("SBGGR10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "wfun": wfun_10_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats (packed)
    "pRAA": {
        "alias": ("SRGGB10P", "MIPI-RAW10-RGGB"),
        "blen": 5,
        "clen": 4,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "pgAA": {
        "alias": ("SGRBG10P", "MIPI-RAW10-GRBG"),
        "blen": 5,
        "clen": 4,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "pGAA": {
        "alias": ("SGBRG10P", "MIPI-RAW10-GBRG"),
        "blen": 5,
        "clen": 4,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "pBAA": {
        "alias": ("SBGGR10P", "MIPI-RAW10-BGGR"),
        "blen": 5,
        "clen": 4,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_packed_expanded_to_16,
        "wfun": wfun_10_packed_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats compressed to 8 bits using a-law
    "aRA8": {
        "alias": ("SRGGB10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "aBA8": {
        "alias": ("SBGGR10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    "aGA8": {
        "alias": ("SGBRG10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "agA8": {
        "alias": ("SGRBG10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_alaw_expanded_to_16,
        "wfun": wfun_10_alaw_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats compressed to 8 bits using dpcm
    "bRA8": {
        "alias": ("SRGGB10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "bBA8": {
        "alias": ("SBGGR10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    "bGA8": {
        "alias": ("SGBRG10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BD10": {
        "alias": ("SGRBG10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_dpcm_expanded_to_16,
        "wfun": wfun_10_dpcm_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    # 10-bit Bayer formats compressed a la Intel IPU3 driver
    "ip3r": {
        "alias": ("IPU3_SRGGB10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "ip3b": {
        "alias": ("IPU3_SBGGR10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    "ip3g": {
        "alias": ("IPU3_SGBRG10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "ip3G": {
        "alias": ("IPU3_SGRBG10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 10,
        "rdepth": 16,
        "rfun": rfun_10_ipu3_expanded_to_16,
        "wfun": wfun_10_ipu3_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    # 12-bit Bayer formats expanded to 16 bits
    "RG12": {
        "alias": ("SRGGB12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "BA12": {
        "alias": ("SGRBG12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB12": {
        "alias": ("SGBRG12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG12": {
        "alias": ("SBGGR12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "wfun": wfun_12_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 12-bit Bayer formats (packed)
    "pRCC": {
        "alias": ("SRGGB12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "pgCC": {
        "alias": ("SGRBG12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "pGCC": {
        "alias": ("SGBRG12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "pBCC": {
        "alias": ("SBGGR12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 12,
        "rdepth": 16,
        "rfun": rfun_12_packed_expanded_to_16,
        "wfun": wfun_12_packed_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 14-bit Bayer formats expanded to 16 bits
    "RG14": {
        "alias": ("SRGGB14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "GR14": {
        "alias": ("SGRBG14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "GB14": {
        "alias": ("SGBRG14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "BG14": {
        "alias": ("SBGGR14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "wfun": wfun_14_expanded_to_16,
        "order": "BGgR",
        "ffmpeg": False,
    },
    # 14-bit Bayer formats (packed)
    "pREE": {
        "alias": ("SRGGB14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "order": "RGgB",
        "ffmpeg": False,
    },
    "pgEE": {
        "alias": ("SGRBG14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "order": "GRBg",
        "ffmpeg": False,
    },
    "pGEE": {
        "alias": ("SGBRG14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
        "order": "GBRg",
        "ffmpeg": False,
    },
    "pBEE": {
        "alias": ("SBGGR14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 14,
        "rdepth": 16,
        "rfun": rfun_14_packed_expanded_to_16,
        "wfun": wfun_14_packed_expanded_to_16,
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
        "order": "BGgR",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_rggb16le": {
        "order": "RGgB",
        "ffmpeg": True,
        "alias": (
            "RG16",
            "SRGGB16",
        ),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_gbrg16le": {
        "alias": (
            "GB16",
            "SGBRG16",
        ),
        "order": "GBRg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_grbg16le": {
        "alias": (
            "GR16",
            "SGRBG16",
        ),
        "order": "GRBg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_bggr16be": {
        "order": "BGgR",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_rggb16be": {
        "order": "RGgB",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_gbrg16be": {
        "order": "GBRg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_grbg16be": {
        "order": "GRBg",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
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
    i_pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
    return BAYER_FORMATS[i_pix_fmt]["cdepth"]


class BayerImage:

    def __init__(self, infile, buffer, packed, planar, width, height, pix_fmt, debug=0):
        self.infile = infile
        self.buffer = buffer
        self.packed = packed
        self.planar = planar
        self.width = width
        self.height = height
        self.pix_fmt = pix_fmt
        self.debug = debug

    # accessors
    def GetBuffer(self):
        assert self.buffer is not None, "error: invalid buffer"
        return self.buffer

    def GetPacked(self):
        if self.packed is None:
            # convert buffer to packed
            assert self.buffer is not None, "error: invalid buffer"
            # convert buffer into a bayer packed image
            depth = get_depth(self.pix_fmt)
            # TODO(chema): make this a function of pix_fmt
            if depth == 8:
                self.packed = np.frombuffer(self.buffer, dtype=np.uint8)
            elif depth == 16:
                self.packed = np.frombuffer(
                    self.buffer, dtype=np.uint16
                )  # TODO(chema): this assumes little-endian
                # self.packed = np.frombuffer(self.buffer, dtype=">u2")  # big-endian
            elif depth in (10, 12, 14):
                # cv2 assumes color to be 16-bit depth if dtype is uint16
                # For 10/12/14-bit color, let's expand to 16-bit before
                # further processing. This also helps unify all further
                # processing as 16-bit.
                # support expanded and/or packed bayer formats
                # if expanded:
                # a. read as little-endian
                self.packed = np.frombuffer(self.buffer, dtype=np.uint16)
                # b. expand to 16 bits
                self.packed <<= 16 - depth
                # elif packed:
            else:
                raise ValueError(f"Unsupported depth value: {depth}")
            # reshape image
            self.packed = self.packed.reshape(self.width, self.height)
        return self.packed

    def GetPlanar(self, planar_order):
        if self.planar is None:
            # convert buffer to planar
            assert self.buffer is not None, "error: invalid buffer"
            # get format info
            pix_fmt = self.pix_fmt
            rdepth = INPUT_FORMATS[pix_fmt]["rdepth"]
            clen = INPUT_FORMATS[pix_fmt]["clen"]
            order = INPUT_FORMATS[pix_fmt]["order"]
            # create bayer planar image
            dtype = np.uint16 if rdepth > 8 else np.uint8
            self.planar = np.zeros((4, self.height // 2, self.width // 2), dtype=dtype)
            # fill it up
            row = 0
            col = 0
            i = 0
            while True:
                if self.debug > 0:
                    print(f"debug: {row=} {col=}")
                # 1. get affected plane IDs
                plane_ids = self.GetPlaneIds(order, row, planar_order)
                # 2. read components from the input
                components = ()
                while len(components) < clen:
                    length = INPUT_FORMATS[pix_fmt]["blen"]
                    idata = self.buffer[i : i + length]
                    i += length
                    if not idata:
                        break
                    components += INPUT_FORMATS[pix_fmt]["rfun"](idata, self.debug)
                if len(components) < clen:
                    # end of input
                    break
                # 3. convert component depth to 16-bit
                if rdepth > 8 and rdepth < 16:
                    components = list(c << (16 - rdepth) for c in components)
                if self.debug > 1:
                    print(f"debug:  {components=}")
                # 4. convert component order
                for component in components:
                    plane_id = plane_ids[col % len(plane_ids)]
                    # get planar row and col
                    prow = row // 2
                    pcol = col // 2
                    self.planar[plane_id][prow][pcol] = component
                    if self.debug > 1:
                        print(f"debug: {plane_id=} {prow=} {pcol=}")
                    col += 1
                # 5. update input row numbers
                if col == self.width:
                    col = 0
                    row += 1
        return self.planar

    @classmethod
    def GetPlanarOrder(cls, planar_order):
        assert set(planar_order) == COLOR_COMPONENTS, f"error: invalid Bayer components {planar_order}"
        return planar_order

    @classmethod
    def GetPlaneIds(cls, order, row, planar_order):
        order = cls.GetPlanarOrder(order)
        plane_names = order[0:2] if row % 2 == 0 else order[2:4]
        plane_ids = list(planar_order.index(plane_name) for plane_name in list(plane_names))
        return plane_ids

    # factory methods
    @classmethod
    def FromFile(cls, infile, pix_fmt, width, height, debug=0):
        # check image resolution
        assert width % 2 == 0, f"error: only accept images with even width {width=}"
        assert height % 2 == 0, f"error: only accept images with even height {height=}"
        # check image pix_fmt
        pix_fmt = get_canonical_input_pix_fmt(pix_fmt)
        # get format info
        clen = INPUT_FORMATS[pix_fmt]["clen"]
        blen = INPUT_FORMATS[pix_fmt]["blen"]
        order = INPUT_FORMATS[pix_fmt]["order"]

        # make sure the width is OK
        # for Bayer pixel formats, only the width is important
        assert (
            width % clen == 0
        ), f"error: invalid width ({width}) as clen: {clen} for {pix_fmt}"

        # make sure the dimensions are OK
        file_size = os.stat(infile).st_size
        expected_size = int((height * width * blen) / clen)
        assert (
            expected_size == file_size
        ), f"error: invalid dimensions: {height}x{width}, {pix_fmt=}, {expected_size=}, {file_size=}"

        # read the file into a buffer
        with open(infile, "rb") as fin:
            buffer = fin.read(expected_size)

        return BayerImage(infile, buffer, None, None, width, height, pix_fmt, debug)

    @classmethod
    def FromPlanar(cls, planar, pix_fmt, width, height, planar_order, debug=0):
        # get format info
        rdepth = OUTPUT_FORMATS[pix_fmt]["rdepth"]
        clen = OUTPUT_FORMATS[pix_fmt]["clen"]
        order = OUTPUT_FORMATS[pix_fmt]["order"]

        # make sure the width is OK
        # for Bayer pixel formats, only the width is important
        assert (
            width % clen == 0
        ), f"error: invalid width ({width}) as clen: {clen} for {pix_fmt}"

        row = 0
        col = 0
        buffer = b""
        while row < height:
            # 1. get affected plane IDs
            plane_ids = cls.GetPlaneIds(order, row, planar_order)
            # 2. get components in order
            components = []
            for component_id in range(clen):
                plane_id = plane_ids[col % len(plane_ids)]
                # get planar row and col
                prow = row // 2
                pcol = col // 2
                if debug > 0:
                    print(f"debug: {plane_id=} {prow=} {pcol=}")
                # TODO(chema): is planar a list or a dict?
                component = planar[plane_id][prow][pcol]
                components.append(component)
                col += 1
            # 3. convert component depth from 16-bit
            if rdepth > 8 and rdepth < 16:
                components = list(c >> (16 - rdepth) for c in components)
            if debug > 1:
                print(f"debug:  {components=}")
            # 4. write components to the output
            odata = OUTPUT_FORMATS[pix_fmt]["wfun"](*components[0:clen], debug)
            buffer += odata
            # 5. update row numbers
            if col == width:
                col = 0
                row += 1
        return BayerImage("", buffer, None, planar, width, height, pix_fmt, debug)


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
        choices=I_PIX_FMT_LIST,
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


def convert_image_planar_mode(
    infile, i_pix_fmt, width, height, outfile, o_pix_fmt, planar_order, debug
):
    # check input parameters
    planar_order = BayerImage.GetPlanarOrder(planar_order)

    # check the input pixel format
    i_pix_fmt = get_canonical_input_pix_fmt(i_pix_fmt)
    # check the output pixel format
    o_pix_fmt = get_canonical_output_pix_fmt(o_pix_fmt)

    # get common depth
    irdepth = INPUT_FORMATS[i_pix_fmt]["rdepth"]
    ordepth = OUTPUT_FORMATS[o_pix_fmt]["rdepth"]
    process_using_8bits = False
    if irdepth == 8 and ordepth == 8:
        # process the planes in 8-bit planes
        process_using_8bits = True

    # read input image file (packed) into planar
    bayer_image = BayerImage.FromFile(infile, i_pix_fmt, width, height)

    # write planar into output image file (packed)
    bayer_image_copy = BayerImage.FromPlanar(
        bayer_image.GetPlanar(planar_order), o_pix_fmt, width, height, planar_order, debug
    )
    with open(outfile, "wb") as fout:
        fout.write(bayer_image_copy.GetBuffer())

    ffmpeg_support = OUTPUT_FORMATS[o_pix_fmt]["ffmpeg"]
    if debug > 0 and ffmpeg_support:
        print(
            f"info: {itools_common.FFMPEG_SILENT} -f rawvideo -pixel_format {o_pix_fmt} "
            f"-s {width}x{height} -i {outfile} {outfile}.png"
        )
    return bayer_image.GetPlanar(planar_order)


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

    convert_image_planar_mode(
        options.infile,
        options.i_pix_fmt,
        options.width,
        options.height,
        options.outfile,
        options.o_pix_fmt,
        DEFAULT_PLANAR_ORDER,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
