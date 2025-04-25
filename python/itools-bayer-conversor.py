#!/usr/bin/env python3

"""Module to convert (raw) Bayer (CFA) images to ffmpeg bayer formats.

ffmpeg only supports 8 Bayer formats (12 when considering that the 16-bit
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
DEFAULT_PLANE_ORDER = list("GgRB")


# read/write functions


# 2 bytes -> 2 components
def rfun_8(data, logfd, debug):
    return data[0], data[1]


# 2 bytes -> 2 components
def wfun_8(c0, c1, logfd, debug):
    return int(c0).to_bytes(1, "big") + int(c1).to_bytes(1, "big")


# 4 bytes -> 2 components
def rfun_10_expanded_to_16(data, logfd, debug):
    # check the high 6 bits of both components are 0x0
    # 10-expanded-to-16 packs using
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    # |b7 |b6 |b5 |b4 |b3 |b2 |b1 |b0 | | 0 | 0 | 0 | 0 | 0 | 0 |b9 |b8 |
    # +---+---+---+---+---+---+---+---+ +---+---+---+---+---+---+---+---+
    if (data[1] & 0xFC) != 0 or (data[3] & 0xFC) != 0:
        print("warn: upper 6 bits are not zero", file=logfd)
    return (
        ((data[0]) | ((data[1] & 0x03) << 8)) << 6,
        ((data[2]) | ((data[3] & 0x03) << 8)) << 6,
    )


# 2 bytes -> 2 components
def wfun_10_expanded_to_16(c0, c1, logfd, debug):
    c0 >>= 6
    c1 >>= 6
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


# 5 bytes -> 4 components
def rfun_10_packed_expanded_to_16(data, logfd, debug):
    low = data[4]
    return (
        (data[0] << 8) | ((low & 0x03) << 6),
        (data[1] << 8) | ((low & 0x0C) << 4),
        (data[2] << 8) | ((low & 0x30) << 2),
        (data[3] << 8) | ((low & 0xC0) << 0),
    )


def wfun_10_packed_expanded_to_16(c0, c1, c2, c3, logfd, debug):
    main = ((c0 >> 8) << 24) | ((c1 >> 8) << 16) | ((c2 >> 8) << 8) | ((c3 >> 8) << 0)
    remaining = (
        (((c3 >> 6) & 0x03) << 6)
        | (((c2 >> 6) & 0x03) << 4)
        | (((c1 >> 6) & 0x03) << 2)
        | (((c0 >> 6) & 0x03) << 0)
    )
    return int(main).to_bytes(4, "big") + int(remaining).to_bytes(1, "big")


# 2 bytes -> 2 components
def rfun_10_alaw_expanded_to_16(data, logfd, debug):
    raise AssertionError("rfun_10_alaw_expanded_to_16: unimplemented")


def wfun_10_alaw_expanded_to_16(c0, c1, logfd, debug):
    raise AssertionError("wfun_10_alaw_expanded_to_16: unimplemented")


# 2 bytes -> 2 components
def rfun_10_dpcm_expanded_to_16(data, logfd, debug):
    raise AssertionError("rfun_10_dpcm_expanded_to_16: unimplemented")


def wfun_10_dpcm_expanded_to_16(c0, c1, logfd, debug):
    raise AssertionError("wfun_10_dpcm_expanded_to_16: unimplemented")


# 32 bytes -> 25 components
def rfun_10_ipu3_expanded_to_16(data, logfd, debug):
    raise AssertionError("rfun_10_ipu3_expanded_to_16: unimplemented")


def wfun_10_ipu3_expanded_to_16(carray, logfd, debug):
    raise AssertionError("wfun_10_ipu3_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_12_expanded_to_16(data, logfd, debug):
    # check the high 4 bits of both components are 0x0
    if (data[1] & 0xF0) != 0 or (data[3] & 0xF0) != 0:
        print("warn: upper 4 bits are not zero", file=logfd)
    return (
        (data[0] << 4) | ((data[1] & 0x0F) << 12),
        (data[2] << 4) | ((data[3] & 0x0F) << 12),
    )


def wfun_12_expanded_to_16(c0, c1, logfd, debug):
    raise AssertionError("wfun_12_expanded_to_16: unimplemented")


# 3 bytes -> 2 components
def rfun_12_packed_expanded_to_16(data, logfd, debug):
    low = data[2]
    return (
        (data[0] << 8) | ((low & 0x0F) << 4),
        (data[1] << 8) | ((low & 0xF0) << 0),
    )


def wfun_12_packed_expanded_to_16(c0, c1, logfd, debug):
    raise AssertionError("wfun_12_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_14_expanded_to_16(data, logfd, debug):
    # check the high 2 bits of both components are 0x0
    if (data[1] & 0xC0) != 0 or (data[3] & 0xC0) != 0:
        print("warn: upper 2 bits are not zero", file=logfd)
    return (
        (data[0] << 2) | ((data[1] & 0x3F) << 10),
        (data[2] << 2) | ((data[3] & 0x3F) << 10),
    )


def wfun_14_expanded_to_16(c0, c1, logfd, debug):
    raise AssertionError("wfun_14_expanded_to_16: unimplemented")


# 7 bytes -> 4 components
def rfun_14_packed_expanded_to_16(data, logfd, debug):
    low0, low1, low2 = data[4:6]
    return (
        (data[0] << 8) | ((low0 & 0x3F) << 2),
        (data[1] << 8) | ((low1 & 0x0F) << 2) | ((low0 & 0xC0) << 0),
        (data[2] << 8) | ((low2 & 0x03) << 2) | ((low1 & 0xF0) << 0),
        (data[3] << 8) | ((low2 & 0xFC) << 0),
    )


def wfun_14_packed_expanded_to_16(c0, c1, c2, c3, logfd, debug):
    raise AssertionError("wfun_14_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_16le(data, logfd, debug):
    return (
        (data[0] << 0) | (data[1] << 8),
        (data[2] << 0) | (data[3] << 8),
    )


def rfun_16be(data, logfd, debug):
    return (
        (data[1] << 0) | (data[0] << 8),
        (data[3] << 0) | (data[2] << 8),
    )


# 4 bytes -> 2 components
def wfun_16be(c0, c1, logfd, debug):
    return int(c0).to_bytes(2, "big") + int(c1).to_bytes(2, "big")


def wfun_16le(c0, c1, logfd, debug):
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
        # read function
        "rfun": rfun_8,
        # write function
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


default_values = {
    "debug": 0,
    "dry_run": False,
    "i_pix_fmt": None,
    "o_pix_fmt": None,
    "width": 0,
    "height": 0,
    "infile": None,
    "outfile": None,
    "logfile": None,
}


def check_input_pix_fmt(i_pix_fmt):
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


def check_output_pix_fmt(o_pix_fmt):
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


def get_planes(order, row, plane_order):
    assert set(order) == COLOR_COMPONENTS, f"error: invalid Bayer components {order}"
    plane_names = order[0:2] if row % 2 == 0 else order[2:4]
    plane_ids = list(plane_order.index(plane_name) for plane_name in list(plane_names))
    return plane_ids


# read bayer packed format into bayer planar image
def rfun_image_file(
    infile,
    i_pix_fmt,
    width,
    height,
    process_using_8bits,
    logfd,
    debug,
    plane_order=DEFAULT_PLANE_ORDER,
):
    # check input image resolution
    assert width % 2 == 0, f"error: only accept images with even width {width=}"
    assert height % 2 == 0, f"error: only accept images with even height {height=}"

    # get format info
    irdepth = INPUT_FORMATS[i_pix_fmt]["rdepth"]
    iclen = INPUT_FORMATS[i_pix_fmt]["clen"]
    iblen = INPUT_FORMATS[i_pix_fmt]["blen"]
    iorder = INPUT_FORMATS[i_pix_fmt]["order"]

    # make sure the width is OK
    # for Bayer pixel formats, only the width is important
    assert (
        width % iclen == 0
    ), f"error: invalid width ({width}) as clen: {iclen} for {i_pix_fmt}"

    # make sure the dimensions are OK
    file_size = os.stat(infile).st_size
    expected_size = int((height * width * iblen) / iclen)
    assert (
        expected_size == file_size
    ), f"error: invalid dimensions: {height}x{width}, {i_pix_fmt=}, {expected_size=}, {file_size=}"

    # create bayer planar image
    dtype = np.uint8 if process_using_8bits else np.uint16
    planar_image = np.zeros((4, height // 2, width // 2), dtype=dtype)

    # open infile
    row = 0
    col = 0
    with open(infile, "rb") as fin:
        while True:
            if debug > 0:
                print(f"debug: {row=} {col=}", file=logfd)
            # 1. get affected plane IDs
            plane_ids = get_planes(iorder, row, plane_order)
            # 2. read components from the input
            components = ()
            while len(components) < iclen:
                idata = fin.read(INPUT_FORMATS[i_pix_fmt]["blen"])
                if not idata:
                    break
                components += INPUT_FORMATS[i_pix_fmt]["rfun"](idata, logfd, debug)
            if len(components) < iclen:
                # end of input
                break
            # 3. convert component depth to 16-bit
            if not process_using_8bits and irdepth < 16:
                components = list(c << (16 - irdepth) for c in components)
            if debug > 1:
                print(f"debug:  {components=}", file=logfd)
            # 4. convert component order
            for component in components:
                plane_id = plane_ids[col % len(plane_ids)]
                # get planar row and col
                prow = row // 2
                pcol = col // 2
                planar_image[plane_id][prow][pcol] = component
                if debug > 1:
                    print(f"debug: {plane_id=} {prow=} {pcol=}", file=logfd)
                col += 1
            # 5. update input row numbers
            if col == width:
                col = 0
                row += 1
    return planar_image


# write bayer planar format into bayer packed image
def wfun_image_file(
    planar_image,
    outfile,
    o_pix_fmt,
    width,
    height,
    process_using_8bits,
    logfd,
    debug,
    plane_order=DEFAULT_PLANE_ORDER,
):
    # get format info
    ordepth = OUTPUT_FORMATS[o_pix_fmt]["rdepth"]
    oclen = OUTPUT_FORMATS[o_pix_fmt]["clen"]
    oblen = OUTPUT_FORMATS[o_pix_fmt]["blen"]
    oorder = OUTPUT_FORMATS[o_pix_fmt]["order"]

    # make sure the width is OK
    # for Bayer pixel formats, only the width is important
    assert (
        width % oclen == 0
    ), f"error: invalid width ({width}) as clen: {oclen} for {o_pix_fmt}"

    row = 0
    col = 0
    with open(outfile, "wb") as fout:
        while row < height:
            # 1. get affected plane IDs
            plane_ids = get_planes(oorder, row, plane_order)
            # 2. get components in order
            components = []
            for component_id in range(oclen):
                plane_id = plane_ids[col % len(plane_ids)]
                # get planar row and col
                prow = row // 2
                pcol = col // 2
                if debug > 0:
                    print(f"debug: {plane_id=} {prow=} {pcol=}", file=logfd)
                component = planar_image[plane_id][prow][pcol]
                components.append(component)
                col += 1
            # 3. convert component depth from 16-bit
            if not process_using_8bits and ordepth < 16:
                components = list(c >> (16 - ordepth) for c in components)
            if debug > 1:
                print(f"debug:  {components=}", file=logfd)
            # 4. write components to the output
            odata = OUTPUT_FORMATS[o_pix_fmt]["wfun"](
                *components[0:oclen], logfd, debug
            )
            fout.write(odata)
            # 5. update row numbers
            if col == width:
                col = 0
                row += 1


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
    I_PIX_FMT_LIST = INPUT_CANONICAL_LIST + INPUT_ALIAS_LIST
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
    O_PIX_FMT_LIST = OUTPUT_CANONICAL_LIST + OUTPUT_ALIAS_LIST
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
    parser.add_argument(
        "--logfile",
        action="store",
        dest="logfile",
        type=str,
        default=default_values["logfile"],
        metavar="log-file",
        help="log file",
    )

    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def process_image(infile, i_pix_fmt, width, height, outfile, o_pix_fmt, logfd, debug):
    # get common depth
    # check the input pixel format
    i_pix_fmt = check_input_pix_fmt(i_pix_fmt)
    # check the output pixel format
    o_pix_fmt = check_output_pix_fmt(o_pix_fmt)

    irdepth = INPUT_FORMATS[i_pix_fmt]["rdepth"]
    ordepth = OUTPUT_FORMATS[o_pix_fmt]["rdepth"]
    process_using_8bits = False
    if irdepth == 8 and ordepth == 8:
        # process the planes in 8-bit planes
        process_using_8bits = True

    # read input image file (packed) into planar
    planar_image = rfun_image_file(
        infile,
        i_pix_fmt,
        width,
        height,
        process_using_8bits,
        logfd,
        debug,
    )

    # write planar into output image file (packed)
    wfun_image_file(
        planar_image,
        outfile,
        o_pix_fmt,
        width,
        height,
        process_using_8bits,
        logfd,
        debug,
    )
    ffmpeg_support = OUTPUT_FORMATS[o_pix_fmt]["ffmpeg"]
    if ffmpeg_support:
        print(
            f"info: {itools_common.FFMPEG_SILENT} -f rawvideo -pixel_format {o_pix_fmt} "
            f"-s {width}x{height} -i {outfile} {outfile}.png"
        )
    return planar_image


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get logfile descriptor
    if options.logfile is None:
        logfd = sys.stdout
    else:
        logfd = open(options.logfile, "w")
    # get infile/outfile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")

    process_image(
        options.infile,
        options.i_pix_fmt,
        options.width,
        options.height,
        options.outfile,
        options.o_pix_fmt,
        logfd,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
