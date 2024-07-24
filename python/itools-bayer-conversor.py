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

COLOR_ORDER = ["RGGB", "BGGR", "GRBG", "GBRG"]


# read/write functions


# 2 bytes -> 2 components
def rfun_8(data):
    return data[0], data[1]


# 2 bytes -> 2 components
def wfun_8(c0, c1):
    return int(c0).to_bytes(1, "big") + int(c1).to_bytes(1, "big")


# 4 bytes -> 2 components
def rfun_10_expanded_to_16(data):
    # check the high 6 bits of both components are 0x0
    if (data[1] & 0xFC) != 0 or (data[3] & 0xFC) != 0:
        print("warning: upper 6 bits are not zero")
    return (
        (data[0] << 6) | ((data[1] & 0x03) << 14),
        (data[2] << 6) | ((data[3] & 0x03) << 14),
    )


# 2 bytes -> 2 components
def wfun_10_expanded_to_16(c0, c1):
    c0 >>= 6
    c1 >>= 6
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


# 5 bytes -> 4 components
def rfun_10_packed_expanded_to_16(data):
    low = data[4]
    return (
        (data[0] << 8) | ((low & 0x03) << 6),
        (data[1] << 8) | ((low & 0x0C) << 4),
        (data[2] << 8) | ((low & 0x30) << 2),
        (data[3] << 8) | ((low & 0xC0) << 0),
    )


def wfun_10_packed_expanded_to_16(c0, c1, c2, c3):
    main = ((c0 >> 8) << 24) | ((c1 >> 8) << 16) | ((c2 >> 8) << 8) | ((c3 >> 8) << 0)
    remaining = (
        (((c3 >> 6) & 0x03) << 6)
        | (((c2 >> 6) & 0x03) << 4)
        | (((c1 >> 6) & 0x03) << 2)
        | (((c0 >> 6) & 0x03) << 0)
    )
    return int(main).to_bytes(4, "big") + int(remaining).to_bytes(1, "big")


# 2 bytes -> 2 components
def rfun_10_alaw_expanded_to_16(data):
    raise AssertionError("rfun_10_alaw_expanded_to_16: unimplemented")


def wfun_10_alaw_expanded_to_16(c0, c1):
    raise AssertionError("wfun_10_alaw_expanded_to_16: unimplemented")


# 2 bytes -> 2 components
def rfun_10_dpcm_expanded_to_16(data):
    raise AssertionError("rfun_10_dpcm_expanded_to_16: unimplemented")


def wfun_10_dpcm_expanded_to_16(c0, c1):
    raise AssertionError("wfun_10_dpcm_expanded_to_16: unimplemented")


# 32 bytes -> 25 components
def rfun_10_ipu3_expanded_to_16(data):
    raise AssertionError("rfun_10_ipu3_expanded_to_16: unimplemented")


def wfun_10_ipu3_expanded_to_16(carray):
    raise AssertionError("wfun_10_ipu3_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_12_expanded_to_16(data):
    # check the high 4 bits of both components are 0x0
    if (data[1] & 0xF0) != 0 or (data[3] & 0xF0) != 0:
        print("warning: upper 4 bits are not zero")
    return (
        (data[0] << 4) | ((data[1] & 0x0F) << 12),
        (data[2] << 4) | ((data[3] & 0x0F) << 12),
    )


def wfun_12_expanded_to_16(c0, c1):
    raise AssertionError("wfun_12_expanded_to_16: unimplemented")


# 3 bytes -> 2 components
def rfun_12_packed_expanded_to_16(data):
    low = data[2]
    return (
        (data[0] << 8) | ((low & 0x0F) << 4),
        (data[1] << 8) | ((low & 0xF0) << 0),
    )


def wfun_12_packed_expanded_to_16(c0, c1):
    raise AssertionError("wfun_12_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_14_expanded_to_16(data):
    # check the high 2 bits of both components are 0x0
    if (data[1] & 0xC0) != 0 or (data[3] & 0xC0) != 0:
        print("warning: upper 2 bits are not zero")
    return (
        (data[0] << 2) | ((data[1] & 0x3F) << 10),
        (data[2] << 2) | ((data[3] & 0x3F) << 10),
    )


def wfun_14_expanded_to_16(c0, c1):
    raise AssertionError("wfun_14_expanded_to_16: unimplemented")


# 7 bytes -> 4 components
def rfun_14_packed_expanded_to_16(data):
    low0, low1, low2 = data[4:6]
    return (
        (data[0] << 8) | ((low0 & 0x3F) << 2),
        (data[1] << 8) | ((low1 & 0x0F) << 2) | ((low0 & 0xC0) << 0),
        (data[2] << 8) | ((low2 & 0x03) << 2) | ((low1 & 0xF0) << 0),
        (data[3] << 8) | ((low2 & 0xFC) << 0),
    )


def wfun_14_packed_expanded_to_16(c0, c1, c2, c3):
    raise AssertionError("wfun_14_packed_expanded_to_16: unimplemented")


# 4 bytes -> 2 components
def rfun_16le(data):
    return (
        (data[0] << 0) | (data[1] << 8),
        (data[2] << 0) | (data[3] << 8),
    )


def rfun_16be(data):
    return (
        (data[1] << 0) | (data[0] << 8),
        (data[3] << 0) | (data[2] << 8),
    )


# 4 bytes -> 2 components
def wfun_16be(c0, c1):
    return int(c0).to_bytes(2, "big") + int(c1).to_bytes(2, "big")


def wfun_16le(c0, c1):
    return int(c0).to_bytes(2, "little") + int(c1).to_bytes(2, "little")


BAYER_FORMATS = {
    # 8-bit Bayer formats
    "bayer_bggr8": {
        "alias": (
            "BA81",
            "SBGGR8",
        ),
        # component order
        "order": "BGGR",
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
        "order": "RGGB",
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
        "order": "GBRG",
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
        "order": "GRBG",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rdepth": 8,
        "rfun": rfun_8,
        "wfun": wfun_8,
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
        "order": "RGGB",
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
        "order": "GRBG",
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
        "order": "GBRG",
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
        "order": "BGGR",
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
        "order": "RGGB",
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
        "order": "GRBG",
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
        "order": "GBRG",
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
        "order": "BGGR",
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
        "order": "RGGB",
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
        "order": "BGGR",
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
        "order": "GBRG",
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
        "order": "GRBG",
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
        "order": "RGGB",
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
        "order": "BGGR",
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
        "order": "GBRG",
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
        "order": "GRBG",
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
        "order": "RGGB",
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
        "order": "BGGR",
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
        "order": "GBRG",
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
        "order": "GRBG",
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
        "order": "RGGB",
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
        "order": "GRBG",
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
        "order": "GBRG",
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
        "order": "BGGR",
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
        "order": "RGGB",
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
        "order": "GRBG",
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
        "order": "GBRG",
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
        "order": "BGGR",
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
        "order": "RGGB",
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
        "order": "GRBG",
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
        "order": "GBRG",
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
        "order": "BGGR",
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
        "order": "RGGB",
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
        "order": "GRBG",
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
        "order": "GBRG",
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
        "order": "BGGR",
        "ffmpeg": False,
    },
    # 16-bit Bayer formats
    "bayer_bggr16le": {
        "alias": (
            "BA82",
            "BYR2",
            "SBGGR16",
        ),
        "order": "BGGR",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_rggb16le": {
        "order": "RGGB",
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
        "order": "GBRG",
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
        "order": "GRBG",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_bggr16be": {
        "order": "BGGR",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_rggb16be": {
        "order": "RGGB",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_gbrg16be": {
        "order": "GBRG",
        "ffmpeg": True,
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_grbg16be": {
        "order": "GRBG",
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
}


def check_input_pix_fmt(i_pix_fmt):
    # convert input pixel format to the canonical name
    if i_pix_fmt in INPUT_CANONICAL_LIST:
        return i_pix_fmt
    elif i_pix_fmt in INPUT_ALIAS_LIST:
        # find the canonical name
        for canonical, v in INPUT_FORMATS.items():
            if i_pix_fmt in v["alias"]:
                return canonical
    else:
        raise AssertionError(f"error: unknown input pix_fmt: {i_pix_fmt}")


def check_output_pix_fmt(o_pix_fmt):
    # convert output pixel format to the canonical name
    if o_pix_fmt in OUTPUT_CANONICAL_LIST:
        o_pix_fmt = o_pix_fmt
    elif o_pix_fmt in OUTPUT_ALIAS_LIST:
        # find the canonical name
        for canonical, v in OUTPUT_FORMATS.items():
            if o_pix_fmt in v["alias"]:
                o_pix_fmt = canonical
                break
    else:
        raise AssertionError(f"error: unknown output pix_fmt: {o_pix_fmt}")

    return o_pix_fmt


def get_planes(order, row):
    if order == "RGGB" and row % 2 == 0:  # RG
        plane_ids = (0, 1)
    elif order == "RGGB" and row % 2 == 1:  # GB
        plane_ids = (1, 2)
    elif order == "BGGR" and row % 2 == 0:  # BG
        plane_ids = (2, 1)
    elif order == "BGGR" and row % 2 == 1:  # GR
        plane_ids = (1, 0)
    elif order == "GRBG" and row % 2 == 0:  # GR
        plane_ids = (1, 0)
    elif order == "GRBG" and row % 2 == 1:  # BG
        plane_ids = (2, 1)
    elif order == "GBRG" and row % 2 == 0:  # GB
        plane_ids = (1, 2)
    elif order == "GBRG" and row % 2 == 1:  # RG
        plane_ids = (0, 1)
    return plane_ids


def rfun_image_file(infile, i_pix_fmt, width, height, cdepth, debug):
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

    # create rgb16be image
    rgb16be_image = np.zeros((3, height, width), dtype=np.uint16)

    # open infile
    row = 0
    col = 0
    with open(infile, "rb") as fin:
        while True:
            if debug > 0:
                print(f"{row=} {col=}")
            # 1. get affected plane IDs
            plane_ids = get_planes(iorder, row)
            # 2. read components from the input
            components = ()
            while len(components) < iclen:
                idata = fin.read(INPUT_FORMATS[i_pix_fmt]["blen"])
                if not idata:
                    break
                components += INPUT_FORMATS[i_pix_fmt]["rfun"](idata)
            if len(components) < iclen:
                # end of input
                break
            # 3. convert component depth to 16-bit
            if irdepth < 16:
                components = list(c << (16 - irdepth) for c in components)
            if debug > 1:
                print(f"  {components=}")
            # 4. convert component order
            for component_id, component in enumerate(components):
                plane_id = plane_ids[col % len(plane_ids)]
                row1 = row
                row2 = (row1 + 1) if row1 % 2 == 0 else (row1 - 1)
                col1 = col
                col2 = (col1 + 1) if col1 % 2 == 0 else (col1 - 1)
                if debug > 1:
                    print(f"{row1=} {row2=} {col1=} {col2=}")
                if row % 2 == 0 or plane_id != 1:
                    rgb16be_image[plane_id][row1][col1] = component
                    rgb16be_image[plane_id][row1][col2] = component
                    rgb16be_image[plane_id][row2][col1] = component
                    rgb16be_image[plane_id][row2][col2] = component
                else:  # if row % 2 == 1 and plane_id == 1:
                    # TODO(chema) fix the second row in the G component (?)
                    # When converting an RGGB matrix to non-Bayer components,
                    # we repeat the R and B components 4x times. Now, for the
                    # G component, we need to combine 2x different G
                    # components. We opt for respecting rows, as it allows
                    # going back and forth between different formats
                    # losslessly.
                    # e.g. a (0, 2, 4, 6) RGGB 2x2 matrix will be converted to
                    # * R = [[0, 0], [0, 0]},
                    # * R = [[2, 2], [4, 4]},
                    # * B = [[6, 6], [6, 6]}.
                    # do not touch value in the previous row
                    # rgb16be_image[plane_id][row2][col1] = (
                    #    rgb16be_image[plane_id][row1][col1] + component
                    # ) // 2
                    # rgb16be_image[plane_id][row2][col2] = (
                    #    rgb16be_image[plane_id][row1][col2] + component
                    # ) // 2
                    # overwrite main value
                    rgb16be_image[plane_id][row1][col1] = component
                    rgb16be_image[plane_id][row1][col2] = component
                col += 1
            # 5. update row numbers
            if col == width:
                col = 0
                row += 1

    return rgb16be_image


def wfun_image_file(rgb16be_image, outfile, o_pix_fmt, width, height, cdepth, debug):
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
            plane_ids = get_planes(oorder, row)
            # 2. get components in order
            components = []
            for component_id in range(oclen):
                plane_id = plane_ids[col % len(plane_ids)]
                if debug > 0:
                    print(f"{plane_id=} {row=} {col=}")
                component = rgb16be_image[plane_id][row][col]
                components.append(component)
                col += 1
            # 3. convert component depth from 16-bit
            if ordepth < 16:
                components = list(c >> (16 - ordepth) for c in components)
            if debug > 1:
                print(f"  {components=}")
            # 4. write components to the output
            odata = OUTPUT_FORMATS[o_pix_fmt]["wfun"](*components[0:oclen])
            fout.write(odata)
            # 5. update row numbers
            if col == width:
                col = 0
                row += 1

    return rgb16be_image


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
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


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
        print(options)

    # get common depth
    # check the input pixel format
    i_pix_fmt = check_input_pix_fmt(options.i_pix_fmt)
    # check the output pixel format
    o_pix_fmt = check_output_pix_fmt(options.o_pix_fmt)
    icdepth = INPUT_FORMATS[i_pix_fmt]["cdepth"]
    ocdepth = OUTPUT_FORMATS[o_pix_fmt]["cdepth"]
    cdepth = max(icdepth, ocdepth)
    # read input image file into rgb16be
    rgb16be_image = rfun_image_file(
        options.infile,
        i_pix_fmt,
        options.width,
        options.height,
        cdepth,
        options.debug,
    )
    # write rgb16be into output image file
    wfun_image_file(
        rgb16be_image,
        options.outfile,
        o_pix_fmt,
        options.width,
        options.height,
        cdepth,
        options.debug,
    )
    ffmpeg_support = OUTPUT_FORMATS[o_pix_fmt]["ffmpeg"]
    if ffmpeg_support:
        print(
            f"{itools_common.FFMPEG_SILENT} -f rawvideo -pixel_format {options.o_pix_fmt} "
            f"-s {options.width}x{options.height} -i {options.outfile} {options.outfile}.png"
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
