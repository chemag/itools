#!/usr/bin/env python3

"""Module to convert (raw) Bayer (CFA) images to ffmpeg bayer formats.

ffmpeg only supports 8 Bayer formats (12 when considering that the 16-bit
formats exist in both BE and LE flavors). We want to allow converting
other Bayer formats to any of the ffmpeg ones. Main goal is to allow
ffmpeg access to generic Bayer formats.
"""


import argparse
import importlib
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
    return c0.to_bytes(1, "big") + c1.to_bytes(1, "big")


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
    return c0.to_bytes(2, "little") + c1.to_bytes(2, "little")


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
    return main.to_bytes(4, "big") + remaining.to_bytes(1, "big")


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
    return c0.to_bytes(2, "big") + c1.to_bytes(2, "big")


def wfun_16le(c0, c1):
    return c0.to_bytes(2, "little") + c1.to_bytes(2, "little")


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
    },
    # 16-bit Bayer formats
    "bayer_bggr16le": {
        "alias": (
            "BA82",
            "BYR2",
            "SBGGR16",
        ),
        "order": "BGGR",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_rggb16le": {
        "order": "RGGB",
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
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16le,
        "wfun": wfun_16le,
    },
    "bayer_bggr16be": {
        "order": "BGGR",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_rggb16be": {
        "order": "RGGB",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_gbrg16be": {
        "order": "GBRG",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_grbg16be": {
        "order": "GRBG",
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


def check_output_pix_fmt(o_pix_fmt, i_pix_fmt):
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


def gcd(a, b):
    if a == 0:
        return b
    # recursively calcule the gcd
    return gcd(b % a, a)


def lcm(a, b):
    return (a // gcd(a, b)) * b


def sort_components(components, iorder, oorder):
    c0, c1, c2, c3 = components
    if iorder == oorder:
        return c0, c1, c2, c3

    elif iorder == "GRBG" and oorder == "GBRG":
        return c0, c2, c1, c3
    elif iorder == "GRBG" and oorder == "BGGR":
        return c2, c0, c3, c1
    elif iorder == "GRBG" and oorder == "RGGB":
        return c1, c0, c3, c2

    elif iorder == "GBRG" and oorder == "GRBG":
        return c0, c2, c1, c3
    elif iorder == "GBRG" and oorder == "BGGR":
        return c1, c0, c3, c2
    elif iorder == "GBRG" and oorder == "RGGB":
        return c2, c0, c3, c1

    elif iorder == "BGGR" and oorder == "RGGB":
        return c3, c1, c2, c0
    elif iorder == "BGGR" and oorder == "GRBG":
        return c1, c3, c0, c1
    elif iorder == "BGGR" and oorder == "GBRG":
        return c1, c0, c3, c2

    elif iorder == "RGGB" and oorder == "BGGR":
        return c3, c1, c2, c0
    elif iorder == "RGGB" and oorder == "GRBG":
        return c1, c0, c3, c2
    elif iorder == "RGGB" and oorder == "GBRG":
        return c1, c3, c0, c2


# for Bayer pixel formats, only the width is important
def rfun_image_file(infile, i_pix_fmt, width, height, outfile, o_pix_fmt, debug):
    # check the input pixel format
    i_pix_fmt = check_input_pix_fmt(i_pix_fmt)

    # check the output pixel format
    o_pix_fmt = check_output_pix_fmt(o_pix_fmt, i_pix_fmt)

    # get depths
    irdepth = INPUT_FORMATS[i_pix_fmt]["rdepth"]
    ordepth = OUTPUT_FORMATS[o_pix_fmt]["rdepth"]

    # get component length
    iclen = INPUT_FORMATS[i_pix_fmt]["clen"]
    oclen = OUTPUT_FORMATS[o_pix_fmt]["clen"]
    # use the least common multiple
    clen = lcm(iclen, oclen)

    # get component order
    iorder = INPUT_FORMATS[i_pix_fmt]["order"]
    oorder = OUTPUT_FORMATS[o_pix_fmt]["order"]
    if iorder != oorder:
        # make sure we see at least 4 components
        clen = lcm(clen, 4)

    # open infile and outfile
    with open(infile, "rb") as fin, open(outfile, "wb") as fout:
        # process infile
        while True:
            # 1. read components from the input
            components = ()
            while len(components) < clen:
                idata = fin.read(INPUT_FORMATS[i_pix_fmt]["blen"])
                if not idata:
                    break
                components += INPUT_FORMATS[i_pix_fmt]["rfun"](idata)
            if len(components) < clen:
                # end of input
                break
            # 2. convert component depth
            if irdepth > ordepth:
                components = list(c >> (irdepth - ordepth) for c in components)
            elif irdepth < ordepth:
                components = list(c << (ordepth - irdepth) for c in components)
            # 3. convert component order
            if iorder != oorder:
                c_index = 0
                new_components = ()
                while c_index < len(components):
                    new_components += sort_components(
                        components[c_index : c_index + 4], iorder, oorder
                    )
                    c_index += 4
                components = new_components
            # 4. write components to the output
            c_index = 0
            while c_index < len(components):
                odata = OUTPUT_FORMATS[o_pix_fmt]["wfun"](
                    *components[c_index : c_index + oclen]
                )
                fout.write(odata)
                c_index += oclen

            # TODO(chema): enforce input width (width % iclen == 0)

        print(
            f"{itools_common.FFMPEG_SILENT} -f rawvideo -pixel_format {o_pix_fmt} "
            f"-s {width}x{height} -i {outfile} {outfile}.png"
        )


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
    if options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # convert input image file
    rfun_image_file(
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
