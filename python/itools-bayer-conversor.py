#!/usr/bin/env python3

"""Module to convert (raw) Bayer (CFA) images to ffmpeg bayer formats.

ffmpeg only supports 8 Bayer formats (12 when considering that the 16-bit
formats exist in both BE and LE flavors). We want to allow converting
other Bayer formats to any of the ffmpeg ones. Main goal is to allow
ffmpeg access to generic Bayer formats.
"""


import argparse
import sys


__version__ = "0.1"

COLOR_ORDER = ["RGGB", "BGGR", "GRBG", "GBRG"]


# read functions

# 2 bytes -> 2 components
def rfun_8(data):
    return data[0], data[1]


# 4 bytes -> 2 components
def rfun_10_expanded_to_16(data):
    # check the high 6 bits of both components are 0x0
    if (data[1] & 0xFC) != 0 or (data[3] & 0xFC) != 0:
        print("warning: upper 6 bits are not zero")
    return (
        (data[0] << 6) | ((data[1] & 0x03) << 14),
        (data[2] << 6) | ((data[3] & 0x03) << 14),
    )


# 5 bytes -> 4 components
def rfun_10_packed(data):
    low = data[4]
    return (
        (data[0] << 8) | ((low & 0x03) << 6),
        (data[1] << 8) | ((low & 0x0C) << 4),
        (data[2] << 8) | ((low & 0x30) << 2),
        (data[3] << 8) | ((low & 0xC0) << 0),
    )


# 2 bytes -> 2 components
def rfun_10_alaw(data):
    raise AssertionError("rfun_10_alaw: unimplemented")


# 2 bytes -> 2 components
def rfun_10_dpcm(data):
    raise AssertionError("rfun_10_dpcm: unimplemented")


# 32 bytes -> 25 components
def rfun_10_ipu3(data):
    raise AssertionError("rfun_10_ipu3: unimplemented")


# 4 bytes -> 2 components
def rfun_12_expanded_to_16(data):
    # check the high 4 bits of both components are 0x0
    if (data[1] & 0xF0) != 0 or (data[3] & 0xF0) != 0:
        print("warning: upper 4 bits are not zero")
    return (
        (data[0] << 4) | ((data[1] & 0x0F) << 12),
        (data[2] << 4) | ((data[3] & 0x0F) << 12),
    )


# 3 bytes -> 2 components
def rfun_12_packed(data):
    low = data[2]
    return (
        (data[0] << 8) | ((low & 0x0F) << 4),
        (data[1] << 8) | ((low & 0xF0) << 0),
    )


# 4 bytes -> 2 components
def rfun_14_expanded_to_16(data):
    # check the high 2 bits of both components are 0x0
    if (data[1] & 0xC0) != 0 or (data[3] & 0xC0) != 0:
        print("warning: upper 2 bits are not zero")
    return (
        (data[0] << 2) | ((data[1] & 0x3F) << 10),
        (data[2] << 2) | ((data[3] & 0x3F) << 10),
    )


# 7 bytes -> 4 components
def rfun_14_packed(data):
    low0, low1, low2 = data[4:6]
    return (
        (data[0] << 8) | ((low0 & 0x3F) << 2),
        (data[1] << 8) | ((low1 & 0x0F) << 2) | ((low0 & 0xC0) << 0),
        (data[2] << 8) | ((low2 & 0x03) << 2) | ((low1 & 0xF0) << 0),
        (data[3] << 8) | ((low2 & 0xFC) << 0),
    )


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


# write functions

# 2 bytes -> 2 components
def wfun_8(c0, c1):
    raise AssertionError("wfun_8: unimplemented")
    return c0.to_bytes(1, "big") + c1.to_bytes(2, "big")


# 4 bytes -> 2 components
def wfun_16be(c0, c1):
    return c0.to_bytes(2, "big") + c1.to_bytes(2, "big")


BAYER_FORMATS = {
    # 8-bit Bayer formats
    "RGGB": {
        "alias": ("SRGGB8",),
        # byte length
        "blen": 2,
        # component length
        "clen": 2,
        # component depth (in bits)
        "cdepth": 8,
        # read function
        "rfun": rfun_8,
        # component order
        "order": "RGGB",
    },
    "GRBG": {
        "alias": ("SGRBG8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rfun": rfun_8,
        "order": "GRBG",
    },
    "GBRG": {
        "alias": ("SGBRG8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rfun": rfun_8,
        "order": "GBRG",
    },
    "BA81": {
        "alias": ("SBGGR8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "rfun": rfun_8,
        "order": "BGGR",
    },
    # 10-bit Bayer formats expanded to 16 bits
    "RG10": {
        "alias": ("SRGGB10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "order": "RGGB",
    },
    "BA10": {
        "alias": ("SGRBG10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "order": "GRBG",
    },
    "GB10": {
        "alias": ("SGBRG10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "order": "GBRG",
    },
    "BG10": {
        "alias": ("SBGGR10",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_expanded_to_16,
        "order": "BGGR",
    },
    # 10-bit Bayer formats (packed)
    "pRAA": {
        "alias": ("SRGGB10P", "MIPI-RAW10-RGGB"),
        "blen": 5,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_10_packed,
        "order": "RGGB",
    },
    "pgAA": {
        "alias": ("SGRBG10P", "MIPI-RAW10-GRBG"),
        "blen": 5,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_10_packed,
        "order": "GRBG",
    },
    "pGAA": {
        "alias": ("SGBRG10P", "MIPI-RAW10-GBRG"),
        "blen": 5,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_10_packed,
        "order": "GBRG",
    },
    "pBAA": {
        "alias": ("SBGGR10P", "MIPI-RAW10-BGGR"),
        "blen": 5,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_10_packed,
        "order": "BGGR",
    },
    # 10-bit Bayer formats compressed to 8 bits using a-law
    "aRA8": {
        "alias": ("SRGGB10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_alaw,
        "order": "RGGB",
    },
    "aBA8": {
        "alias": ("SBGGR10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_alaw,
        "order": "BGGR",
    },
    "aGA8": {
        "alias": ("SGBRG10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_alaw,
        "order": "GBRG",
    },
    "agA8": {
        "alias": ("SGRBG10ALAW8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_alaw,
        "order": "GRBG",
    },
    # 10-bit Bayer formats compressed to 8 bits using dpcm
    "bRA8": {
        "alias": ("SRGGB10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_dpcm,
        "order": "RGGB",
    },
    "bBA8": {
        "alias": ("SBGGR10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_dpcm,
        "order": "BGGR",
    },
    "bGA8": {
        "alias": ("SGBRG10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_dpcm,
        "order": "GBRG",
    },
    "BD10": {
        "alias": ("SGRBG10DPCM8",),
        "blen": 2,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_10_dpcm,
        "order": "GRBG",
    },
    # 10-bit Bayer formats compressed a la Intel IPU3 driver
    "ip3r": {
        "alias": ("IPU3_SRGGB10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 16,
        "rfun": rfun_10_ipu3,
        "order": "RGGB",
    },
    "ip3b": {
        "alias": ("IPU3_SBGGR10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 16,
        "rfun": rfun_10_ipu3,
        "order": "BGGR",
    },
    "ip3g": {
        "alias": ("IPU3_SGBRG10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 16,
        "rfun": rfun_10_ipu3,
        "order": "GBRG",
    },
    "ip3G": {
        "alias": ("IPU3_SGRBG10",),
        "blen": 32,
        "clen": 25,
        "cdepth": 16,
        "rfun": rfun_10_ipu3,
        "order": "GRBG",
    },
    # 12-bit Bayer formats expanded to 16 bits
    "RG12": {
        "alias": ("SRGGB12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "order": "RGGB",
    },
    "BA12": {
        "alias": ("SGRBG12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "order": "GRBG",
    },
    "GB12": {
        "alias": ("SGBRG12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "order": "GBRG",
    },
    "BG12": {
        "alias": ("SBGGR12",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_expanded_to_16,
        "order": "BGGR",
    },
    # 12-bit Bayer formats (packed)
    "pRCC": {
        "alias": ("SRGGB12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_packed,
        "order": "RGGB",
    },
    "pgCC": {
        "alias": ("SGRBG12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_packed,
        "order": "GRBG",
    },
    "pGCC": {
        "alias": ("SGBRG12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_packed,
        "order": "GBRG",
    },
    "pBCC": {
        "alias": ("SBGGR12P",),
        "blen": 3,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_12_packed,
        "order": "BGGR",
    },
    # 14-bit Bayer formats expanded to 16 bits
    "RG14": {
        "alias": ("SRGGB14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "order": "RGGB",
    },
    "GR14": {
        "alias": ("SGRBG14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "order": "GRBG",
    },
    "GB14": {
        "alias": ("SGBRG14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "order": "GBRG",
    },
    "BG14": {
        "alias": ("SBGGR14",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_14_expanded_to_16,
        "order": "BGGR",
    },
    # 14-bit Bayer formats (packed)
    "pREE": {
        "alias": ("SRGGB14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_14_packed,
        "order": "RGGB",
    },
    "pgEE": {
        "alias": ("SGRBG14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_14_packed,
        "order": "GRBG",
    },
    "pGEE": {
        "alias": ("SGBRG14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_14_packed,
        "order": "GBRG",
    },
    "pBEE": {
        "alias": ("SBGGR14P",),
        "blen": 7,
        "clen": 4,
        "cdepth": 16,
        "rfun": rfun_14_packed,
        "order": "BGGR",
    },
    # 16-bit Bayer formats
    "RG16": {
        "alias": ("SRGGB16",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16le,
        "order": "RGGB",
    },
    "GR16": {
        "alias": ("SGRBG16",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16le,
        "order": "GRBG",
    },
    "GB16": {
        "alias": ("SGBRG16",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16le,
        "order": "GBRG",
    },
    "BA82": {
        "alias": ("SBGGR16",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16le,
        "order": "BGGR",
    },
    "BYR2": {
        "alias": ("SBGGR16",),
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16le,
        "order": "BGGR",
    },
    # ffmpeg bayer formats
    "bayer_bggr8": {
        "order": "BGGR",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        # write function
        "wfun": wfun_8,
    },
    "bayer_rggb8": {
        "order": "RGGB",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "wfun": wfun_8,
    },
    "bayer_gbrg8": {
        "order": "GBRG",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "wfun": wfun_8,
    },
    "bayer_grbg8": {
        "order": "GRBG",
        "blen": 2,
        "clen": 2,
        "cdepth": 8,
        "wfun": wfun_8,
    },
    "bayer_bggr16be": {
        "order": "BGGR",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_rggb16be": {
        "order": "RGGB",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_gbrg16be": {
        "order": "GBRG",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
    "bayer_grbg16be": {
        "order": "GRBG",
        "blen": 4,
        "clen": 2,
        "cdepth": 16,
        "rfun": rfun_16be,
        "wfun": wfun_16be,
    },
}

# calculate INPUT_FORMATS and OUTPUT_FORMATS
INPUT_FORMATS = {k: v for (k, v) in BAYER_FORMATS.items() if "rfun" in v}
OUTPUT_FORMATS = {k: v for (k, v) in BAYER_FORMATS.items() if "wfun" in v}
INPUT_CANONICAL_LIST = list(INPUT_FORMATS.keys())
INPUT_ALIAS_LIST = list(alias for v in INPUT_FORMATS.values() if "alias" in v for alias in v["alias"])
OUTPUT_CANONICAL_LIST = list(OUTPUT_FORMATS.keys())
OUTPUT_ALIAS_LIST = list(alias for v in OUTPUT_FORMATS.values() if "alias" in v for alias in v["alias"])


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

    # get recommended output pixel format
    icdepth = INPUT_FORMATS[i_pix_fmt]["cdepth"]
    iorder = INPUT_FORMATS[i_pix_fmt]["order"]
    # find an output pix_fmt with the same cdepth and order
    for pix_fmt, v in OUTPUT_FORMATS.items():
        if v["cdepth"] == icdepth and v["order"] == iorder:
            expected_o_pix_fmt = pix_fmt
            break
    else:
        raise AssertionError(f"error: no match for input pix_fmt: {i_pix_fmt}")

    # enforce requested output pix_fmt is expected one
    assert o_pix_fmt == expected_o_pix_fmt, (
        f"error: {expected_o_pix_fmt} is preferred to {o_pix_fmt} as "
        "output pixel format")
    return o_pix_fmt


# for Bayer pixel formats, only the width is important
def rfun_image_file(infile, i_pix_fmt, width, height, outfile, o_pix_fmt, debug):
    # check the input pixel format
    i_pix_fmt = check_input_pix_fmt(i_pix_fmt)

    # check the output pixel format
    o_pix_fmt = check_output_pix_fmt(o_pix_fmt, i_pix_fmt)

    # ifmt = INPUT_FORMATS[i_pix_fmt]  # XXX
    # ofmt = OUTPUT_FORMATS[o_pix_fmt]  # XXX

    # TODO(chema): fix conversion limitation
    # * cdepth: all conversions should be valid (including those that do 10+ -> 8 bits)
    # * order: we enforce the same component order to make the code simpler

    # open infile and outfile
    with open(infile, "rb") as fin, open(outfile, "wb") as fout:
        # process infile
        while True:
            # row_index = 0
            idata = fin.read(INPUT_FORMATS[i_pix_fmt]["blen"])
            if not idata:
                break
            components = INPUT_FORMATS[i_pix_fmt]["rfun"](idata)
            c_index = 0
            while c_index < len(components):
                clen = OUTPUT_FORMATS[o_pix_fmt]["clen"]
                odata = OUTPUT_FORMATS[o_pix_fmt]["wfun"](
                    *components[c_index: c_index + clen]
                )
                fout.write(odata)
                c_index += clen

            # TODO(chema): enforce input width (width % iclen == 0)
            # row_index = 0

        print(
            f"ffmpeg -f rawvideo -pixel_format {o_pix_fmt} "
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
        "--video_size",
        action=VideoSizeAction,
        nargs=1,
        help="use <width>x<height>",
    )

    parser.add_argument(
        "infile",
        type=str,
        nargs="?",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "outfile",
        type=str,
        nargs="?",
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
