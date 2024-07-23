#!/usr/bin/env python3

"""itools-generate.py module description.


Module that generates color patterns.
"""


import argparse
import numpy as np
import subprocess
import sys
import importlib

itools_common = importlib.import_module("itools-common")


default_values = {
    "debug": 0,
    "dry_run": False,
    "json_output": False,
    "pattern": "kwrgb",
    "width": 16,
    "height": 16,
    "outfile": None,
}


# generator colors

# RGGB: (0, 0, 0, 0)
BLACK0_COMPONENT_4_TOP = b"\x00\x00\x00\x00\x00"
BLACK0_COMPONENT_4_BOT = BLACK0_COMPONENT_4_TOP
# RGGB: (0x40, 0x40, 0x40, 0x40)
BLACK64_COMPONENT_4_TOP = b"\x10\x10\x10\x10\x00"
BLACK64_COMPONENT_4_BOT = BLACK64_COMPONENT_4_TOP
# RGGB: (0x100, 0x100, 0x100, 0x100)
GRAY100_COMPONENT_4_TOP = b"\x40\x40\x40\x40\x00"
GRAY100_COMPONENT_4_BOT = GRAY100_COMPONENT_4_TOP
# RGGB: (0x200, 0x200, 0x200, 0x200)
GRAY200_COMPONENT_4_TOP = b"\x80\x80\x80\x80\x00"
GRAY200_COMPONENT_4_BOT = GRAY200_COMPONENT_4_TOP
# RGGB: (0x300, 0x300, 0x300, 0x300)
GRAY300_COMPONENT_4_TOP = b"\xc0\xc0\xc0\xc0\x00"
GRAY300_COMPONENT_4_BOT = GRAY300_COMPONENT_4_TOP
# RGGB: (0x3ff, 0x3ff, 0x3ff, 0x3ff)
WHITE_COMPONENT_4_TOP = b"\xff\xff\xff\xff\xff"
WHITE_COMPONENT_4_BOT = WHITE_COMPONENT_4_TOP

# assume RGGB
RED_COMPONENT_4_TOP = b"\xff\x10\xff\x10\x33"
RED_COMPONENT_4_BOT = b"\x10\x10\x10\x10\x00"
BLUE_COMPONENT_4_TOP = b"\x10\x10\x10\x10\x00"
BLUE_COMPONENT_4_BOT = b"\x10\xff\x10\xff\xcc"
GREEN_COMPONENT_4_TOP = b"\x10\xff\x10\xff\xcc"
GREEN_COMPONENT_4_BOT = b"\xff\x10\xff\x10\x33"
TEST_COMPONENT_1_TOP = b"\x02\x04\x06\x08\x00"
TEST_COMPONENT_1_BOT = b"\x0a\x0c\x0e\x10\x00"
TEST_COMPONENT_2_TOP = b"\x12\x14\x16\x18\x00"
TEST_COMPONENT_2_BOT = b"\x1a\x1c\x1e\x20\x00"


COLOR_LIST = {
    "black0": (BLACK0_COMPONENT_4_TOP, BLACK0_COMPONENT_4_BOT),
    "black64": (BLACK64_COMPONENT_4_TOP, BLACK64_COMPONENT_4_BOT),
    "gray100": (GRAY100_COMPONENT_4_TOP, GRAY100_COMPONENT_4_BOT),
    "gray200": (GRAY200_COMPONENT_4_TOP, GRAY200_COMPONENT_4_BOT),
    "gray300": (GRAY300_COMPONENT_4_TOP, GRAY300_COMPONENT_4_BOT),
    "white": (WHITE_COMPONENT_4_TOP, WHITE_COMPONENT_4_BOT),
    "red": (RED_COMPONENT_4_TOP, RED_COMPONENT_4_BOT),
    "green": (GREEN_COMPONENT_4_TOP, GREEN_COMPONENT_4_BOT),
    "blue": (BLUE_COMPONENT_4_TOP, BLUE_COMPONENT_4_BOT),
    "test1": (TEST_COMPONENT_1_TOP, TEST_COMPONENT_1_BOT),
    "test2": (TEST_COMPONENT_2_TOP, TEST_COMPONENT_2_BOT),
}


PATTERN_COLORS = {
    "bandw": ("black64", "white"),
    "kwrgb": ("black64", "white", "red", "green", "blue"),
    "grayscale": ("black0", "black64", "gray100", "gray200", "gray300", "white"),
    "test": ("test1", "test2"),
}

PATTERN_LIST = PATTERN_COLORS.keys()


# MIPI-RAW10-RGGB generator (SGRBG10P, pgAA)
def generate_bayer_pgAA(outfile, num_cols, num_rows, pattern, debug):
    with open(outfile, "wb") as fout:
        num_bands = len(PATTERN_COLORS[pattern])
        delta_rows = num_rows // num_bands
        row = 0

        while row < num_rows:
            # select the color
            cur_band = min(num_bands - 1, row // delta_rows)
            cur_color = PATTERN_COLORS[pattern][cur_band]
            component_4_top, component_4_bot = COLOR_LIST[cur_color]
            # do the top row
            col = 0
            while col < num_cols:
                fout.write(component_4_top)
                col += 4
            row += 1
            # do the bottom row
            col = 0
            while col < num_cols:
                fout.write(component_4_bot)
                col += 4
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=default_values["dry_run"],
        help="Dry run",
    )
    parser.add_argument(
        "--pattern",
        action="store",
        type=str,
        dest="pattern",
        default=default_values["pattern"],
        choices=PATTERN_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                PATTERN_LIST,
            )
        ),
        help="pattern arg",
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
        "-o",
        "--outfile",
        action="store",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # implement help
    return options


def main(argv):
    # parse options
    options = get_options(argv)

    # get outfile
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    generate_bayer_raw10_rggb(
        options.outfile,
        options.width,
        options.height,
        options.pattern,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
