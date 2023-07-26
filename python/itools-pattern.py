#!/usr/bin/env python3

"""itools-pattern.py module description.


Module that generates color patterns in MIPI-RAW10-RGGB, and
parses them in yuv420p10le format.
"""


import argparse
import numpy as np
import sys


PATTERN_LIST = ("kwrgb", "bandw", "grayscale")
RANGE_CONVERSION_LIST = ("fr2fr", "fr2lr", "lr2fr", "lr2lr")

FUNC_CHOICES = {
    "help": "show help options",
    "generate": "generate pattern input (MIPI-RAW10-RGGB)",
    "parse": "parse pattern output (yuv420p10le)",
    "range-convert": "convert the range of the input image (yuv420p10le)",
}


default_values = {
    "debug": 0,
    "dry_run": False,
    "json_output": False,
    "pattern": "kwrgb",
    "width": 4032,
    "height": 3024,
    "range_conversion": "fr2fr",
    "func": "help",
    "infile": None,
    "infile2": None,
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
}


PATTERN_COLORS = {
    "bandw": ("black64", "white"),
    "kwrgb": ("black64", "white", "red", "green", "blue"),
    "grayscale": ("black0", "black64", "gray100", "gray200", "gray300", "white"),
}


def read_yuv420p10le_to_ndarray(infile, num_cols, num_rows):
    # preallocate the array
    y = np.zeros((num_rows, num_cols), dtype=np.uint16)
    u = np.zeros((num_rows >> 1, num_cols >> 1), dtype=np.uint16)
    v = np.zeros((num_rows >> 1, num_cols >> 1), dtype=np.uint16)
    with open(infile, "rb") as fin:
        # read the luma plane
        for row in range(num_rows):
            for col in range(num_cols):
                value_packed = fin.read(2)
                y[row][col] = ((value_packed[1] << 8) | (value_packed[0])) & 0x03ff
        # read the u plane
        for row in range(num_rows >> 1):
            for col in range(num_cols >> 1):
                value_packed = fin.read(2)
                u[row][col] = ((value_packed[1] << 8) | (value_packed[0])) & 0x03ff
        # read the v plane
        for row in range(num_rows >> 1):
            for col in range(num_cols >> 1):
                value_packed = fin.read(2)
                v[row][col] = ((value_packed[1] << 8) | (value_packed[0])) & 0x03ff
    return y, u, v


def write_ndarray_to_yuv420p10le(outfile, y, u, v, num_cols, num_rows):
    with open(outfile, "wb") as fout:
        # write the luma plane
        for row in range(num_rows):
            for col in range(num_cols):
                fout.write(int(y[row][col]).to_bytes(2, byteorder="little"))
        # read the u plane
        for row in range(num_rows >> 1):
            for col in range(num_cols >> 1):
                fout.write(int(u[row][col]).to_bytes(2, byteorder="little"))
        # read the v plane
        for row in range(num_rows >> 1):
            for col in range(num_cols >> 1):
                fout.write(int(v[row][col]).to_bytes(2, byteorder="little"))
    return


# yuv420p10le parser
def parse(infile, num_cols, num_rows, pattern, debug):
    # read the input file
    y, u, v = read_yuv420p10le_to_ndarray(infile, num_cols, num_rows)
    # get the average representation for each color
    num_bands = len(PATTERN_COLORS[pattern])
    delta_rows = num_rows // num_bands
    for cur_band in range(num_bands):
        # get the top and bottom rows
        row_top = delta_rows * cur_band
        row_bot = (delta_rows * (cur_band + 1)) - 1
        row_mid = (row_top + row_bot) >> 1
        yavg, ystd = np.average(y[row_mid]), np.std(y[row_mid])
        uavg, ustd = np.average(u[row_mid >> 1]), np.std(u[row_mid >> 1])
        vavg, vstd = np.average(v[row_mid >> 1]), np.std(v[row_mid >> 1])
        # print results
        cur_color = PATTERN_COLORS[pattern][cur_band]
        print(f"{cur_color:10}   Y: {int(yavg):4}   U: {int(uavg):4}   V: {int(vavg):4}    Ystd: {ystd} Ustd: {ustd} Vstd: {vstd}")


def convert_range(matrix_in, imin, imax, omin, omax):
    convert_fun = lambda x: (x - imin) * ((omax - omin) / (imax - imin)) + omin
    matrix_out = np.vectorize(convert_fun)(matrix_in)
    # clip the output matrix
    matrix_out = matrix_out.clip(0, 65536)
    # round the output matrix
    # https://stackoverflow.com/a/43920513
    matrix_out = np.around(matrix_out)
    # enforce the type
    matrix_out = matrix_out.astype(np.uint16)
    return matrix_out


# yuv420p10le range conversion
def range_convert(infile, outfile, num_cols, num_rows, range_conversion, debug):
    # read input image
    y, u, v = read_yuv420p10le_to_ndarray(infile, num_cols, num_rows)
    # convert range for each component
    if range_conversion in ("fr2fr", "fr2lr"):
        # input is FR
        yimin, yimax = 0, 1023
        cimin, cimax = 0, 1023
    else:
        # input is LR
        yimin, yimax = 64, 940
        cimin, cimax = 64, 960
    if range_conversion in ("fr2fr", "lr2fr"):
        # output is FR
        yomin, yomax = 0, 1023
        comin, comax = 0, 1023
    else:
        # output is LR
        yomin, yomax = 64, 940
        comin, comax = 64, 960
    yc = convert_range(y, yimin, yimax, yomin, yomax)
    uc = convert_range(u, cimin, cimax, comin, comax)
    vc = convert_range(v, cimin, cimax, comin, comax)
    # write input image
    write_ndarray_to_yuv420p10le(outfile, yc, uc, vc, num_cols, num_rows)


# MIPI-RAW10-RGGB generator
def generate(outfile, num_cols, num_rows, pattern, debug):
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
    parser.add_argument(
        "--range-conversion",
        action="store",
        type=str,
        dest="range_conversion",
        default=default_values["range_conversion"],
        choices=RANGE_CONVERSION_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                RANGE_CONVERSION_LIST,
            )
        ),
        help="range conversion arg",
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
        "func",
        type=str,
        nargs="?",
        default=default_values["func"],
        choices=FUNC_CHOICES.keys(),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FUNC_CHOICES.items())),
    )
    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-j",
        "--infile2",
        action="store",
        type=str,
        dest="infile2",
        default=default_values["infile2"],
        metavar="input-file-2",
        help="input file 2",
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
    if options.func == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)

    # get infile/outfile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    if options.func == "generate":
        generate(options.outfile, options.width, options.height, options.pattern, options.debug)

    elif options.func == "parse":
        parse(options.infile, options.width, options.height, options.pattern, options.debug)

    elif options.func == "range-convert":
        range_convert(options.infile, options.outfile, options.width, options.height, options.range_conversion, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
