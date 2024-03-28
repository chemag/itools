#!/usr/bin/env python3

"""itools-analysis.py module description.

Analyzes a series of image files.
"""
# https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/3.1.0/d7/d9e/tutorial_video_write.html


import argparse
import math
import numpy as np
import os
import pandas as pd
import sys
import importlib

itools_common = importlib.import_module("itools-common")
itools_io = importlib.import_module("itools-io")
itools_heif = importlib.import_module("itools-heif")
itools_version = importlib.import_module("itools-version")


FILTER_CHOICES = {
    "help": "show help options",
    "components": "get the average/stddev of all the components",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "header": True,
    "read_exif_info": True,
    "read_icc_info": True,
    "filter": "components",
    "infile_list": [],
    "outfile": None,
}


SUMMARY_FIELDS_SINGLE = ("pix_fmt",)


SUMMARY_FIELDS_AVERAGE = ("delta_timestamp_ms",)


# calculate average/stddev of all components
def get_components(infile, read_exif_info, read_icc_info, debug):
    if debug > 0:
        print(f"analyzing {infile}")
    # load the input image as both yuv and rgb
    inyvu, status = itools_io.read_image_file(
        infile,
        return_type=itools_common.ProcColor.yvu,
        read_exif_info=read_exif_info,
        read_icc_info=read_icc_info,
        debug=debug,
    )
    inbgr, _ = itools_io.read_image_file(infile)
    # get the requested component: note that options are YVU or BGR
    yd, vd, ud = inyvu[:, :, 0], inyvu[:, :, 1], inyvu[:, :, 2]
    ymean, ystddev = yd.mean(), yd.std()
    umean, ustddev = ud.mean(), ud.std()
    vmean, vstddev = vd.mean(), vd.std()
    bd, gd, rd = inbgr[:, :, 0], inbgr[:, :, 1], inbgr[:, :, 2]
    bmean, bstddev = bd.mean(), bd.std()
    gmean, gstddev = gd.mean(), gd.std()
    rmean, rstddev = rd.mean(), rd.std()
    # store results
    columns = [
        "filename",
        "ymean",
        "ystddev",
        "umean",
        "ustddev",
        "vmean",
        "vstddev",
        "rmean",
        "rstddev",
        "gmean",
        "gstddev",
        "bmean",
        "bstddev",
    ]
    columns += list(status.keys())
    df = pd.DataFrame(columns=columns)
    df.loc[df.size] = [
        infile,
        ymean,
        ystddev,
        umean,
        ustddev,
        vmean,
        vstddev,
        rmean,
        rstddev,
        gmean,
        gstddev,
        bmean,
        bstddev,
        *status.values(),
    ]
    return df


# process input
def get_colorimetry(infile, debug):
    df = None

    # 1. run generic analysis

    # 2. run colorimetry analysis for HEIC images

    return df


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
        action="version",
        version=itools_version.__version__,
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
        "--no-header",
        action="store_const",
        const=False,
        dest="header",
        default=default_values["header"],
        help="Do not read CSV header from first row (even if no #)",
    )
    parser.add_argument(
        "--exif",
        dest="read_exif_info",
        action="store_true",
        default=default_values["read_exif_info"],
        help="Parse EXIF Info%s"
        % (" [default]" if default_values["read_exif_info"] else ""),
    )
    parser.add_argument(
        "--no-exif",
        dest="read_exif_info",
        action="store_false",
        help="Do not parse EXIF Info%s"
        % (" [default]" if not default_values["read_exif_info"] else ""),
    )
    parser.add_argument(
        "--icc",
        dest="read_icc_info",
        action="store_true",
        default=default_values["read_icc_info"],
        help="Parse ICC Info%s"
        % (" [default]" if default_values["read_icc_info"] else ""),
    )
    parser.add_argument(
        "--no-icc",
        dest="read_icc_info",
        action="store_false",
        help="Do not parse ICC Info%s"
        % (" [default]" if not default_values["read_icc_info"] else ""),
    )
    parser.add_argument(
        "--filter",
        action="store",
        type=str,
        dest="filter",
        default=default_values["filter"],
        choices=FILTER_CHOICES.keys(),
        metavar="{%s}" % (" | ".join("{}".format(k) for k in FILTER_CHOICES.keys())),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FILTER_CHOICES.items())),
    )
    parser.add_argument(
        dest="infile_list",
        type=str,
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
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

    if options.filter == "components":
        # process input files
        df = None
        for infile in options.infile_list:
            dftmp = get_components(
                infile, options.read_exif_info, options.read_icc_info, options.debug
            )
            df = dftmp if df is None else pd.concat([df, dftmp])
        df.to_csv(options.outfile, header=options.header, index=False)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
