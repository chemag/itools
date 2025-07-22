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
itools_y4m = importlib.import_module("itools-y4m")
itools_version = importlib.import_module("itools-version")


FILTER_CHOICES = {
    "help": "show help options",
    "components": "get the average/stddev of all the components",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "cleanup": 1,
    "header": True,
    "roi_x0": None,
    "roi_y0": None,
    "roi_x1": None,
    "roi_y1": None,
    "roi_dump": None,
    "filter": "components",
    "infile_list": [],
    "outfile": None,
    "logfile": None,
}


SUMMARY_FIELDS_SINGLE = ("pix_fmt",)


SUMMARY_FIELDS_AVERAGE = ("delta_timestamp_ms",)


# calculate average/stddev of all components
def get_components(infile, roi, roi_dump, config_dict, cleanup, logfd, debug):
    if debug > 0:
        print(f"debug: analyzing {infile}", file=logfd)
    # load the input image as both yuv and rgb
    inbgr, inyvu, status = itools_io.read_image_file(
        infile,
        config_dict,
        proc_color=itools_common.ProcColor.both,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    read_image_components = config_dict.get("read_image_components")
    if read_image_components:
        # calculate the coordinates
        (roi_x0, roi_y0), (roi_x1, roi_y1) = roi
        roi_x0 = 0 if roi_x0 is None else roi_x0
        roi_y0 = 0 if roi_y0 is None else roi_y0
        roi_x1 = inbgr.shape[1] if roi_x1 is None else roi_x1
        roi_y1 = inbgr.shape[0] if roi_y1 is None else roi_y1
        # get the requested component: note that options are YVU or BGR
        yd, vd, ud = (
            inyvu[roi_y0:roi_y1, roi_x0:roi_x1, 0],
            inyvu[roi_y0:roi_y1, roi_x0:roi_x1, 1],
            inyvu[roi_y0:roi_y1, roi_x0:roi_x1, 2],
        )
        ymean, ystddev = yd.mean(), yd.std()
        umean, ustddev = ud.mean(), ud.std()
        vmean, vstddev = vd.mean(), vd.std()
        bd, gd, rd = (
            inbgr[roi_y0:roi_y1, roi_x0:roi_x1, 0],
            inbgr[roi_y0:roi_y1, roi_x0:roi_x1, 1],
            inbgr[roi_y0:roi_y1, roi_x0:roi_x1, 2],
        )
        bmean, bstddev = bd.mean(), bd.std()
        gmean, gstddev = gd.mean(), gd.std()
        rmean, rstddev = rd.mean(), rd.std()
    # store results
    columns = [
        "filename",
        "size",
    ]
    columns_image = [
        "roi",
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
    if read_image_components:
        columns += columns_image
    columns += list(status.keys())
    df = pd.DataFrame(columns=columns)
    size = os.path.getsize(infile)
    row = [
        infile,
        size,
    ]
    if read_image_components:
        row += [
            f"({roi_x0} {roi_y0}) ({roi_x1} {roi_y1})",
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
        ]
    row += status.values()
    df.loc[df.size] = row
    if roi_dump is not None:
        outyvu = itools_common.yuv_planar_to_yuv_cv2(yd, ud, vd)
        itools_y4m.write_y4m_image(roi_dump, outyvu, colorspace="444")

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
        "--cleanup",
        action="store_const",
        dest="cleanup",
        const=1,
        default=default_values["cleanup"],
        help="Cleanup Raw Files%s"
        % (" [default]" if default_values["cleanup"] == 1 else ""),
    )
    parser.add_argument(
        "--full-cleanup",
        action="store_const",
        dest="cleanup",
        const=2,
        default=default_values["cleanup"],
        help="Cleanup All Files%s"
        % (" [default]" if default_values["cleanup"] == 2 else ""),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_const",
        dest="cleanup",
        const=0,
        help="Do Not Cleanup Files%s"
        % (" [default]" if not default_values["cleanup"] == 0 else ""),
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
        "--roi-x0",
        dest="roi_x0",
        type=int,
        default=default_values["roi_x0"],
        help="ROI x0",
    )
    parser.add_argument(
        "--roi-y0",
        dest="roi_y0",
        type=int,
        default=default_values["roi_y0"],
        help="ROI y0",
    )
    parser.add_argument(
        "--roi-x1",
        dest="roi_x1",
        type=int,
        default=default_values["roi_x1"],
        help="ROI x1",
    )
    parser.add_argument(
        "--roi-y1",
        dest="roi_y1",
        type=int,
        default=default_values["roi_y1"],
        help="ROI y1",
    )
    parser.add_argument(
        "--roi-dump",
        action="store",
        type=str,
        dest="roi_dump",
        default=default_values["roi_dump"],
        help="File where to dump ROI array",
    )
    itools_common.Config.set_parser_options(parser)
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
        metavar="input-file-list",
        help="input file list",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        dest="outfile",
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
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get logfile descriptor
    if options.logfile is None:
        logfd = sys.stdout
    else:
        logfd = open(options.logfile, "w")
    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")
    # create configuration
    config_dict = itools_common.Config.Create(options)
    # process infile
    if options.filter == "components":
        # process input files
        df = None
        roi = ((options.roi_x0, options.roi_y0), (options.roi_x1, options.roi_y1))
        for infile in options.infile_list:
            dftmp = get_components(
                infile,
                roi,
                options.roi_dump,
                config_dict,
                options.cleanup,
                logfd,
                options.debug,
            )
            df = dftmp if df is None else pd.concat([df, dftmp])
        df.to_csv(options.outfile, header=options.header, index=False)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
