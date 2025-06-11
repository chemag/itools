#!/usr/bin/env python3

"""itools-alpha-enctool.py module description.

This is a tool to test Alpha image encoding.
"""


import argparse
import cv2
import importlib
import io
import itertools
import json
import math
import numpy as np
import os
import pandas as pd
import PIL
import pillow_heif
import re
import shutil
import sys
import tempfile

itools_version = importlib.import_module("itools-version")
itools_common = importlib.import_module("itools-common")
itools_alpha = importlib.import_module("itools-alpha")
itools_y4m = importlib.import_module("itools-y4m")


default_values = {
    "debug": 0,
    "dry_run": False,
    "cleanup": 1,
    "codec_list": ",".join(str(k) for k in itools_alpha.CODEC_LIST),
    "workdir": tempfile.gettempdir(),
    "add_average": True,
    "psnr_infinity": True,
    "infile_list": None,
    "outfile": None,
}


COLUMN_LIST = [
    "infile",
    "height",
    "width",
    "depth",
    "codec",
    "raw_size",
    "encoded_size",
    "encoded_bpp",
    "encoded_cr",
    "psnr",
    "aepp",
    "stats",
]


def get_average_results(df):
    # import the results
    new_df = pd.DataFrame(columns=list(df.columns.values))
    for codec in list(df["codec"].unique()):
        # select interesting data
        tmp_df = df[(df["codec"] == codec)]
        if tmp_df.size == 0:
            # no entries with this (codec, ) combo
            continue
        # start with empty data
        derived_dict = {key: np.nan for key in list(df.columns.values)}
        derived_dict["infile"] = "average"
        derived_dict["codec"] = codec
        # average a few columns
        COLUMNS_MEAN = (
            "height",
            "width",
            "depth",
            "raw_size",
            "encoded_size",
            "encoded_bpp",
            "encoded_cr",
            "psnr",
            "aepp",
        )
        for key in COLUMNS_MEAN:
            derived_dict[key] = tmp_df[key].mean()
        new_df.loc[new_df.size] = list(derived_dict.values())
    return new_df


def process_data(
    infile_list,
    codec_list,
    workdir,
    outfile,
    add_average,
    psnr_infinity,
    cleanup,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)

    # run all the input files
    block_size = 8
    depth = 8
    for infile, codec in itertools.product(infile_list, codec_list):
        # 1. encode the file into an alpha channel
        alpha_file = tempfile.NamedTemporaryFile(
            prefix=f"itools-alpha-enctools.codec_{codec}.", suffix=".alpha"
        ).name
        itools_alpha.encode_file(infile, alpha_file, codec, block_size, debug)
        # 2. decode the alpha file
        out_y4m_file = tempfile.NamedTemporaryFile(
            prefix=f"itools-alpha-enctools.codec_{codec}.", suffix=".alpha.y4m"
        ).name
        stats = itools_alpha.decode_file(alpha_file, out_y4m_file, debug)
        # 3. calculate the error
        outyvu1 = itools_y4m.read_y4m_image(
            infile,
            output_colorrange=itools_common.ColorRange.full,
            debug=debug,
        )
        yarray1 = outyvu1[:, :, 0]
        height1, width1 = yarray1.shape
        outyvu2 = itools_y4m.read_y4m_image(
            out_y4m_file,
            output_colorrange=itools_common.ColorRange.full,
            debug=debug,
        )
        yarray2 = outyvu2[:, :, 0]
        height2, width2 = yarray2.shape
        psnr = itools_common.calculate_psnr_planar(
            yarray1, yarray2, depth, psnr_infinity
        )
        aepp = itools_common.calculate_aepp_planar(yarray1, yarray2)
        # 4. calculate results
        raw_size = os.path.getsize(infile)
        encoded_size = os.path.getsize(alpha_file)
        encoded_bpp = 8.0 * encoded_size / (width1 * height1)
        encoded_cr = raw_size / encoded_size
        stats_str = ":".join(str(count) for count in stats.values())
        df.loc[df.size] = (
            infile,
            height1,
            width1,
            depth,
            codec,
            raw_size,
            encoded_size,
            encoded_bpp,
            encoded_cr,
            psnr,
            aepp,
            stats_str,
        )

    # 2. get average results
    if add_average:
        derived_df = get_average_results(df)
        df = pd.concat([df, derived_df], ignore_index=True, axis=0)

    # 4. write the results
    df.to_csv(outfile, index=False)


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
        "--add-average",
        action="store_true",
        dest="add_average",
        default=default_values["add_average"],
        help="Add average%s" % (" [default]" if default_values["add_average"] else ""),
    )
    parser.add_argument(
        "--no-add-average",
        action="store_false",
        dest="add_average",
        help="Do not add average%s"
        % (" [default]" if not default_values["add_average"] else ""),
    )
    parser.add_argument(
        "--psnr-infinity",
        action="store_true",
        dest="psnr_infinity",
        default=default_values["psnr_infinity"],
        help="Use infinity in PSNR%s"
        % (" [default]" if default_values["psnr_infinity"] else ""),
    )
    parser.add_argument(
        "--no-psnr-infinity",
        action="store_false",
        dest="psnr_infinity",
        help="Do not use infinity in PSNR%s"
        % (" [default]" if not default_values["psnr_infinity"] else ""),
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
        "--codec",
        action="store",
        type=str,
        dest="codec_list",
        default=default_values["codec_list"],
        metavar="[%s]"
        % (
            " | ".join(
                list(
                    itools_alpha.CODEC_LIST
                    + [
                        "all",
                    ]
                ),
            )
        ),
        help="codec",
    )
    parser.add_argument(
        "--codec-list",
        dest="show_codec_list",
        action="store_true",
        default=False,
        help="List available codecs and exit",
    )
    parser.add_argument(
        "--workdir",
        action="store",
        dest="workdir",
        type=str,
        default=default_values["workdir"],
        metavar="Work directory",
        help="work directory",
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
    # do the parsing
    options = parser.parse_args(argv[1:])
    # parse quick options
    if options.show_codec_list:
        print(f"list of valid codecs: {itools_alpha.CODEC_LIST}")
        sys.exit()
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # set workdir
    if options.workdir is not None:
        os.makedirs(options.workdir, exist_ok=True)
        tempfile.tempdir = options.workdir
    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")
    # fix comma-separated lists
    options.codec_list = options.codec_list.split(",")
    if "all" in options.codec_list:
        options.codec_list = itools_alpha.CODEC_LIST
    # process infile
    process_data(
        options.infile_list,
        options.codec_list,
        options.workdir,
        options.outfile,
        options.add_average,
        options.psnr_infinity,
        options.cleanup,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
