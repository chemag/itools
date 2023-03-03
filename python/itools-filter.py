#!/usr/bin/env python3

"""filter.py module description.

Runs generic image transformation on input images.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import argparse
import cv2
import itertools
import numpy as np
import os
import sys


DEFAULT_NOISE_LEVEL = 50

FILTER_CHOICES = {
    "help": "show help options",
    "copy": "copy input to output",
    "gray": "convert image to GRAY scale",
    "xchroma": "swap chromas",
    "noise": "add noise",
    "diff": "diff 2 frames",
    "compose": "compose 2 frames",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "help",
    "noise_level": DEFAULT_NOISE_LEVEL,
    "x": 10,
    "y": 20,
    "infile": None,
    "infile2": None,
    "outfile": None,
}


def image_to_gray(infile, outfile, debug):
    # load the input image
    inimg = cv2.imread(cv2.samples.findFile(infile))
    # convert to gray
    tmpimg = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)
    outimg = cv2.cvtColor(tmpimg, cv2.COLOR_GRAY2BGR)
    # store the output image
    cv2.imwrite(outfile, outimg)


def swap_xchroma(infile, outfile, debug):
    # load the input image
    inimg = cv2.imread(cv2.samples.findFile(infile))
    # swap chromas
    yuvimg = cv2.cvtColor(inimg, cv2.COLOR_BGR2YCrCb)
    yuvimg = yuvimg[:, :, [0, 2, 1]]
    outimg = cv2.cvtColor(yuvimg, cv2.COLOR_YCrCb2BGR)
    # store the output image
    cv2.imwrite(outfile, outimg)


def add_noise(infile, outfile, noise_level, debug):
    # load the input image
    inimg = cv2.imread(cv2.samples.findFile(infile))
    # convert to gray
    noiseimg = np.random.randint(
        -noise_level, noise_level, size=inimg.shape, dtype=np.int16
    )
    outimg = inimg + noiseimg
    outimg[outimg > np.iinfo(np.uint8).max] = np.iinfo(np.uint8).max
    outimg[outimg < np.iinfo(np.uint8).min] = np.iinfo(np.uint8).min
    outimg = outimg.astype(np.uint8)
    # store the output image
    cv2.imwrite(outfile, outimg)


def diff_images(infile1, infile2, outfile, debug):
    # load the input images
    inimg1 = cv2.imread(cv2.samples.findFile(infile1))
    inimg2 = cv2.imread(cv2.samples.findFile(infile2))
    # diff them
    diffimg = np.absolute(inimg1.astype(np.int16) - inimg2.astype(np.int16)).astype(
        np.uint8
    )
    # diff them
    # remove the color components
    tmpimg = cv2.cvtColor(diffimg, cv2.COLOR_BGR2GRAY)
    outimg = cv2.cvtColor(tmpimg, cv2.COLOR_GRAY2BGR)
    # reverse the colors, so darker means more change
    outimg = 255 - outimg
    # store the output image
    cv2.imwrite(outfile, outimg)


# composes infile2 on top of infile1, at (xloc, yloc)
# uses alpha
def compose_images(infile1, infile2, xloc, yloc, outfile, debug):
    # load the input images
    inimg1 = cv2.imread(cv2.samples.findFile(infile1))
    inimg2 = cv2.imread(cv2.samples.findFile(infile2), cv2.IMREAD_UNCHANGED)
    # compose them
    width1, height1, _ = inimg1.shape
    width2, height2, _ = inimg2.shape
    assert xloc + width2 < width1
    assert yloc + height2 < height1
    if inimg2.shape[2] == 3:
        # no alpha channel: just use 50% ((im1 + im2) / 2)
        outimg = inimg1.astype(np.int16)
        outimg[yloc : yloc + height2, xloc : xloc + width2] += inimg2
        outimg[yloc : yloc + height2, xloc : xloc + width2] /= 2

    elif inimg2.shape[2] == 4:
        outimg = inimg1.astype(np.int16)
        for (x2, y2) in itertools.product(range(width2), range(height2)):
            x1 = xloc + x2
            y1 = yloc + y2
            alpha_value = inimg2[y2][x2][3] / 256
            outimg[y1][x1] = np.rint(
                outimg[y1][x1] * (1 - alpha_value) + inimg2[y2][x2][:3] * alpha_value
            )

    # store the output image
    outimg = outimg.astype(np.uint8)
    cv2.imwrite(outfile, outimg)


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
        "--noise-level",
        action="store",
        type=int,
        dest="noise_level",
        default=default_values["noise_level"],
        help="Noise Level",
    )
    parser.add_argument(
        "-x",
        action="store",
        type=int,
        dest="x",
        default=default_values["x"],
        help="Composition X Coordinate",
    )
    parser.add_argument(
        "-y",
        action="store",
        type=int,
        dest="y",
        default=default_values["y"],
        help="Composition Y Coordinate",
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
        "infile",
        type=str,
        nargs="?",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-i",
        "--infile2",
        action="store",
        type=str,
        dest="infile2",
        default=default_values["infile2"],
        metavar="input-file2",
        help="input file 2",
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
    # implement help
    if options.filter == "help":
        parser.print_help()
        sys.exit(0)
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

    if options.filter == "diff":
        outimg = diff_images(
            options.infile, options.infile2, options.outfile, options.debug
        )

    elif options.filter == "compose":
        outimg = compose_images(
            options.infile,
            options.infile2,
            options.x,
            options.y,
            options.outfile,
            options.debug,
        )

    elif options.filter == "gray":
        image_to_gray(options.infile, options.outfile, options.debug)

    elif options.filter == "xchroma":
        swap_xchroma(options.infile, options.outfile, options.debug)

    elif options.filter == "noise":
        add_noise(options.infile, options.outfile, options.noise_level, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
