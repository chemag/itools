#!/usr/bin/env python3

"""Module to convert (raw) Bayer (CFA) image pixel formats.

Supported formats:
* all ffmpeg formats
* all Linux V4L2 formats
* some MIPI-RAW formats.

Notes:
* ffmpeg only supports 8 Bayer formats (12 when considering that the 16-bit
  formats exist in both BE and LE flavors). We want to allow converting
  other Bayer formats to any of the ffmpeg ones. Main goal is to allow
  ffmpeg access to generic Bayer formats.
"""


import argparse
import cv2
import enum
import importlib
import numpy as np
import os
import sys

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")
itools_bayer = importlib.import_module("itools-bayer")


default_values = {
    "debug": 0,
    "dry_run": False,
    "o_pix_fmt": None,
    "infile": None,
    "outfile": None,
}


# assume Y4M files with EXTCS metadata
class BayerY4MReader:

    def __init__(
        self,
        infile,
        y4m_file_reader,
        debug=0,
    ):
        # input elements
        self.infile = infile
        self.y4m_file_reader = y4m_file_reader
        self.debug = debug
        # derived elements
        # ensure that the image is annotated
        assert (
            "EXTCS" in self.y4m_file_reader.extension_dict.keys()
        ), f"error: monochrome image does not contain extended color space (EXTCS)"
        i_pix_fmt = self.y4m_file_reader.extension_dict["EXTCS"]
        i_pix_fmt = itools_bayer.get_canonical_input_pix_fmt(i_pix_fmt)
        assert (
            i_pix_fmt in itools_bayer.BAYER_FORMATS
        ), f"error: unknown extended color space: {i_pix_fmt}"
        self.i_pix_fmt = i_pix_fmt
        self.layout = itools_bayer.BAYER_FORMATS[self.i_pix_fmt]["layout"]
        self.order = itools_bayer.BAYER_FORMATS[self.i_pix_fmt]["order"]
        # check the color space is supported
        assert self.y4m_file_reader.colorspace in (
            "mono",
            "mono10",
            "mono12",
            "mono14",
            "mono16",
            "yuv420",
            "yuv420p10",
            "yuv444",
            "yuv444p10",
        ), f"error: invalid y4m colorspace: {self.y4m_file_reader.colorspace}"
        # TODO(chema): support only mono for now
        assert (
            self.y4m_file_reader.colorspace in itools_common.MONO_COLORSPACES
        ), f"error: unsupported y4m colorspace: {self.y4m_file_reader.colorspace}"

    def __del__(self):
        # clean up
        del self.y4m_file_reader

    @classmethod
    def FromY4MFile(cls, infile, debug=0):
        # read the video header
        y4m_file_reader = itools_y4m.Y4MFileReader(
            infile, output_colorrange=None, debug=debug
        )
        return cls(infile, y4m_file_reader, debug)

    def GetFrame(self, debug=0):
        buf_raw = self.y4m_file_reader.read_frame_raw()
        if buf_raw is None:
            return None
        # plug the buffer into a bayer image
        width = self.y4m_file_reader.width
        height = self.y4m_file_reader.height
        pix_fmt = self.y4m_file_reader.extension_dict["EXTCS"]
        colordepth = itools_bayer.get_depth(pix_fmt)
        infile = self.infile
        # adjust height/width parameters
        if self.y4m_file_reader.colorspace == "mono" and colordepth > 8:
            # this is a wide-depth signal in an 8-bit container
            width = int(width / itools_bayer.get_width_adjustment(pix_fmt))
        # create the BayerImage object
        return itools_bayer.BayerImage.FromBuffer(
            buf_raw, width, height, pix_fmt, infile, debug
        )


class BayerY4MWriter:

    # assume Y4M source only for now
    def __init__(
        self,
        outfile,
        y4m_file_writer,
        height,
        width,
        colorspace,
        colorrange,
        o_pix_fmt,
        debug=0,
    ):
        # input elements
        self.outfile = outfile
        self.y4m_file_writer = y4m_file_writer
        self.height = height
        self.width = width
        self.colorspace = colorspace
        self.colorrange = colorrange
        self.o_pix_fmt = o_pix_fmt
        self.debug = debug

    def __del__(self):
        # clean up
        del self.y4m_file_writer

    @classmethod
    def ToY4MFile(cls, outfile, height, width, colorrange, o_pix_fmt, debug=0):
        # canonicalize the pixel format
        o_pix_fmt = itools_bayer.get_canonical_input_pix_fmt(o_pix_fmt)
        # set the colorspace from the pixel format
        y4m_colorspace = itools_bayer.BAYER_FORMATS[o_pix_fmt].get(
            "y4m", itools_bayer.DEFAULT_Y4M_COLORSPACE
        )
        # create the EXTCS header
        extension_dict = {"EXTCS": o_pix_fmt}
        # adjust height/width parameters
        colordepth = itools_bayer.get_depth(o_pix_fmt)
        if y4m_colorspace == "mono" and colordepth > 8:
            # this is a wide-depth signal in an 8-bit container
            width = int(width * itools_bayer.get_width_adjustment(o_pix_fmt))
        # create a writer and write the header
        y4m_file_writer = itools_y4m.Y4MFileWriter(
            height, width, y4m_colorspace, colorrange, outfile, extension_dict, debug
        )
        # create the video object
        return cls(
            outfile,
            y4m_file_writer,
            height,
            width,
            y4m_colorspace,
            colorrange,
            o_pix_fmt,
            debug,
        )

    def AddFrame(self, bayer_image):
        # convert input frame to the write pixel format
        bayer_image_copy = bayer_image.Copy(self.o_pix_fmt, self.debug)
        # write up to file
        buffer = bayer_image_copy.GetBuffer()
        self.y4m_file_writer.write_frame_raw(buffer)
        return bayer_image_copy


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
    output_choices_str = " | ".join(itools_bayer.O_PIX_FMT_LIST)
    parser.add_argument(
        "--o_pix_fmt",
        action="store",
        type=str,
        dest="o_pix_fmt",
        default=default_values["o_pix_fmt"],
        choices=itools_bayer.O_PIX_FMT_LIST,
        metavar=f"[{output_choices_str}]",
        help="output pixel format",
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


def convert_video_file(infile, outfile, o_pix_fmt, debug):
    bayer_video_reader = BayerY4MReader.FromY4MFile(infile, debug)
    bayer_video_writer = None
    num_frames = 0
    while True:
        # read the frame
        bayer_image = bayer_video_reader.GetFrame()
        if bayer_image is None:
            break
        # create the writer if needed
        if bayer_video_writer is None:
            height = bayer_image.height
            width = bayer_image.width
            colorrange = bayer_video_reader.y4m_file_reader.input_colorrange
            bayer_video_writer = BayerY4MWriter.ToY4MFile(
                outfile, height, width, colorrange, o_pix_fmt, debug
            )
        # write the frame
        bayer_video_writer.AddFrame(bayer_image)
        num_frames += 1
    if debug > 0:
        print(f"convert {infile=} {outfile=} {o_pix_fmt=} {num_frames=}")


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
        print(f"debug: {options}")

    convert_video_file(
        options.infile,
        options.outfile,
        options.o_pix_fmt,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
