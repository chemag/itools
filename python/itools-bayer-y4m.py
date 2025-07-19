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
    "i_pix_fmt": None,
    "colorrange": itools_common.ColorRange.full,
    "o_pix_fmt": None,
    "width": 0,
    "height": 0,
    "infile": None,
    "outfile": None,
}


class BayerY4M:

    # assuming image size is width: 4 height: 4
    # bayer (mono): y4m_width, y4m_height = 2, 8
    @classmethod
    def convertResolutionInternalToY4M(cls, width, height, y4m_colorspace, pix_fmt):
        component_type = itools_bayer.get_component_type(pix_fmt)
        colordepth = itools_bayer.get_depth(pix_fmt)
        layout = itools_bayer.BAYER_FORMATS[pix_fmt]["layout"]
        if layout == itools_bayer.LayoutType.packed:
            colordepth = itools_bayer.get_depth(pix_fmt)
            y4m_height = height
            if y4m_colorspace == "mono" and colordepth > 8:
                # this is a wide-depth signal in an 8-bit container
                y4m_width = int(width * itools_bayer.get_width_adjustment(pix_fmt))
            else:
                y4m_width = width
        elif layout == itools_bayer.LayoutType.planar:
            if component_type == itools_bayer.ComponentType.bayer:
                # bayer planes: 4x, half-width, half-height, vertical layout
                y4m_width = width >> 1
                y4m_height = height << 1
            elif component_type == itools_bayer.ComponentType.ydgcocg:
                # YDgCoCg planes: 4x, half-width, half-height, vertical layout
                y4m_width = width >> 1
                y4m_height = height << 1
            elif component_type == itools_bayer.ComponentType.rgb:
                # RGB planes: 3x, full-width, full-height, vertical layout
                y4m_width = width
                y4m_height = height * 3
            elif component_type == itools_bayer.ComponentType.yuv:
                if itools_common.is_mono_colorspace(y4m_colorspace):
                    # YUV planes: 3x, full-width, full-height, vertical layout
                    y4m_width = width
                    y4m_height = height * 3
                else:
                    y4m_width = width
                    y4m_height = height
            else:
                raise AssertionError("error: invalid component type: {component_type}")
        return y4m_width, y4m_height

    @classmethod
    def convertResolutionY4MToInternal(
        cls, y4m_width, y4m_height, y4m_colorspace, pix_fmt
    ):
        component_type = itools_bayer.get_component_type(pix_fmt)
        colordepth = itools_bayer.get_depth(pix_fmt)
        layout = itools_bayer.BAYER_FORMATS[pix_fmt]["layout"]
        if layout == itools_bayer.LayoutType.packed:
            colordepth = itools_bayer.get_depth(pix_fmt)
            height = y4m_height
            if y4m_colorspace == "mono" and colordepth > 8:
                # this is a wide-depth signal in an 8-bit container
                width = int(y4m_width / itools_bayer.get_width_adjustment(pix_fmt))
            else:
                width = y4m_width
        elif layout == itools_bayer.LayoutType.planar:
            if component_type == itools_bayer.ComponentType.bayer:
                # bayer planes: 4x, half-width, half-height, vertical layout
                width = y4m_width << 1
                height = y4m_height >> 1
            elif component_type == itools_bayer.ComponentType.ydgcocg:
                # YDgCoCg planes: 4x, half-width, half-height, vertical layout
                width = y4m_width << 1
                height = y4m_height >> 1
            elif component_type == itools_bayer.ComponentType.rgb:
                # RGB planes: 3x, full-width, full-height, vertical layout
                width = y4m_width
                height = int(y4m_height / 3)
            elif component_type == itools_bayer.ComponentType.yuv:
                if itools_common.is_mono_colorspace(y4m_colorspace):
                    # YUV planes: 3x, full-width, full-height, vertical layout
                    width = y4m_width
                    height = int(y4m_height / 3)
                else:
                    width = y4m_width
                    height = y4m_height
            else:
                raise AssertionError("error: invalid component type: {component_type}")
        return width, height


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
            "420",
            "420p10",
            "444",
            "444p10",
        ), f"error: invalid y4m colorspace: {self.y4m_file_reader.colorspace}"

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
        # convert y4m data to internal representation
        y4m_width = self.y4m_file_reader.width
        y4m_height = self.y4m_file_reader.height
        y4m_colorspace = self.y4m_file_reader.colorspace
        pix_fmt = self.y4m_file_reader.extension_dict["EXTCS"]
        pix_fmt = itools_bayer.get_canonical_output_pix_fmt(pix_fmt)
        width, height = BayerY4M.convertResolutionY4MToInternal(
            y4m_width, y4m_height, y4m_colorspace, pix_fmt
        )
        # create the BayerImage object
        return itools_bayer.BayerImage.FromBuffer(
            buf_raw, width, height, pix_fmt, self.infile, debug
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
        pix_fmt,
        debug=0,
    ):
        # input elements
        self.outfile = outfile
        self.y4m_file_writer = y4m_file_writer
        self.height = height
        self.width = width
        self.colorspace = colorspace
        self.colorrange = colorrange
        self.pix_fmt = pix_fmt
        self.debug = debug

    def __del__(self):
        # clean up
        del self.y4m_file_writer

    @classmethod
    def ToY4MFile(cls, outfile, height, width, colorrange, pix_fmt, debug=0):
        # canonicalize the pixel format
        pix_fmt = itools_bayer.get_canonical_input_pix_fmt(pix_fmt)
        # set the colorspace from the pixel format
        y4m_colorspace = itools_bayer.BAYER_FORMATS[pix_fmt].get(
            "y4m", itools_bayer.DEFAULT_Y4M_COLORSPACE
        )
        # create the EXTCS header
        extension_dict = {"EXTCS": pix_fmt}
        # adjust height/width parameters
        y4m_width, y4m_height = BayerY4M.convertResolutionInternalToY4M(
            width, height, y4m_colorspace, pix_fmt
        )
        # create a writer and write the header
        y4m_file_writer = itools_y4m.Y4MFileWriter(
            y4m_height,
            y4m_width,
            y4m_colorspace,
            colorrange,
            outfile,
            extension_dict,
            debug,
        )
        # create the video object
        return cls(
            outfile,
            y4m_file_writer,
            y4m_height,
            y4m_width,
            y4m_colorspace,
            colorrange,
            pix_fmt,
            debug,
        )

    def AddFrame(self, bayer_image):
        # convert input frame to the write pixel format
        bayer_image_copy = bayer_image.Copy(self.pix_fmt, self.debug)
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
    input_choices_str = " | ".join(itools_bayer.I_PIX_FMT_LIST)
    parser.add_argument(
        "--i_pix_fmt",
        action="store",
        type=str,
        dest="i_pix_fmt",
        default=default_values["i_pix_fmt"],
        choices=itools_bayer.I_PIX_FMT_LIST
        + [
            None,
        ],
        metavar=f"[{input_choices_str}]",
        help="input pixel format",
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
    input_choices_str = " | ".join(itools_common.ColorRange.list())
    parser.add_argument(
        "--colorrange",
        action="store",
        type=str,
        dest="colorrange",
        default=default_values["colorrange"],
        choices=itools_common.ColorRange.list()
        + [
            None,
        ],
        metavar=f"[{input_choices_str}]",
        help="input colorrange",
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


def convert_video_file(infile, outfile, pix_fmt, debug):
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
                outfile, height, width, colorrange, pix_fmt, debug
            )
        # write the frame
        bayer_video_writer.AddFrame(bayer_image)
        num_frames += 1
    if debug > 0:
        print(f"convert {infile=} {outfile=} {pix_fmt=} {num_frames=}")


def convert_raw_image_file(
    infile,
    i_pix_fmt,
    colorrange,
    width,
    height,
    outfile,
    o_pix_fmt,
    debug,
):
    # 1. read input image file
    bayer_image = itools_bayer.BayerImage.FromFile(
        infile, i_pix_fmt, width, height, debug, strict_size_check=False
    )

    # 2. convert image to new pixel format
    bayer_image_copy = bayer_image.Copy(o_pix_fmt, debug)

    # 3. create the writer
    height = bayer_image_copy.height
    width = bayer_image.width
    bayer_video_writer = BayerY4MWriter.ToY4MFile(
        outfile, height, width, colorrange, o_pix_fmt, debug
    )

    # 4. write converted image into output file
    bayer_image_copy.ToFile(outfile, debug)
    bayer_video_writer.AddFrame(bayer_image_copy)


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

    if (
        options.height != 0
        and options.width != 0
        and options.i_pix_fmt is not None
        and os.path.splitext(options.infile)[1] != ".y4m"
    ):
        # read raw file
        convert_raw_image_file(
            options.infile,
            options.i_pix_fmt,
            options.colorrange,
            options.width,
            options.height,
            options.outfile,
            options.o_pix_fmt,
            options.debug,
        )
        return

    convert_video_file(
        options.infile,
        options.outfile,
        options.o_pix_fmt,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
