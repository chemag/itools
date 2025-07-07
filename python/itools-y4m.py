#!/usr/bin/env python3

"""itools-y4m.py module description.

Runs generic y4m I/O. Similar to python-y4m.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import importlib
import numpy as np
import os.path
import sys

itools_common = importlib.import_module("itools-common")

FRAME_INDICATOR = b"FRAME\n"


def color_range_conversion(inyvu, input_colorrange, output_colorrange, colorspace):
    # get components
    ya = inyvu[:, :, 0]
    va = inyvu[:, :, 1]
    ua = inyvu[:, :, 2]
    ya, ua, va, status = color_range_conversion_components(
        ya,
        ua,
        va,
        input_colorrange,
        output_colorrange,
        colorspace,
    )
    outyvu = np.stack((ya, va, ua), axis=2)
    return outyvu


def color_range_conversion_components(
    ya, ua, va, input_colorrange, output_colorrange, colorspace
):
    ya, ya_broken = luma_range_conversion(
        ya,
        colorspace,
        src=input_colorrange,
        dst=output_colorrange,
    )
    ua, ua_broken = chroma_range_conversion(
        ua,
        colorspace,
        src=input_colorrange,
        dst=output_colorrange,
    )
    va, va_broken = chroma_range_conversion(
        va,
        colorspace,
        src=input_colorrange,
        dst=output_colorrange,
    )
    status = {}
    status["y4m:ybroken"] = int(ya_broken)
    status["y4m:ubroken"] = int(ua_broken)
    status["y4m:vbroken"] = int(va_broken)
    status["y4m:broken"] = int(ya_broken or ua_broken or va_broken)
    return ya, ua, va, status


def do_range_conversion(inarr, srcmin, srcmax, dstmin, dstmax, dt):
    # conversion function is $yout = a * yin + b$
    # Conversion requirements:
    # * (1) dstmin = a * srcmin + b
    # * (2) dstmax = a * srcmax + b
    a = (dstmax - dstmin) / (srcmax - srcmin)
    b = dstmin - a * srcmin
    f = lambda yin: np.round(a * yin + b)
    outarr = f(inarr)
    # look for invalid values
    broken_range = False
    if len(outarr[outarr < dstmin]) > 0 or len(outarr[outarr > dstmax]) > 0:
        # strictly speaking, this y4m is wrong
        broken_range = True

    # clip values
    outarr[outarr < dstmin] = dstmin
    outarr[outarr > dstmax] = dstmax
    outarr = outarr.astype(dt)
    return outarr, broken_range


def luma_range_conversion(ya, colorspace, src, dst):
    color_depth = itools_common.COLORSPACES[colorspace]["depth"]
    if color_depth == itools_common.ColorDepth.depth_10:
        srcmin, srcmax = (0, 1023) if src.name == "full" else (64, 940)
        dstmin, dstmax = (0, 1023) if dst.name == "full" else (64, 940)
        return do_range_conversion(ya, srcmin, srcmax, dstmin, dstmax, np.uint16)
    elif color_depth == itools_common.ColorDepth.depth_8:
        srcmin, srcmax = (0, 255) if src.name == "full" else (16, 235)
        dstmin, dstmax = (0, 255) if dst.name == "full" else (16, 235)
        return do_range_conversion(ya, srcmin, srcmax, dstmin, dstmax, np.uint8)


def chroma_range_conversion(va, colorspace, src, dst):
    color_depth = itools_common.COLORSPACES[colorspace]["depth"]
    if color_depth == itools_common.ColorDepth.depth_10:
        srcmin, srcmax = (0, 1023) if src.name == "full" else (64, 960)
        dstmin, dstmax = (0, 1023) if dst.name == "full" else (64, 960)
        return do_range_conversion(va, srcmin, srcmax, dstmin, dstmax, np.uint16)
    elif color_depth == itools_common.ColorDepth.depth_8:
        srcmin, srcmax = (0, 255) if src.name == "full" else (16, 240)
        dstmin, dstmax = (0, 255) if dst.name == "full" else (16, 240)
        return do_range_conversion(va, srcmin, srcmax, dstmin, dstmax, np.uint8)


class Y4MFileReader:
    VALID_INTERLACED = ("p", "t", "b", "m")
    VALID_COLORRANGES = ("FULL", "LIMITED")
    DEFAULT_COLORSPACE = "420"

    def __init__(self, infile, output_colorrange=None, debug=0):
        # store the input parameters
        self.infile = infile
        self.output_colorrange = itools_common.ColorRange.parse(output_colorrange)
        self.debug = debug
        # open the file descriptor
        self.fin = open(self.infile, "rb")
        # read the header line
        header_line = self.fin.readline()
        self.parse_header_line(header_line)

    def __del__(self):
        self.fin.close()

    def parse_header_line(self, header_line):
        parameters = header_line.decode("ascii").split(" ")
        assert (
            parameters[0] == "YUV4MPEG2"
        ), "invalid y4m file: starts with {parameters[0]}"
        # ensure all the required parameters exist
        assert "W" in (
            val[0] for val in parameters[1:]
        ), "error: no width parameter in y4m header"
        assert "H" in (
            val[0] for val in parameters[1:]
        ), "error: no height parameter in y4m header"
        assert "F" in (
            val[0] for val in parameters[1:]
        ), "error: no frame-rate parameter in y4m header"
        # default parameters
        height = width = framerate = interlaced = aspect = colorspace = None
        comments = {}
        # parse parameters
        for v in parameters[1:]:
            key, val = v[0], v[1:]
            if key == "W":
                width = int(val)
            elif key == "H":
                height = int(val)
            elif key == "F":
                framerate = val
            elif key == "I":
                interlaced = val
                assert (
                    interlaced in self.VALID_INTERLACED
                ), f"error: invalid interlace: {interlace}"
            elif key == "A":
                aspect = val
            elif key == "C":
                colorspace = val
                assert (
                    colorspace in itools_common.COLORSPACES.keys()
                ), f"error: invalid colorspace: {colorspace}"
            elif key == "X":
                key2, val2 = val.split("=")
                if key2 == "COLORRANGE":
                    colorrange = val2
                    assert (
                        colorrange in self.VALID_COLORRANGES
                    ), f"error: invalid colorrange: {colorrange}"
                comments[key2] = val2.strip()
        self.height = height
        self.width = width
        self.framerate = framerate
        self.interlaced = interlaced
        self.aspect = aspect
        self.colorspace = (
            colorspace if colorspace is not None else self.DEFAULT_COLORSPACE
        )
        self.comments = comments
        # derived values
        self.input_colorrange = itools_common.ColorRange.parse(
            self.comments.get("COLORRANGE")
        )
        self.chroma_subsample = itools_common.COLORSPACES[self.colorspace][
            "chroma_subsample"
        ]
        self.input_colordepth = itools_common.COLORSPACES[self.colorspace]["depth"]
        if self.debug > 0:
            print(
                f"debug: y4m frame read with input_colorrange: {input_colorrange.name}",
            )
        self.status = {
            "y4m:colorrange": self.input_colorrange.name,
            "y4m:broken": 0,
            "colorrange_input": self.input_colorrange,
            "colorrange": self.output_colorrange.name,
            "colordepth": self.input_colordepth,
        }

    def get_frame_size(self):
        # TODO(chema): this is broken with odd dimensions
        if self.colorspace.startswith("420"):
            return self.width * self.height * 3 // 2
        if self.colorspace.startswith("422"):
            return self.width * self.height * 2
        if self.colorspace.startswith("444"):
            return self.width * self.height * 3
        raise f"only support 420, 422, 444 colorspaces (not {self.colorspace})"

    def read_frame(self):
        # 1. read the "FRAME\n" tidbit
        frame_line = self.fin.read(len(FRAME_INDICATOR))
        if len(frame_line) == 0:
            # no more frames
            return None
        assert (
            frame_line == FRAME_INDICATOR
        ), f"error: invalid frame indicator: '{frame_line}'"
        # 2. get the exact frame size
        # 2.1. get the number of pixels
        luma_size_pixels = self.width * self.height
        # process chroma subsampling
        if self.chroma_subsample == itools_common.ChromaSubsample.chroma_420:
            chroma_w_pixels = self.width >> 1
            chroma_h_pixels = self.height >> 1
        elif self.chroma_subsample == itools_common.ChromaSubsample.chroma_422:
            chroma_w_pixels = self.width >> 1
            chroma_h_pixels = self.height
        elif self.chroma_subsample == itools_common.ChromaSubsample.chroma_444:
            chroma_w_pixels = self.width
            chroma_h_pixels = self.height
        elif self.chroma_subsample == itools_common.ChromaSubsample.chroma_400:
            chroma_w_pixels = 0
            chroma_h_pixels = 0
        chroma_size_pixels = chroma_w_pixels * chroma_h_pixels
        # 2.2. get the pixel depth
        if self.input_colordepth == itools_common.ColorDepth.depth_8:
            dt = np.dtype(np.uint8)
            luma_size = luma_size_pixels
            chroma_size = chroma_size_pixels
        elif self.input_colordepth == itools_common.ColorDepth.depth_10:
            dt = np.dtype(np.uint16)
            luma_size = 2 * luma_size_pixels
            chroma_size = 2 * chroma_size_pixels
        # 3. read the exact frame size
        ya = np.fromfile(self.fin, dtype=dt, count=luma_size).reshape(
            self.height, self.width
        )
        ua = np.fromfile(self.fin, dtype=dt, count=chroma_size).reshape(
            chroma_h_pixels, chroma_w_pixels
        )
        va = np.fromfile(self.fin, dtype=dt, count=chroma_size).reshape(
            chroma_h_pixels, chroma_w_pixels
        )
        # 4. undo chroma subsample in order to combine same-size matrices
        ua_full = itools_common.chroma_subsample_reverse(ya, ua, self.colorspace)
        va_full = itools_common.chroma_subsample_reverse(ya, va, self.colorspace)
        # 5. fix color range if needed
        if (
            self.output_colorrange is not None
            and self.output_colorrange is not itools_common.ColorRange.unspecified
            and self.input_colorrange is not itools_common.ColorRange.unspecified
            and self.output_colorrange != self.input_colorrange
        ):
            ya, ua_full, va_full, tmp_status = color_range_conversion_components(
                ya,
                ua_full,
                va_full,
                self.input_colorrange,
                self.output_colorrange,
                self.colorspace,
            )
            status.update(tmp_status)
        # 6. stack the components
        # note that OpenCV conversions use YCrCb (YVU) instead of YCbCr (YUV)
        outyvu = np.stack((ya, va_full, ua_full), axis=2)
        return outyvu


def read_y4m_image(infile, output_colorrange=None, debug=0):
    # read the y4m file
    y4m_file_reader = Y4MFileReader(infile, output_colorrange, debug)
    # read one frame
    frame = y4m_file_reader.read_frame()
    return frame


class Y4MFileWriter:
    SUPPORTED_COLORSPACES = (
        "mono",
        "420",
        "444",
        "mono10",
        "420p10",
        "444p10",
    )

    def __init__(
        self, height, width, colorspace, colorrange, outfile, extcs=None, debug=0
    ):
        # store the input parameters
        self.height = height
        self.width = width
        assert (
            colorspace in self.SUPPORTED_COLORSPACES
        ), f"error: unsupported {colorspace=}"
        self.colorspace = colorspace
        self.colorrange = colorrange
        self.outfile = outfile
        self.extcs = extcs
        self.debug = debug
        self.fout = open(outfile, "wb")
        self.write_header()

    def __del__(self):
        self.fout.close()

    def get_header(self):
        header = (
            f"YUV4MPEG2 W{self.width} H{self.height} F25:1 Ip A0:0 C{self.colorspace}"
        )
        if self.colorrange in (
            itools_common.ColorRange.limited,
            itools_common.ColorRange.full,
        ):
            colorrange_str = itools_common.ColorRange.to_str(self.colorrange).upper()
            header += f" XCOLORRANGE={colorrange_str}"
        if self.extcs is not None:
            header += f" XEXTCS={self.extcs}"
        header += "\n"
        return header.encode("utf-8")

    def write_header(self):
        header = self.get_header()
        self.fout.write(header)

    def write_frame(self, outyvu):
        # 1. write frame line
        self.fout.write(FRAME_INDICATOR)
        # 2. write grayscale
        if self.colorspace in ("mono", "mono10"):
            self.fout.write(outyvu.flatten())
            return
        # 3. write y
        ya = outyvu[:, :, 0]
        self.fout.write(ya.flatten())
        # 4. write u (implementing chroma subsample)
        ua_full = outyvu[:, :, 2]
        if self.colorspace in ("420", "420p10"):
            ua = itools_common.chroma_subsample_direct(ua_full, self.colorspace)
        elif self.colorspace in ("444", "444p10"):
            ua = ua_full
        self.fout.write(ua.flatten())
        # 5. write v (implementing chroma subsample)
        va_full = outyvu[:, :, 1]
        if self.colorspace in ("420", "420p10"):
            va = itools_common.chroma_subsample_direct(va_full, self.colorspace)
        elif self.colorspace in ("444", "444p10"):
            va = va_full
        self.fout.write(va.flatten())


def write_y4m_image(
    outfile,
    outyvu,
    colorspace="420",
    colorrange=itools_common.ColorRange.full,
    extcs=None,
    debug=0,
):
    # 1. get file dimensions from frame
    try:
        height, width, _ = outyvu.shape
    except ValueError:
        height, width = outyvu.shape
    # 2. write header
    y4m_file_writer = Y4MFileWriter(
        height, width, colorspace, colorrange, outfile, extcs, debug
    )
    # 3. write frame
    y4m_file_writer.write_frame(outyvu)
