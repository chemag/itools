#!/usr/bin/env python3

"""itools-y4m.py module description.

Runs generic y4m I/O. Similar to python-y4m.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import importlib
import numpy as np
import os.path

itools_common = importlib.import_module("itools-common")


def range_conversion(inarr, srcmin, srcmax, dstmin, dstmax):
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
    if (
        len(outarr[outarr < np.iinfo(np.uint8).min]) > 0
        or len(outarr[outarr > np.iinfo(np.uint8).max]) > 0
    ):
        # strictly speaking, this y4m is wrong
        broken_range = True
    # clip values
    outarr[outarr < np.iinfo(np.uint8).min] = np.iinfo(np.uint8).min
    outarr[outarr > np.iinfo(np.uint8).max] = np.iinfo(np.uint8).max
    outarr = outarr.astype(np.uint8)
    return outarr, broken_range


def luma_range_conversion(ya, src, dst):
    srcmin = 0 if src == "FULL" else 16
    dstmin = 0 if dst == "FULL" else 16
    srcmax = 255 if src == "FULL" else 235
    dstmax = 255 if dst == "FULL" else 235
    return range_conversion(ya, srcmin, srcmax, dstmin, dstmax)


def chroma_range_conversion(va, src, dst):
    srcmin = 0 if src == "FULL" else 16
    dstmin = 0 if dst == "FULL" else 16
    srcmax = 255 if src == "FULL" else 240
    dstmax = 255 if dst == "FULL" else 240
    return range_conversion(va, srcmin, srcmax, dstmin, dstmax)


class Y4MHeader:
    VALID_INTERLACED = ("p", "t", "b", "m")
    VALID_COLORSPACES = ("420", "420jpeg", "420paldv", "420mpeg2", "422", "444")
    VALID_COLORRANGES = ("FULL", "LIMITED")
    DEFAULT_COLORSPACE = "420"

    def __init__(
        self, width, height, framerate, interlaced, aspect, colorspace, comment
    ):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.interlaced = interlaced
        self.aspect = aspect
        self.colorspace = (
            colorspace if colorspace is not None else self.DEFAULT_COLORSPACE
        )
        self.comment = comment

    @classmethod
    def parse(cls, header_line):
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
        width = height = framerate = interlaced = aspect = colorspace = None
        comment = {}
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
                    interlaced in cls.VALID_INTERLACED
                ), f"error: invalid interlace: {interlace}"
            elif key == "A":
                aspect = val
            elif key == "C":
                colorspace = val
                assert (
                    colorspace in cls.VALID_COLORSPACES
                ), f"error: invalid colorspace: {colorspace}"
            elif key == "X":
                key2, val2 = val.split("=")
                if key2 == "COLORRANGE":
                    colorrange = val2
                    assert (
                        colorrange in cls.VALID_COLORRANGES
                    ), f"error: invalid colorrange: {colorrange}"
                comment[key2] = val2
        return Y4MHeader(
            width, height, framerate, interlaced, aspect, colorspace, comment
        )

    @classmethod
    def read(cls, data):
        # parameters are in the first line
        header_line = data.split(b"\n", 1)[0]
        header = Y4MHeader.parse(header_line)
        offset = len(header_line) + 1
        return header, offset

    def get_frame_size(self):
        if self.colorspace.startswith("420"):
            return self.width * self.height * 3 // 2
        if self.colorspace.startswith("422"):
            return self.width * self.height * 2
        if self.colorspace.startswith("444"):
            return self.width * self.height * 3
        raise f"only support 420, 422, 444 colorspaces (not {self.colorspace})"

    def read_frame(self, data, colorrange, debug):
        # read "FRAME\n" tidbit
        assert data[:6] == b"FRAME\n", f"error: invalid FRAME: starts with {data[:6]}"
        offset = 6
        # read luminance
        dt = np.dtype(np.uint8)
        luma_size = self.width * self.height
        ya = np.frombuffer(data[offset : offset + luma_size], dtype=dt).reshape(
            self.height, self.width
        )
        offset += luma_size
        # read chromas
        if self.colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
            chroma_w = self.width >> 1
            chroma_h = self.height >> 1
        elif self.colorspace in ("422",):
            chroma_w = self.width >> 1
            chroma_h = self.height
        elif self.colorspace in ("444",):
            chroma_w = self.width
            chroma_h = self.height
        chroma_size = chroma_w * chroma_h
        ua = np.frombuffer(data[offset : offset + chroma_size], dtype=dt).reshape(
            chroma_h, chroma_w
        )
        offset += chroma_size
        va = np.frombuffer(data[offset : offset + chroma_size], dtype=dt).reshape(
            chroma_h, chroma_w
        )
        offset += chroma_size
        # combine the color components
        # undo chroma subsample in order to combine same-size matrices
        ua_full = itools_common.chroma_subsample_reverse(ua, self.colorspace)
        va_full = itools_common.chroma_subsample_reverse(va, self.colorspace)
        if debug > 0:
            print(f"debug: y4m frame read with {self.comment.get('COLORRANGE', None)}")
        status = {
            "y4m:colorrange": self.comment.get("COLORRANGE", "default").lower(),
            "y4m:broken": 0,
        }
        if colorrange is not None and colorrange.upper() != self.comment.get(
            "COLORRANGE", None
        ):
            ya, ya_broken = luma_range_conversion(
                ya,
                src=self.comment.get("COLORRANGE", None),
                dst=colorrange.upper(),
            )
            ua_full, ua_broken = chroma_range_conversion(
                ua_full,
                src=self.comment.get("COLORRANGE", None),
                dst=colorrange.upper(),
            )
            va_full, va_broken = chroma_range_conversion(
                va_full,
                src=self.comment.get("COLORRANGE", None),
                dst=colorrange.upper(),
            )
            status["y4m:ybroken"] = int(ya_broken)
            status["y4m:ubroken"] = int(ua_broken)
            status["y4m:vbroken"] = int(va_broken)
            status["y4m:broken"] = int(ya_broken or ua_broken or va_broken)
        # note that OpenCV conversions use YCrCb (YVU) instead of YCbCr (YUV)
        outyvu = np.stack((ya, va_full, ua_full), axis=2)
        return outyvu, offset, status


def read_y4m(infile, colorrange=None, debug=0):
    # read the y4m frame
    with open(infile, "rb") as fin:
        # read y4m header
        data = fin.read()
        header, offset = Y4MHeader.read(data)
        # read y4m frame
        frame, offset, status = header.read_frame(data[offset:], colorrange, debug)
        return frame, header, offset, status


def write_header(width, height, colorspace, colorrange="FULL"):
    return f"YUV4MPEG2 W{width} H{height} F30000:1001 Ip C{colorspace} XCOLORRANGE={colorrange}\n"


def write_y4m(outfile, outyvu, colorspace="420"):
    with open(outfile, "wb") as fout:
        # write header
        height, width, _ = outyvu.shape
        header = write_header(width, height, colorspace)
        fout.write(header.encode("utf-8"))
        # write frame line
        frame = "FRAME\n"
        fout.write(frame.encode("utf-8"))
        # write y
        ya = outyvu[:, :, 0]
        fout.write(ya.flatten())
        # write u (implementing chroma subsample)
        ua_full = outyvu[:, :, 2]
        ua = itools_common.chroma_subsample_direct(ua_full, colorspace)
        fout.write(ua.flatten())
        # write v (implementing chroma subsample)
        va_full = outyvu[:, :, 1]
        va = itools_common.chroma_subsample_direct(va_full, colorspace)
        fout.write(va.flatten())
