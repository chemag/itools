#!/usr/bin/env python3

"""itools-y4m.py module description.

Runs generic y4m I/O. Similar to python-y4m.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import importlib
import numpy as np
import os.path

itools_common = importlib.import_module("itools-common")


def color_range_conversion(inyvu, input_colorrange, output_colorrange, colorspace):
    # get components
    ya = inyvu[:, :, 0]
    va = inyvu[:, :, 1]
    ua = inyvu[:, :, 2]
    ya, ua, va, status = color_range_conversion_components(
        ya, ua, va, input_colorrange, output_colorrange, colorspace,
    )
    outyvu = np.stack((ya, va, ua), axis=2)
    return outyvu


def color_range_conversion_components(ya, ua, va, input_colorrange, output_colorrange, colorspace):
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


def do_range_conversion(inarr, srcmin, srcmax, dstmin, dstmax):
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


def luma_range_conversion(ya, colorspace, src, dst):
    if colorspace in ("420p10"):
        srcmin, srcmax = (0, 1023) if src.name == "full" else (64, 940)
        dstmin, dstmax = (0, 1023) if dst.name == "full" else (64, 940)
        return do_range_conversion(ya, srcmin, srcmax, dstmin, dstmax, np.uint16)
    else:
        srcmin, srcmax = (0, 255) if src.name == "full" else (16, 235)
        dstmin, dstmax = (0, 255) if dst.name == "full" else (16, 235)
        return do_range_conversion(ya, srcmin, srcmax, dstmin, dstmax, np.uint8)


def chroma_range_conversion(va, colorspace, src, dst):
    if colorspace in ("420p10"):
        srcmin, srcmax = (0, 1023) if src.name == "full" else (64, 960)
        dstmin, dstmax = (0, 1023) if dst.name == "full" else (64, 960)
        return do_range_conversion(va, srcmin, srcmax, dstmin, dstmax, np.uint16)
    else:
        srcmin, srcmax = (0, 255) if src.name == "full" else (16, 240)
        dstmin, dstmax = (0, 255) if dst.name == "full" else (16, 240)
        return do_range_conversion(va, srcmin, srcmax, dstmin, dstmax, np.uint8)

class Y4MHeader:
    VALID_INTERLACED = ("p", "t", "b", "m")
    VALID_COLORSPACES = ("420", "420jpeg", "420paldv", "420mpeg2", "422", "444", "420p10")
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

    def read_frame(self, data, output_colorrange, debug):
        # 1. read "FRAME\n" tidbit
        assert data[:6] == b"FRAME\n", f"error: invalid FRAME: starts with {data[:6]}"
        offset = 6

        dt = np.dtype(np.uint8)
        luma_size = self.width * self.height

        if self.colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
            chroma_w = self.width >> 1
            chroma_h = self.height >> 1
        elif self.colorspace in ("422",):
            chroma_w = self.width >> 1
            chroma_h = self.height
        elif self.colorspace in ("444",):
            chroma_w = self.width
            chroma_h = self.height
        elif self.colorspace in ("420p10",):
            chroma_w = self.width >> 1
            chroma_h = self.height >> 1
            luma_size *= 2
            dt = np.dtype(np.uint16)

        chroma_size = chroma_w * chroma_h
        if self.colorspace in ("420p10",):
            chroma_size *= 2

        # 2. read luminance
        ya = np.frombuffer(data[offset : offset + luma_size], dtype=dt).reshape(
            self.height, self.width
        )
        offset += luma_size
        # 3. read chromas
        ua = np.frombuffer(data[offset : offset + chroma_size], dtype=dt).reshape(
            chroma_h, chroma_w
        )
        offset += chroma_size
        va = np.frombuffer(data[offset : offset + chroma_size], dtype=dt).reshape(
            chroma_h, chroma_w
        )
        offset += chroma_size
        # 4. combine the color components
        # undo chroma subsample in order to combine same-size matrices
        ua_full = itools_common.chroma_subsample_reverse(ua, self.colorspace)
        va_full = itools_common.chroma_subsample_reverse(va, self.colorspace)
        # 5. fix color range
        input_colorrange = itools_common.ColorRange.parse(
            self.comment.get("COLORRANGE")
        )
        if debug > 0:
            print(
                f"debug: y4m frame read with input_colorrange: {input_colorrange.name}"
            )
        status = {
            "y4m:colorrange": input_colorrange.name,
            "y4m:broken": 0,
        }
        if (
            output_colorrange is not None
            and output_colorrange is not itools_common.ColorRange.unspecified
            and input_colorrange is not itools_common.ColorRange.unspecified
            and output_colorrange != input_colorrange
        ):
            if debug > 0:
                print(
                    f"debug: Y4MHeader.read_frame() converting colorrange from {input_colorrange.name} to {output_colorrange.name}"
                )
            ya, ua_full, va_full, tmp_status = color_range_conversion_components(
                ya, ua_full, va_full, input_colorrange, output_colorrange, self.colorspace
            )
            status.update(tmp_status)
        status["colorrange"] = itools_common.ColorRange.parse(status["y4m:colorrange"])
        # 6. stack the components
        # note that OpenCV conversions use YCrCb (YVU) instead of YCbCr (YUV)
        outyvu = np.stack((ya, va_full, ua_full), axis=2)
        return outyvu, offset, status


def read_y4m(infile, output_colorrange=None, debug=0):
    # read the y4m frame
    with open(infile, "rb") as fin:
        # read y4m header
        data = fin.read()
        header, offset = Y4MHeader.read(data)
        # read y4m frame
        frame, offset, status = header.read_frame(
            data[offset:], output_colorrange, debug
        )
        return frame, header, offset, status


def write_header(width, height, colorspace, colorrange):
    header = f"YUV4MPEG2 W{width} H{height} F30000:1001 Ip C{colorspace}"
    if colorrange in (
        itools_common.ColorRange.limited,
        itools_common.ColorRange.full,
    ):
        colorrange_str = itools_common.ColorRange.to_str(colorrange).upper()
        header += f" XCOLORRANGE={colorrange_str}"
    header += "\n"
    return header


def write_y4m(
    outfile, outyvu, colorspace="420", colorrange=itools_common.ColorRange.full
):
    with open(outfile, "wb") as fout:
        # write header
        height, width, _ = outyvu.shape
        header = write_header(width, height, colorspace, colorrange)
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
