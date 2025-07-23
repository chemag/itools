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
    ya, ua, va = itools_common.yuv_cv2_to_yuv_planar(inyvu)
    ya, ua, va, status = color_range_conversion_components(
        ya,
        ua,
        va,
        input_colorrange,
        output_colorrange,
        colorspace,
    )
    outyvu = itools_common.yuv_planar_to_yuv_cv2(ya, ua, va)
    return outyvu


def color_range_conversion_components(
    ya, ua, va, input_colorrange, output_colorrange, colorspace
):
    status = {}
    ya, ya_broken = luma_range_conversion(
        ya,
        colorspace,
        src=input_colorrange,
        dst=output_colorrange,
    )
    status["y4m:ybroken"] = int(ya_broken)
    ua_broken = False
    if ua is not None:
        ua, ua_broken = chroma_range_conversion(
            ua,
            colorspace,
            src=input_colorrange,
            dst=output_colorrange,
        )
        status["y4m:ubroken"] = int(ua_broken)
    va_broken = False
    if va is not None:
        va, va_broken = chroma_range_conversion(
            va,
            colorspace,
            src=input_colorrange,
            dst=output_colorrange,
        )
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
    color_depth = itools_common.Y4M_COLORSPACES[colorspace]["depth"]
    if color_depth == itools_common.ColorDepth.depth_10:
        srcmin, srcmax = (0, 1023) if src.name == "full" else (64, 940)
        dstmin, dstmax = (0, 1023) if dst.name == "full" else (64, 940)
        return do_range_conversion(ya, srcmin, srcmax, dstmin, dstmax, np.uint16)
    elif color_depth == itools_common.ColorDepth.depth_8:
        srcmin, srcmax = (0, 255) if src.name == "full" else (16, 235)
        dstmin, dstmax = (0, 255) if dst.name == "full" else (16, 235)
        return do_range_conversion(ya, srcmin, srcmax, dstmin, dstmax, np.uint8)


def chroma_range_conversion(va, colorspace, src, dst):
    color_depth = itools_common.Y4M_COLORSPACES[colorspace]["depth"]
    if color_depth == itools_common.ColorDepth.depth_10:
        srcmin, srcmax = (0, 1023) if src.name == "full" else (64, 960)
        dstmin, dstmax = (0, 1023) if dst.name == "full" else (64, 960)
        return do_range_conversion(va, srcmin, srcmax, dstmin, dstmax, np.uint16)
    elif color_depth == itools_common.ColorDepth.depth_8:
        srcmin, srcmax = (0, 255) if src.name == "full" else (16, 240)
        dstmin, dstmax = (0, 255) if dst.name == "full" else (16, 240)
        return do_range_conversion(va, srcmin, srcmax, dstmin, dstmax, np.uint8)


class Y4MFile:

    def __init__(self, debug=0):
        self.chroma_subsample = itools_common.Y4M_COLORSPACES[self.colorspace][
            "chroma_subsample"
        ]
        self.colordepth = itools_common.Y4M_COLORSPACES[self.colorspace]["depth"]
        self.input_colorrange = itools_common.ColorRange.parse(
            self.extension_dict.get("COLORRANGE", itools_common.ColorRange.unspecified)
        )
        if self.debug > 0:
            print(
                f"debug: y4m frame read with input_colorrange: {self.input_colorrange.name}",
            )
        self.status = {
            "y4m:colorrange": self.input_colorrange.name,
            "y4m:broken": 0,
            "colorrange": self.colorrange.name,
            "colordepth": self.colordepth,
        }
        # get derived info
        self.get_size_info()
        self.depth = itools_common.ColorDepth.get_depth(
            itools_common.get_y4m_depth(self.colorspace)
        )
        self.dtype = itools_common.get_dtype(self.depth)

    def get_size_info(self):
        # 1. get the number of pixels
        self.luma_size_pixels = self.width * self.height
        # process chroma subsampling
        if self.chroma_subsample == itools_common.ChromaSubsample.chroma_420:
            self.chroma_w_pixels = self.width >> 1
            self.chroma_h_pixels = self.height >> 1
        elif self.chroma_subsample == itools_common.ChromaSubsample.chroma_422:
            self.chroma_w_pixels = self.width >> 1
            self.chroma_h_pixels = self.height
        elif self.chroma_subsample == itools_common.ChromaSubsample.chroma_444:
            self.chroma_w_pixels = self.width
            self.chroma_h_pixels = self.height
        elif self.chroma_subsample == itools_common.ChromaSubsample.chroma_400:
            self.chroma_w_pixels = 0
            self.chroma_h_pixels = 0
        self.chroma_size_pixels = self.chroma_w_pixels * self.chroma_h_pixels
        # get the pixel depth
        if self.colordepth == itools_common.ColorDepth.depth_8:
            dt = np.dtype(np.uint8)
            luma_size_bytes = self.luma_size_pixels
            chroma_size_bytes = self.chroma_size_pixels
        else:
            dt = np.dtype(np.uint16)
            luma_size_bytes = 2 * self.luma_size_pixels
            chroma_size_bytes = 2 * self.chroma_size_pixels
        self.frame_size_pixels = self.luma_size_pixels + 2 * self.chroma_size_pixels
        self.frame_size_bytes = luma_size_bytes + 2 * chroma_size_bytes

    def buffer_to_array(self, buffer):
        arr = np.frombuffer(buffer, dtype=self.dtype, count=self.frame_size_pixels)
        ya = arr[0 : self.luma_size_pixels].reshape(self.height, self.width)
        ua = arr[
            self.luma_size_pixels : self.luma_size_pixels + self.chroma_size_pixels
        ].reshape(self.chroma_h_pixels, self.chroma_w_pixels)
        va = arr[self.luma_size_pixels + self.chroma_size_pixels :].reshape(
            self.chroma_h_pixels, self.chroma_w_pixels
        )
        return ya, ua, va


class Y4MFileReader(Y4MFile):
    VALID_INTERLACED = ("p", "t", "b", "m")
    VALID_COLORRANGES = ("FULL", "LIMITED")
    DEFAULT_COLORSPACE = "420"

    def __init__(self, infile, colorrange=None, debug=0):
        # store the input parameters
        self.infile = infile
        self.colorrange = itools_common.ColorRange.parse(colorrange)
        self.debug = debug
        # open the file descriptor
        self.fin = open(self.infile, "rb")
        # read the header line
        header_line = self.fin.readline()
        self.extension_dict = {}
        self.parse_header_line(header_line)
        # derived values
        super().__init__()
        self.status["colorrange_input"] = self.input_colorrange

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
        # parse parameters
        for v in parameters[1:]:
            key, val = v[0], v[1:]
            if key == "W":
                width = int(val)
            elif key == "H":
                height = int(val)
            elif key == "F":
                framerate = val.strip()
            elif key == "I":
                interlaced = val.strip()
                assert (
                    interlaced in self.VALID_INTERLACED
                ), f"error: invalid interlace: {interlace}"
            elif key == "A":
                aspect = val.strip()
            elif key == "C":
                colorspace = val.strip()
                assert (
                    colorspace in itools_common.Y4M_COLORSPACES.keys()
                ), f"error: invalid colorspace: {colorspace}"
            elif key == "X":
                key2, val2 = val.split("=")
                if key2 == "COLORRANGE":
                    colorrange = val2.strip()
                    assert (
                        colorrange in self.VALID_COLORRANGES
                    ), f"error: invalid colorrange: {colorrange}"
                self.extension_dict[key2] = val2.strip()
        self.height = height
        self.width = width
        self.framerate = framerate
        self.interlaced = interlaced
        self.aspect = aspect
        self.colorspace = (
            colorspace if colorspace is not None else self.DEFAULT_COLORSPACE
        )

    def get_frame_size(self):
        # TODO(chema): this is broken with odd dimensions
        if self.colorspace.startswith("420"):
            return self.width * self.height * 3 // 2
        if self.colorspace.startswith("422"):
            return self.width * self.height * 2
        if self.colorspace.startswith("444"):
            return self.width * self.height * 3
        raise f"only support 420, 422, 444 colorspaces (not {self.colorspace})"

    # returns the next frame, as a raw buffer
    def read_frame_raw(self):
        # 1. read the "FRAME\n" tidbit
        frame_line = self.fin.read(len(FRAME_INDICATOR))
        if len(frame_line) == 0:
            # no more frames
            return None
        assert (
            frame_line == FRAME_INDICATOR
        ), f"error: invalid frame indicator: '{frame_line}'"
        # 2. read the exact frame size
        buf = self.fin.read(self.frame_size_bytes)
        return buf

    # returns the next frame, in YVU format (OpenCV-preferred)
    def read_frame(self):
        # get the planes
        ya, va, ua = self.read_frame_planes()
        if ya is None:
            return None
        if self.chroma_subsample == itools_common.ChromaSubsample.chroma_400:
            return ya
        # upsample the frame
        ya, ua_full, va_full = itools_common.planar_upsample(
            ya, ua, va, self.chroma_subsample
        )
        # convert to CV2 YUV
        outyvu = itools_common.yuv_planar_to_yuv_cv2(ya, ua_full, va_full)
        return outyvu

    def read_frame_planes(self):
        # 1. read the raw buffer
        buffer = self.read_frame_raw()
        if buffer is None:
            return None, None, None
        # 2. convert the buffer into a set of arrays
        ya, ua, va = self.buffer_to_array(buffer)
        # 3. fix color range if needed
        if (
            self.colorrange is not None
            and self.colorrange != itools_common.ColorRange.unspecified
            and self.input_colorrange is not itools_common.ColorRange.unspecified
            and self.colorrange != self.input_colorrange
        ):
            ya, ua, va, tmp_status = color_range_conversion_components(
                ya,
                ua,
                va,
                self.input_colorrange,
                self.colorrange,
                self.colorspace,
            )
            status.update(tmp_status)
        else:
            self.colorrange = self.input_colorrange
        return ya, va, ua


def read_y4m_image(infile, output_colorrange=None, debug=0):
    # read the y4m file
    y4m_file_reader = Y4MFileReader(infile, output_colorrange, debug)
    # read one frame
    frame = y4m_file_reader.read_frame()
    return frame


class Y4MFileWriter(Y4MFile):
    SUPPORTED_COLORSPACES = (
        "mono",
        "420",
        "422",
        "444",
        "mono10",
        "420p10",
        "422p10",
        "444p10",
        "mono12",
        "mono14",
        "mono16",
    )

    def __init__(
        self,
        height,
        width,
        colorspace,
        colorrange,
        outfile,
        extension_dict={},
        debug=0,
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
        self.extension_dict = extension_dict
        self.debug = debug
        self.fout = open(outfile, "wb")
        self.write_header()
        # derived values
        super().__init__()

    def __del__(self):
        self.fout.close()

    def get_header(self, framerate="25:1", interlace="p", aspect="0:0"):
        header = f"YUV4MPEG2 W{self.width} H{self.height} F{framerate} I{interlace} A{aspect} C{self.colorspace}"
        if self.colorrange in (
            itools_common.ColorRange.limited,
            itools_common.ColorRange.full,
        ):
            colorrange_str = itools_common.ColorRange.to_str(self.colorrange).upper()
            header += f" XCOLORRANGE={colorrange_str}"
        if self.extension_dict is not None:
            for key, val in self.extension_dict.items():
                header += f" X{key}={val}"
        header += "\n"
        return header.encode("utf-8")

    def write_header(self):
        header = self.get_header()
        self.fout.write(header)

    # writes a OpenCV yvu -3 planes, each a binary string
    # @ref outyvu: WxHx3 numpy array
    def write_frame_cv2_yvu(self, outyvu, colorspace=None):
        if colorspace is None:
            colorspace = self.colorspace
        if colorspace in itools_common.MONO_COLORSPACES:
            # just flatten the grayscale
            self._write_frame_raw(outyvu.tobytes())
        else:
            ya, ua_full, va_full = itools_common.yuv_cv2_to_yuv_planar(outyvu)
            # write subsampled entry
            ya, ua, va = itools_common.planar_subsample(
                ya, ua_full, va_full, self.chroma_subsample
            )
            self._write_frame_raw(ya.tobytes() + ua.tobytes() + va.tobytes())

    # writes 1-3 planes (y, u, v), each a binary string
    # @ref buffer: 1-3 planes in binary string format, non-subsampled
    def write_frame_buffer(self, buffer):
        if self.colorspace in itools_common.MONO_COLORSPACES:
            # just write the buffer
            self._write_frame_raw(buffer)
        else:
            outyvu = self.buffer_to_cv2_yvu(buffer, self.width, self.height, self.dtype)
            self.write_frame_cv2_yvu(outyvu)

    @classmethod
    def buffer_to_cv2_yvu(cls, buffer, width, height, dtype):
        # convert to CV2 YVU for future subsampling
        plane_size_pixels = width * height
        arr = np.frombuffer(buffer, dtype=dtype, count=3 * plane_size_pixels)
        ya = arr[0:plane_size_pixels].reshape((height, width))
        ua = arr[plane_size_pixels : 2 * plane_size_pixels].reshape((height, width))
        va = arr[2 * plane_size_pixels : 3 * plane_size_pixels].reshape((height, width))
        outyvu = itools_common.yuv_planar_to_yuv_cv2(ya, ua, va)
        return outyvu

    # writes a binary buffer
    # @ref buffer: buffer
    def _write_frame_raw(self, buffer):
        # 1. write frame line
        self.fout.write(FRAME_INDICATOR)
        # 2. write buffer
        self.fout.write(buffer)


def write_y4m_image(
    outfile,
    outyvu,
    colorspace="420",
    colorrange=itools_common.ColorRange.full,
    extension_dict={},
    debug=0,
):
    # 1. get file dimensions from frame
    try:
        height, width, _ = outyvu.shape
    except ValueError:
        height, width = outyvu.shape
    # 2. write header
    y4m_file_writer = Y4MFileWriter(
        height, width, colorspace, colorrange, outfile, extension_dict, debug
    )
    # 3. write frame
    y4m_file_writer.write_frame_cv2_yvu(outyvu)
