#!/usr/bin/env python3

"""itools-y4m.py module description.

Runs generic y4m I/O. Similar to python-y4m.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import numpy as np
import os.path


# converts a chroma-subsampled matrix into a non-chroma subsampled one
# Algo is very simple (just dup values)
def chroma_subsample_reverse(inmatrix, colorspace):
    in_w, in_h = inmatrix.shape
    if colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
        out_w = in_w << 1
        out_h = in_h << 1
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix[::2, ::2] = inmatrix
        outmatrix[1::2, ::2] = inmatrix
        outmatrix[::2, 1::2] = inmatrix
        outmatrix[1::2, 1::2] = inmatrix
    elif colorspace in ("422",):
        out_w = in_w << 1
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix[::, ::2] = inmatrix
        outmatrix[::, 1::2] = inmatrix
    elif colorspace in ("444",):
        out_w = in_w
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix = inmatrix
    return outmatrix


# converts a non-chroma-subsampled matrix into a chroma subsampled one
# Algo is very simple (just average values)
def chroma_subsample_direct(inmatrix, colorspace):
    in_w, in_h = inmatrix.shape
    if colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
        out_w = in_w >> 1
        out_h = in_h >> 1
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint16)
        outmatrix += inmatrix[::2, ::2]
        outmatrix += inmatrix[1::2, ::2]
        outmatrix += inmatrix[::2, 1::2]
        outmatrix += inmatrix[1::2, 1::2]
        outmatrix = outmatrix / 4
        outmatrix = outmatrix.astype(np.uint8)
    elif colorspace in ("422",):
        out_w = in_w >> 1
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint16)
        outmatrix += inmatrix[::, ::2]
        outmatrix += inmatrix[::, 1::2]
        outmatrix = outmatrix / 2
        outmatrix = outmatrix.astype(np.uint8)
    elif colorspace in ("444",):
        out_w = in_w
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix = inmatrix
    return outmatrix


class Y4MHeader:
    VALID_INTERLACED = ("p", "t", "b", "m")
    VALID_COLORSPACES = ("420", "420jpeg", "420paldv", "420mpeg2", "422", "444")
    VALID_COLORRANGES = ("FULL", "LIMITED")

    def __init__(
        self, width, height, framerate, interlaced, aspect, colorspace, comment
    ):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.interlaced = interlaced
        self.aspect = aspect
        self.colorspace = colorspace
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

    def read_frame(self, data):
        # read "FRAME\n" tidbit
        assert data[:6] == b"FRAME\n", f"error: invalid FRAME: starts with {data[:6]}"
        offset = 6
        # read luminance
        dt = np.dtype(np.uint8)
        luma_size = self.width * self.height
        ya = np.frombuffer(data[offset : offset + luma_size], dtype=dt).reshape(
            self.width, self.height
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
            chroma_w, chroma_h
        )
        offset += chroma_size
        va = np.frombuffer(data[offset : offset + chroma_size], dtype=dt).reshape(
            chroma_w, chroma_h
        )
        offset += chroma_size
        # combine the color components
        # undo chroma subsample in order to combine same-size matrices
        ua_full = chroma_subsample_reverse(ua, self.colorspace)
        va_full = chroma_subsample_reverse(va, self.colorspace)
        # note that OpenCV conversions use YCrCb (YVU) instead of YCbCr (YUV)
        outyvu = np.stack((ya, va_full, ua_full), axis=2)
        return outyvu, offset


def read_y4m(infile):
    # read the y4m frame
    with open(infile, "rb") as fin:
        # read y4m header
        data = fin.read()
        header, offset = Y4MHeader.read(data)
        # read y4m frame
        frame, offset = header.read_frame(data[offset:])
        return frame, header, offset


def write_header(width, height, colorspace):
    return f"YUV4MPEG2 W{width} H{height} F30000:1001 Ip C{colorspace}\n"


def write_y4m(outfile, outyvu, colorspace="420"):
    with open(outfile, "wb") as fout:
        # write header
        width, height, _ = outyvu.shape
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
        ua = chroma_subsample_direct(ua_full, colorspace)
        fout.write(ua.flatten())
        # write v (implementing chroma subsample)
        va_full = outyvu[:, :, 1]
        va = chroma_subsample_direct(va_full, colorspace)
        fout.write(va.flatten())
