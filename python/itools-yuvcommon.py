#!/usr/bin/env python3

"""itools-yuvcommon.py module description.

Common YUV/RGB code.
"""

from array import array
import sys


PIX_FMTS = ("yuv420p", "nv12", "rgb24", "rgba", "yuv444p", "yuyv422")


def scale_fr2lr_16_235(x):
    # values outside the full range are not valid
    if x <= 0:
        return 16
    if x >= 255:
        return 235
    # f(x) = a * x + b
    # f(0) = a * 0 + b = 16
    b = 16.0
    # f(255) = a * 255 + b = 235
    # a = (235 - b) / 255
    a = 0.8588235294117647
    return int(a * x + b)


def scale_lr2fr_16_235(x):
    # values outside the limited range are not scaled
    if x <= 16:
        return 16
    if x >= 235:
        return 235
    # f(x) = a * x + b
    # f(16) = a * 16 + b = 0
    # f(235) = a * 235 + b = 255
    # a = 255 / (235 - 16)
    a = 1.1643835616438356
    # b = -a * 16
    b = -18.63013698630137
    return int(a * x + b)


def scale_fr2lr_16_240(x):
    # values outside the full range are not valid
    if x <= 0:
        return 16
    if x >= 255:
        return 240
    # f(x) = a * x + b
    # f(0) = a * 0 + b = 16
    b = 16.0
    # f(255) = a * 255 + b = 240
    # a = (240 - b) / 255
    a = 0.8784313725490196
    return int(a * x + b)


def scale_lr2fr_16_240(x):
    # values outside the limited range are not scaled
    if x <= 16:
        return 16
    if x >= 240:
        return 240
    # f(x) = a * x + b
    # f(16) = a * 16 + b = 0
    # f(240) = a * 240 + b = 255
    # a = 255 / (240 - 16)
    a = 1.1383928571428572
    # b = -a * 16
    b = -18.214285714285715
    return int(a * x + b)


def rgb_fr2lr(r, g, b):
    return scale_fr2lr_16_235(r), scale_fr2lr_16_235(g), scale_fr2lr_16_235(b)


def rgb_lr2fr(r, g, b):
    # studio RGB range uses only the range [16, 235]) for R, G, B
    return scale_lr2fr_16_235(r), scale_lr2fr_16_235(g), scale_lr2fr_16_235(b)


DO_NOT_NORMALIZE = False


def normalize(val):
    if DO_NOT_NORMALIZE:
        return int(val)
    return 0 if val < 0 else (255 if val > 255 else int(val))


def is_yuv(pix_fmt):
    if pix_fmt in ("yuv420p", "nv12", "yuv444p", "yuyv422"):
        return True
    elif pix_fmt in ("rgb24", "rgba"):
        return False
    # unsupported pix_fmt
    print("error: unsupported format: %s" % pix_fmt)
    sys.exit(-1)


# get frame size/luma size ratio
def get_length_factor(pix_fmt):
    if pix_fmt in ("yuv420p", "nv12"):
        return 1.5
    elif pix_fmt in ("yuyv422"):
        return 2
    elif pix_fmt in ("yuv444p", "rgb24"):
        return 3
    elif pix_fmt in ("rgba"):
        return 4
    # unsupported pix_fmt
    print("error: unsupported format: %s" % pix_fmt)
    sys.exit(-1)


def read_image(infile, w, h, pix_fmt, frame_number=0):
    data = array("B")
    # calculate the frame size
    frame_size = int(w * h * get_length_factor(pix_fmt))
    with open(infile, "rb") as fin:
        # seek to the right frame (for multi-frame videos)
        if frame_number > 0:
            fin.seek(frame_number * frame_size)
        data.fromfile(fin, frame_size)
    return data


def get_component_locations(i, j, w, h, pix_fmt):
    if pix_fmt == "yuv420p":
        # planar format, 4:2:0
        return (
            (w * j) + i,
            (w * h) + ((w // 2) * (j // 2)) + (i // 2),
            (w * h) + ((w // 2) * (h // 2)) + ((w // 2) * (j // 2)) + (i // 2),
        )
    elif pix_fmt == "nv12":
        # semi-planar format, 4:2:0
        return (
            (w * j) + i,
            (w * h) + ((w // 2) * 2 * (j // 2)) + 2 * (i // 2),
            (w * h) + ((w // 2) * 2 * (j // 2)) + 2 * (i // 2) + 1,
        )
    elif pix_fmt == "yuv444p":
        # planar format, 4:4:4, no alpha channel
        return ((w * j) + i, (w * h) + (w * j) + i, 2 * (w * h) + (w * j) + i)
    elif pix_fmt == "yuyv422":
        # packed format, 4:2:2, no alpha channel
        # Y00 U00 Y01 V00  Y02 U02 Y03 V02
        first_luma_in_group = i % 2 == 0
        u_shift = +1 if first_luma_in_group else -1
        v_shift = +3 if first_luma_in_group else +1
        return (
            (w * j * 2) + i * 2,
            (w * j * 2) + i * 2 + u_shift,
            (w * j * 2) + i * 2 + v_shift,
        )
    elif pix_fmt == "rgb24":
        # packed format, 4:4:4, no alpha channel
        return ((w * j * 4) + i * 4, (w * j * 4) + i * 4 + 1, (w * j * 4) + i * 4 + 2)
    elif pix_fmt == "rgba":
        # packed format, 4:4:4, includes alpha channel
        return ((w * j * 4) + i * 4, (w * j * 4) + i * 4 + 1, (w * j * 4) + i * 4 + 2)
    else:
        # unsupported pix_fmt
        print("error: unsupported format: %s" % pix_fmt)
        sys.exit(-1)
