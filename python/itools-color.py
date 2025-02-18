#!/usr/bin/env python3

"""itools-color: A yuv/r'g'b'/rgb/xyz converter.

This module converts between YUV, R'G'B', RGB, and XYZ color components.
It supports both single pixels and files.
"""

import argparse
from array import array
import math
import numpy as np
import sys

import importlib

itools_yuvcommon = importlib.import_module("itools-yuvcommon")


FUNC_CHOICES = {
    "help": "show help options",
    "image": "convert color representation in a whole image",
    "pixel": "convert color representation in a single pixel",
}


PIX_FMTS = ("yuv420p", "nv12", "rgb24", "rgba", "yuv444p", "yuyv422")
COLOR_RANGES = ("full", "limited")

H273_MATRIX_COEFFICIENTS = {
    0: {},  # identity
    1: {"Kr": 0.2126, "Kb": 0.0722},
    4: {"Kr": 0.30, "Kb": 0.11},
    5: {"Kr": 0.299, "Kb": 0.114},
    6: {"Kr": 0.299, "Kb": 0.114},
    7: {"Kr": 0.212, "Kb": 0.087},
    8: {},  # YCgCo
    9: {"Kr": 0.2627, "Kb": 0.0593},
    10: {"Kr": 0.2627, "Kb": 0.0593},
    11: {},  # YDZDX (SMPTE ST 2085 (2015))
    14: {},  # ICTCP (Rec. ITU-R BT.2100-2 ICTCP)
}

H273_MATRIX_COEFFICIENTS_LIST = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14)


# conversion data
# Jack, YIQ Color Space (page 18)
def convert_rgb2yuv_yiq(R, G, B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # NOQA: E201,E241
    I = 0.596 * R - 0.275 * G - 0.321 * B  # NOQA: E201,E241,E741
    Q = 0.212 * R - 0.523 * G + 0.311 * B  # NOQA: E201,E241
    return Y, I, Q


def convert_yuv2rgb_yiq(Y, I, Q):  # NOQA: E741
    R = Y + 0.956 * I + 0.621 * Q  # NOQA: E201,E241,E741
    G = Y - 0.272 * I - 0.647 * Q  # NOQA: E201,E241
    B = Y - 1.107 * I + 1.704 * Q  # NOQA: E201,E241
    return R, G, B


# Jack, YUV Color Space (page 18), SDTV with BT.601
# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
# TODO(chemag): broken conversion, fix me (see unittest)
def convert_rgb2yuv_sdtv_basic(R, G, B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # NOQA: E201,E241,E222
    U = -0.147 * R - 0.289 * G + 0.436 * B  # NOQA: E201,E241,E222
    V = 0.615 * R - 0.515 * G - 0.100 * B  # NOQA: E201,E241,E222
    return (
        itools_yuvcommon.normalize(Y),
        itools_yuvcommon.normalize(U),
        itools_yuvcommon.normalize(V),
    )


def convert_yuv2rgb_sdtv_basic(Y, U, V):
    R = Y + 1.140 * V  # NOQA: E201,E241,E221
    G = Y - 0.395 * U - 0.581 * V  # NOQA: E201,E241,E221
    B = Y + 2.032 * U  # NOQA: E201,E241,E221
    return (
        itools_yuvcommon.normalize(R),
        itools_yuvcommon.normalize(G),
        itools_yuvcommon.normalize(B),
    )


# Jack, YCbCr Color Space, SDTV, Analog (page 19)
def convert_rgb2yuv_ycbcr_sdtv_analog(R, G, B):
    R, G, B = R / 256.0, G / 256.0, B / 256.0
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # NOQA: E201,E241,E221,E222
    Pb = -0.169 * R - 0.331 * G + 0.500 * B  # NOQA: E201,E241,E221,E222
    Pr = 0.500 * R - 0.419 * G - 0.081 * B  # NOQA: E201,E241,E221,E222
    Y = int(256 * Y)  # NOQA: E201,E241,E221
    Cb = int(256 * Pb + 128)
    Cr = int(256 * Pr + 128)
    return Y, Cb, Cr


# Jack, YCbCr Color Space, SDTV, Analog (page 20)
def convert_yuv2rgb_ycbcr_sdtv_analog(Y, U, V):
    Y, Pb, Pr = Y / 256.0, (U - 128) / 256.0, (V - 128) / 256.0
    R = Y + 1.402 * Pr
    G = Y - 0.714 * Pr - 0.344 * Pb
    B = Y + 1.772 * Pb  # NOQA: E221
    # convert back to integer
    R = int(256 * R)
    G = int(256 * G)
    B = int(256 * B)
    return (
        itools_yuvcommon.normalize(R),
        itools_yuvcommon.normalize(G),
        itools_yuvcommon.normalize(B),
    )


# Jack, YCbCr Color Space, SDTV, Digital (page 19)
def convert_rgb2yuv_ycbcr_sdtv_digital(R, G, B):
    # converts 8-bit digital RGB data with a 16-235 nominal range (Studio RGB)
    R, G, B = itools_yuvcommon.rgb_fr2lr(R, G, B)
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # NOQA: E201,E241,E221,E222
    Cb = -0.172 * R - 0.339 * G + 0.511 * B  # NOQA: E201,E241,E221,E222
    Cr = 0.511 * R - 0.428 * G - 0.083 * B  # NOQA: E201,E241,E221,E222
    return (
        itools_yuvcommon.normalize(Y),
        itools_yuvcommon.normalize(Cb + 128),
        itools_yuvcommon.normalize(Cr + 128),
    )


# Jack, YCbCr Color Space, SDTV, Digital (page 20)
def convert_yuv2rgb_ycbcr_sdtv_digital(Y, Cb, Cr):
    R = Y + 1.371 * (Cr - 128)
    G = Y - 0.698 * (Cr - 128) - 0.336 * (Cb - 128)
    B = Y + 1.732 * (Cb - 128)
    # generated 8-bit RGB with a 16-235 nominal range (Studio RGB)
    return itools_yuvcommon.rgb_lr2fr(R, G, B)


# Jack, YCbCr Color Space, SDTV, Computer Systems FR (page 20)
def convert_rgb2yuv_ycbcr_sdtv_computer(R, G, B):
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16  # NOQA: E201,E241,E221,E222
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128  # NOQA: E201,E241,E221,E222
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128  # NOQA: E201,E241,E221,E222
    # 8-bit YCbCr and RGB data should be saturated at the 0 and 255 levels
    return (
        itools_yuvcommon.normalize(Y),
        itools_yuvcommon.normalize(Cb),
        itools_yuvcommon.normalize(Cr),
    )


def convert_yuv2rgb_ycbcr_sdtv_computer(Y, Cb, Cr):
    R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.391 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.018 * (Cb - 128)  # NOQA: E221,E501
    # 8-bit YCbCr and RGB data should be saturated at the 0 and 255 levels
    return (
        itools_yuvcommon.normalize(R),
        itools_yuvcommon.normalize(G),
        itools_yuvcommon.normalize(B),
    )


# https://en.wikipedia.org/wiki/YUV#HDTV_with_BT.709
# TODO(chemag): broken conversion, fix me (see unittest)
def convert_rgb2yuv_hdtv_basic(R, G, B):
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B  # NOQA: E201,E241,E221,E222
    U = -0.09991 * R - 0.33609 * G + 0.436 * B  # NOQA: E201,E241,E221,E222
    V = 0.615 * R - 0.55861 * G - 0.05639 * B  # NOQA: E201,E241,E221,E222
    return (
        itools_yuvcommon.normalize(Y),
        itools_yuvcommon.normalize(U),
        itools_yuvcommon.normalize(V),
    )


def convert_yuv2rgb_hdtv_basic(Y, U, V):
    R = Y + 1.28033 * V  # NOQA: E201,E241,E221
    G = Y - 0.21482 * U - 0.38059 * V  # NOQA: E201,E241,E221
    B = Y + 2.12798 * U  # NOQA: E201,E241,E221
    return (
        itools_yuvcommon.normalize(R),
        itools_yuvcommon.normalize(G),
        itools_yuvcommon.normalize(B),
    )


# Jack, YCbCr Color Space, HDTV, Analog (page 20)
def convert_rgb2yuv_ycbcr_hdtv_analog(R, G, B):
    R, G, B = R / 256.0, G / 256.0, B / 256.0
    Y = 0.213 * R + 0.715 * G + 0.072 * B  # NOQA: E201,E241,E221,E222
    Pb = -0.115 * R - 0.385 * G + 0.500 * B  # NOQA: E201,E241,E221,E222
    Pr = 0.500 * R - 0.454 * G - 0.046 * B  # NOQA: E201,E241,E221,E222
    Y = int(256 * Y)  # NOQA: E201,E241,E221
    Cb = int(256 * Pb + 128)
    Cr = int(256 * Pr + 128)
    return Y, Cb, Cr


# Jack, YCbCr Color Space, HDTV, Analog (page 21)
def convert_yuv2rgb_ycbcr_hdtv_analog(Y, U, V):
    Y, Pb, Pr = Y / 256.0, (U - 128) / 256.0, (V - 128) / 256.0
    R = Y + 1.575 * Pr
    G = Y - 0.468 * Pr - 0.187 * Pb
    B = Y + 1.856 * Pb  # NOQA: E221
    # convert back to integer
    R = int(256 * R)
    G = int(256 * G)
    B = int(256 * B)
    return (
        itools_yuvcommon.normalize(R),
        itools_yuvcommon.normalize(G),
        itools_yuvcommon.normalize(B),
    )


# Jack, YCbCr Color Space, HDTV, Digital (page 21)
def convert_rgb2yuv_ycbcr_hdtv_digital(R, G, B):
    # converts 8-bit digital RGB data with a 16-235 nominal range (Studio RGB)
    R, G, B = itools_yuvcommon.rgb_fr2lr(R, G, B)
    Y = 0.213 * R + 0.715 * G + 0.072 * B  # NOQA: E201,E241,E221,E222
    Cb = -0.117 * R - 0.394 * G + 0.511 * B + 128  # NOQA: E201,E241,E221,E222
    Cr = 0.511 * R - 0.464 * G - 0.047 * B + 128  # NOQA: E201,E241,E221,E222
    return Y, Cb, Cr


# Jack, YCbCr Color Space, HDTV, Digital (page 21)
def convert_yuv2rgb_ycbcr_hdtv_digital(Y, Cb, Cr):
    R = Y + 1.540 * (Cr - 128)
    G = Y - 0.459 * (Cr - 128) - 0.183 * (Cb - 128)
    B = Y + 1.816 * (Cb - 128)
    # generated 8-bit RGB with a 16-235 nominal range (Studio RGB)
    return itools_yuvcommon.rgb_lr2fr(R, G, B)


# Jack, YCbCr Color Space, HDTV, Computer Systems FR (page 21)
def convert_rgb2yuv_ycbcr_hdtv_computer(R, G, B):
    Y = 0.183 * R + 0.614 * G + 0.062 * B + 16  # NOQA: E201,E241,E221,E222
    Cb = -0.101 * R - 0.338 * G + 0.439 * B + 128  # NOQA: E201,E241,E221,E222
    Cr = 0.439 * R - 0.399 * G - 0.040 * B + 128  # NOQA: E201,E241,E221,E222
    # 8-bit YCbCr and RGB data should be saturated at the 0 and 255 levels
    return (
        itools_yuvcommon.normalize(Y),
        itools_yuvcommon.normalize(Cb),
        itools_yuvcommon.normalize(Cr),
    )


def convert_yuv2rgb_ycbcr_hdtv_computer(Y, Cb, Cr):
    R = 1.164 * (Y - 16) + 1.793 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.534 * (Cr - 128) - 0.213 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.115 * (Cb - 128)  # NOQA: E221,E501
    # 8-bit YCbCr and RGB data should be saturated at the 0 and 255 levels
    return (
        itools_yuvcommon.normalize(R),
        itools_yuvcommon.normalize(G),
        itools_yuvcommon.normalize(B),
    )


# YCoCg Color Space (https://en.wikipedia.org/wiki/YCoCg)
def convert_rgb2yuv_ycocg(R, G, B):
    Y = (R >> 2) + (G >> 1) + (B >> 2)  # NOQA: E221,E222
    Co = (R >> 1) - (B >> 1)  # NOQA: E221,E222
    Cg = (-R >> 2) + (G >> 1) - (B >> 2)  # NOQA: E221,E222
    # no normalization needed
    return Y, Co, Cg


def convert_yuv2rgb_ycocg(Y, Co, Cg):
    R = Y + Co + Cg  # NOQA: E221,E222
    G = Y + Cg  # NOQA: E221,E222
    B = Y - Co - Cg  # NOQA: E221,E222
    # no normalization needed
    return R, G, B


# YCoCg-R Color Space (https://en.wikipedia.org/wiki/YCoCg)
def convert_rgb2yuv_ycocgr(R, G, B):
    Co = R - B
    tmp = B + (Co >> 1)
    Cg = G - tmp
    Y = tmp + (Cg >> 1)
    # no normalization needed
    return Y, Co, Cg


def convert_yuv2rgb_ycocgr(Y, Co, Cg):
    tmp = Y - (Cg >> 1)
    G = Cg + tmp
    B = tmp - (Co >> 1)
    R = B + Co
    # no normalization needed
    return R, G, B


def h273_Clip1Y(x, BitDepthY):
    return h273_Clip3(0, (1 << BitDepthY) - 1, x)  # Equation (2)


def h273_Clip1C(x, BitDepthC):
    return h273_Clip3(0, (1 << BitDepthC) - 1, x)  # Equation (2)


def h273_Clip3(x, y, z):
    return x if z < x else (y if z > y else z)  # Equation (4) fixed


def h273_Round(x):
    return int(math.copysign(math.floor(abs(x) + 0.5), x))  # Equation (8)


# int->float conversion for Y and RGB
# TODO(chemag): add bit_depth to support != 8 bits/component
def h273_int_to_float_yrgb(x, color_range):
    if color_range == "full":
        # range is [0, 255]
        return x / 255.0
    elif color_range == "limited":
        # range is [16, 235]
        return (x - 16) / 219.0
    raise AssertionError(f"invalid color range: {color_range}")


# int->float conversion for U and V
# TODO(chemag): add bit_depth to support != 8 bits/component
def h273_int_to_float_uv(x, color_range):
    if color_range == "full":
        # range is [0, 255]
        return x / 255.0
    elif color_range == "limited":
        # range is [16, 240]
        return (x - 16) / 224.0
    raise AssertionError(f"invalid color range: {color_range}")


def h273_float_to_int_yrgb(x, color_range):
    BitDepthY = 8
    if color_range == "full":
        # range is [0, 255]
        return h273_Round((1 << (BitDepthY - 8)) * (255 * x))
    elif color_range == "limited":
        # range is [16, 235]
        return h273_Round((1 << (BitDepthY - 8)) * (219 * x + 16))
    raise AssertionError(f"invalid color range: {color_range}")


def h273_float_to_int_uv(x, color_range):
    BitDepthC = 8
    if color_range == "full":
        # range is [0, 255]
        return h273_Round((1 << (BitDepthC - 8)) * (255 * x))
    elif color_range == "limited":
        # range is [16, 235]
        return h273_Round((1 << (BitDepthC - 8)) * (224 * x))
    raise AssertionError(f"invalid color range: {color_range}")


# Implementation of Rec. ITU-T H.273 (07/2021), Section 8.3 ("Matrix coefficients")
def convert_rgb2yuv_h273(R, G, B, mc, color_range_yuv, color_range_rgb):
    BitDepthY = 8
    BitDepthC = 8

    # TODO(chemag): we should be using TransferCharacteristics function
    # to go from ER to E'R

    # 1. pre-process input values
    # normalize ER, EG, EB in (0, 1)
    ER = h273_int_to_float_yrgb(R, color_range_rgb)
    EG = h273_int_to_float_yrgb(G, color_range_rgb)
    EB = h273_int_to_float_yrgb(B, color_range_rgb)

    # 2. apply the transfer function
    if mc in (1, 4, 5, 6, 7, 9, 10, 12, 13):
        if mc in (1, 4, 5, 6, 7, 9, 10):
            Kr = H273_MATRIX_COEFFICIENTS[mc]["Kr"]
            Kb = H273_MATRIX_COEFFICIENTS[mc]["Kb"]
        elif mc in (12, 13):
            raise AssertionError(f"unsupported h273 mc: {mc}")
    if mc not in (0, 8, 10, 11, 13, 14):
        EY = Kr * ER + (1 - Kr - Kb) * EG + Kb * EB  # Equation (38)
        EPB = 0.5 * (EB - EY) / (1 - Kb)  # Equation (39)
        EPR = 0.5 * (ER - EY) / (1 - Kr)  # Equation (40)
    elif mc == 0:
        Y = h273_Round(G)  # Equation (41)
        Cb = h273_Round(B)  # Equation (42)
        Cr = h273_Round(R)  # Equation (43)
        Y = h273_Clip1Y(Y, BitDepthY)
        Cr = h273_Clip1C(Cr, BitDepthC)
        Cb = h273_Clip1C(Cb, BitDepthC)
        return (Y, Cb, Cr)
    elif mc == 8:
        if BitDepthY == BitDepthC:
            Y = h273_Round(0.5 * G + 0.25 * (R + B))  # Equation (44)
            Cb = h273_Round(0.5 * G - 0.25 * (R + B)) + (
                1 << (BitDepthC - 1)
            )  # Equation (45)
            Cr = h273_Round(0.5 * (R - B)) + (1 << (BitDepthC - 1))  # Equation (46)
        else:
            Cr = h273_Round(R) - h273_Round(B) + (1 << (BitDepthC - 1))  # Equation (51)
            t = h273_Round(B) + ((Cr - (1 << (BitDepthC - 1))) >> 1)  # Equation (52)
            Cb = h273_Round(G) - t + (1 << (BitDepthC - 1))  # Equation (53)
            Y = t + ((Cb - (1 << (BitDepthC - 1))) >> 1)  # Equation (54)
        Y = h273_Clip1Y(Y, BitDepthY)
        Cr = h273_Clip1C(Cr, BitDepthC)
        Cb = h273_Clip1C(Cb, BitDepthC)
        return (Y, Cb, Cr)
    elif mc in (10, 13):
        Kr = H273_MATRIX_COEFFICIENTS[mc]["Kr"]
        Kb = H273_MATRIX_COEFFICIENTS[mc]["Kb"]
        NB = 1 - Kb  # Equation (65)
        PB = 1 - (Kb)  # Equation (66)
        NR = 1 - Kr  # Equation (67)
        PR = 1 - (Kr)  # Equation (68)
        EY = Kr * ER + (1 - Kr - Kb) * EG + Kb * EB  # Equation (59)
        if -NB <= (EB - EY) and (EB - EY) <= 0:
            EPB = (EB - EY) / (2 * NB)  # Equation (61)
        if 0 < EB - EY and EB - EY <= PB:
            EPB = (EB - EY) / (2 * PB)  # Equation (62)
        if -NR <= (ER - EY) and (ER - EY) <= 0:
            EPR = (ER - EY) / (2 * NR)  # Equation (63)
        if 0 < (ER - EY) and (ER - EY) <= PR:
            EPR = (ER - EY) / (2 * PR)  # Equation (64)
    elif mc == 11:
        EY = EG  # Equation (69)
        EPB = (0.986566 * EB - EY) / 2.0  # Equation (70)
        EPR = (ER - 0.991902 * EY) / 2.0  # Equation (71)
    elif mc == 14:
        EL = (1688 * ER + 2146 * EG + 262 * EB) / 4096  # Equation (14)
        EM = (683 * ER + 2951 * EG + 462 * EB) / 4096  # Equation (15)
        ES = (99 * ER + 309 * EG + 3688 * EB) / 4096  # Equation (16)
        # If TransferCharacteristics is not equal to 18, equations 72 to 74 apply
        EY = 0.5 * (EL + EM)  # Equation (72)
        EPB = (6610 * EL - 13613 * EM + 7003 * ES) / 4096  # Equation (73)
        EPR = (17933 * EL - 17390 * EM - 543 * ES) / 4096  # Equation (74)
        # If TransferCharacteristics is not equal to 18, equations 75 to 77 apply:

    # 3. normalize Y, Pb, Pb to the color range
    Y = h273_Clip1Y(
        h273_float_to_int_yrgb(EY, color_range_yuv), BitDepthY
    )  # Equation (23)
    Pb = h273_Clip1C(
        h273_float_to_int_uv(EPB + 0.5, color_range_yuv), BitDepthC
    )  # Equation (24)
    Pr = h273_Clip1C(
        h273_float_to_int_uv(EPR + 0.5, color_range_yuv), BitDepthC
    )  # Equation (25)
    return (Y, Pb, Pr)


# Implementation of Rec. ITU-T H.273 (07/2021), Section 8.3 ("Matrix coefficients")
def convert_yuv2rgb_h273(Y, U, V, mc, color_range_yuv, color_range_rgb):
    BitDepthY = 8
    BitDepthC = 8

    # 1. normalize EY, EU, EV in (0, 1)
    EY = h273_int_to_float_yrgb(Y, color_range_yuv)
    EU = h273_int_to_float_uv(U, color_range_yuv)
    EV = h273_int_to_float_uv(V, color_range_yuv)

    # 2. apply the transfer function
    if mc in (1, 4, 5, 6, 7, 9, 10, 12, 13):
        if mc in (1, 4, 5, 6, 7, 9, 10):
            Kr = H273_MATRIX_COEFFICIENTS[mc]["Kr"]
            Kb = H273_MATRIX_COEFFICIENTS[mc]["Kb"]
        elif mc in (12, 13):
            raise AssertionError(f"unsupported h273 mc: {mc}")
    if mc not in (0, 8, 10, 11, 13, 14):
        Kg = 1.0 - Kr - Kb
        u_m = 0.5 / (1.0 - Kb)
        v_m = 0.5 / (1.0 - Kr)
        inv_matrix = np.array(
            [
                [Kr, Kg, Kb, 0.0],  # Y
                [u_m * -Kr, u_m * -Kg, u_m * (1.0 - Kb), 0.5],  # U
                [v_m * (1.0 - Kr), v_m * -Kg, v_m * -Kb, 0.5],  # V
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        matrix = np.linalg.inv(inv_matrix)
        (ER, EG, EB, _) = matrix @ (EY, EU, EV, 1)
    elif mc == 0:
        Y, Cb, Cr = Y, U, V
        G = h273_Round(Y)  # Equation (41)
        B = h273_Round(Cb)  # Equation (42)
        R = h273_Round(Cr)  # Equation (43)
        R = h273_Clip1Y(R, BitDepthY)
        G = h273_Clip1Y(G, BitDepthY)
        B = h273_Clip1Y(B, BitDepthY)
        return (R, G, B)
    elif mc == 8:
        Y, Cb, Cr = Y, U, V
        if BitDepthY == BitDepthC:
            t = Y - (Cb - (1 << (BitDepthC - 1)))  # Equation (47)
            G = h273_Clip1Y(
                Y + (Cb - (1 << (BitDepthC - 1))), BitDepthY
            )  # Equation (48)
            B = h273_Clip1Y(
                t - (Cr - (1 << (BitDepthC - 1))), BitDepthY
            )  # Equation (49)
            R = h273_Clip1Y(
                t + (Cr - (1 << (BitDepthC - 1))), BitDepthY
            )  # Equation (50)
        else:
            t = Y - ((Cb - (1 << (BitDepthC - 1))) >> 1)  # Equation (55)
            G = h273_Clip1Y(
                t + (Cb - (1 << (BitDepthC - 1))), BitDepthY
            )  # Equation (56)
            B = h273_Clip1Y(
                t - ((Cr - (1 << (BitDepthC - 1))) >> 1), BitDepthY
            )  # Equation (57)
            R = h273_Clip1Y(
                B + (Cr - (1 << (BitDepthC - 1))), BitDepthY
            )  # Equation (58)
        R = h273_Clip1Y(R, BitDepthY)
        G = h273_Clip1Y(G, BitDepthY)
        B = h273_Clip1Y(B, BitDepthY)
        return (R, G, B)
    elif mc in (10, 13):
        EY, EPB, EPR = EY, EU - 0.5, EV - 0.5
        if mc == 10:
            # TODO(chemag): assuming NB == PB and NR == PR
            # Kr = 0.2627
            # Kb = 0.0593
            # NB = 1 - Kb  # Equation (65)
            # NR = 1 - Kr  # Equation (67)
            # matrix_rgb2yuv = np.array([[Kr, (1 - Kr - Kb), Kb], [-Kr / (2 * NB), -(1 - Kr - Kb) / (2 * NB), (1 - Kb) / (2 * NB)], [(1 - Kr) / (2 * NR), -(1 - Kr - Kb) / (2 * NR), -Kb/(2 * NR)]])
            # np.linalg.inv(matrix_rgb2yuv)
            matrix = np.array(
                [
                    [1.0, -5.55111512e-17, 1.47460000e00],
                    [1.0, -1.64553127e-01, -5.71353127e-01],
                    [1.0, 1.88140000e00, 2.38961873e-17],
                ]
            )
            ER, EG, EB = matrix @ (EY, EPB, EPR)
        elif mc == 13:
            raise AssertionError(f"unsupported mc: {mc}")
    elif mc == 11:
        EY, EPB, EPR = EY, EU - 0.5, EV - 0.5
        # m = np.array([[0.0, 1.0, 0.0], [0.0, -1.0 / 2.0, 0.986566 / 2.0], [1.0 / 2.0, - 0.991902 / 2.0, 0.0]])
        # np.linalg.inv(m)
        matrix = np.array(
            [
                [0.991902, 0.0, 2.0],
                [1.0, 0.0, 0.0],
                [1.01361693, 2.02723386, 0.0],
            ]
        )
        ER, EG, EB = matrix @ (EY, EPB, EPR)
    elif mc == 14:
        EY, EPB, EPR = EY, EU - 0.5, EV - 0.5
        # m = np.array([[0.5, 0.5, 0.0], [6610/4096, -7465/4096, 3840/4096], [9500/4096, - 9212/4096, - 288/4096]])
        # np.linalg.inv(m)
        matrix = np.array(
            [
                [0.98867466, 0.01554056, 0.20720749],
                [1.01132534, -0.01554056, -0.20720749],
                [0.26416774, 1.00970484, -0.759491],
            ]
        )
        EL, EM, ES = matrix @ (EY, EPB, EPR)
        # m = np.array([[1688, 2146, 262], [683, 2951, 462], [99, 309, 3688]])
        # np.linalg.inv(m)
        matrix = np.array(
            [
                [8.39015306e-04, -6.11926787e-04, 1.70521055e-05],
                [-1.93195692e-04, 4.84277454e-04, -4.69411368e-05],
                [-6.33542473e-06, -2.41488561e-05, 2.74624906e-04],
            ]
        )
        (ER, EG, EB) = 4096 * matrix @ (EL, EM, ES)

    # 3. normalize R, G, B to the color range
    R = h273_Clip1Y(
        h273_float_to_int_yrgb(ER, color_range_rgb), BitDepthY
    )  # Equation (20)
    G = h273_Clip1Y(
        h273_float_to_int_yrgb(EG, color_range_rgb), BitDepthY
    )  # Equation (21)
    B = h273_Clip1Y(
        h273_float_to_int_yrgb(EB, color_range_rgb), BitDepthY
    )  # Equation (22)
    return (R, G, B)


# Implementation from ColorSpace::GetTransferMatrix in
# chromium/src/ui/gfx/color_space.cc
def convert_rgb2yuv_h273_chromium(R, G, B, mc, color_range_yuv, color_range_rgb):
    BitDepthY = 8
    BitDepthC = 8
    # 1. normalize ER, EG, EB in (0, 1)
    ER = h273_int_to_float_yrgb(R, color_range_rgb)
    EG = h273_int_to_float_yrgb(G, color_range_rgb)
    EB = h273_int_to_float_yrgb(B, color_range_rgb)
    # 2. calculate the transfer matrix
    matrix_rgb2yuv = h273_get_transfer_matrix_rgb2yuv(mc)
    # 3. use matrix product to get the output components
    EY, EU, EV, _ = matrix_rgb2yuv @ np.array([ER, EG, EB, 1])
    # 4. normalize Y, U, V to the color range
    Y = h273_Clip1Y(
        h273_float_to_int_yrgb(EY, color_range_yuv), BitDepthY
    )  # Equation (23)
    U = h273_Clip1C(
        h273_float_to_int_uv(EU, color_range_yuv), BitDepthC
    )  # Equation (24)
    V = h273_Clip1C(
        h273_float_to_int_uv(EV, color_range_yuv), BitDepthC
    )  # Equation (25)
    return (Y, U, V)


# Implementation from ColorSpace::GetTransferMatrix in
# chromium/src/ui/gfx/color_space.cc
def convert_yuv2rgb_h273_chromium(Y, U, V, mc, color_range_yuv, color_range_rgb):
    BitDepthY = 8
    # 1. normalize EY, EU, EV in (0, 1)
    EY = h273_int_to_float_yrgb(Y, color_range_yuv)
    EU = h273_int_to_float_uv(U, color_range_yuv)
    EV = h273_int_to_float_uv(V, color_range_yuv)
    # 2. calculate the transfer matrix
    matrix_yuv2rgb = h273_get_transfer_matrix_yuv2rgb(mc)
    # 3. use matrix product to get the output components
    ER, EG, EB, _ = matrix_yuv2rgb @ np.array([EY, EU, EV, 1])
    # 4. normalize R, G, B to the color range
    R = h273_Clip1Y(
        h273_float_to_int_yrgb(ER, color_range_rgb), BitDepthY
    )  # Equation (20)
    G = h273_Clip1Y(
        h273_float_to_int_yrgb(EG, color_range_rgb), BitDepthY
    )  # Equation (21)
    B = h273_Clip1Y(
        h273_float_to_int_yrgb(EB, color_range_rgb), BitDepthY
    )  # Equation (22)
    return (R, G, B)


# calculates a 4x4 transfer matrix for an input matrix coefficients.
# Matrix is (R, G, B) -> (Y, U, V).
# Implementation from ColorSpace::GetTransferMatrix in
# chromium/src/ui/gfx/color_space.cc
# https://chromium.googlesource.com/chromium/src/+/refs/heads/main/ui/gfx/color_space.cc#1009
def h273_get_transfer_matrix_rgb2yuv(mc):
    assert mc in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11), f"error: unsupported mc: {mc}"
    BitDepthY = 8
    BitDepthC = 8

    if mc == 0:  # GBR
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],  # G
                [0.0, 0.0, 1.0, 0.0],  # B
                [1.0, 0.0, 0.0, 0.0],  # R
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    elif mc == 1:  # BT709
        Kr = 0.2126
        Kb = 0.0722

    elif mc == 4:  # FCC
        Kr = 0.30
        Kb = 0.11

    elif mc == 5:  # BT470BG
        Kr = 0.299
        Kb = 0.114

    elif mc == 6:  # SMPTE170M
        Kr = 0.299
        Kb = 0.114

    elif mc == 7:  # SMPTE240M
        Kr = 0.212
        Kb = 0.087

    elif mc == 8:  # YCOCG
        chroma_0_5 = (float)(1 << (BitDepthY - 1)) / ((1 << BitDepthY) - 1)
        return np.array(
            [
                [0.25, 0.5, 0.25, 0.0],  # Y
                [-0.25, 0.5, -0.25, chroma_0_5],  # Cg
                [0.5, 0.0, -0.5, chroma_0_5],  # Co
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    elif mc == 9:  # BT2020_NCL
        Kr = 0.2627
        Kb = 0.0593

    elif mc == 10:  # BT2020_CL
        Kr = 0.2627
        Kb = 0.0593
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # R
                [Kr, 1.0 - Kr - Kb, Kb, 0.0],  # Y
                [0.0, 0.0, 1.0, 0.0],  # B
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    elif mc == 11:  # YDZDX
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],  # Y
                [0.0, -0.5, 0.986566 / 2.0, 0.5],  # DX or DZ
                [0.5, -0.991902 / 2.0, 0.0, 0.5],  # DZ or DX
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    # generic matrix code
    Kg = 1.0 - Kr - Kb
    u_m = 0.5 / (1.0 - Kb)
    v_m = 0.5 / (1.0 - Kr)
    return np.array(
        [
            [Kr, Kg, Kb, 0.0],  # Y
            [u_m * -Kr, u_m * -Kg, u_m * (1.0 - Kb), 0.5],  # U
            [v_m * (1.0 - Kr), v_m * -Kg, v_m * -Kb, 0.5],  # V
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# Matrix is (Y, U, V) -> (R, G, B).
def h273_get_transfer_matrix_yuv2rgb(mc):
    matrix = h273_get_transfer_matrix_rgb2yuv(mc)
    # return the inverse of the matrix
    return np.linalg.inv(matrix)


CONVERSION_DIRECTIONS = (
    "yuv2yuv",
    "yuv2rgb",
    "rgb2yuv",
    "rgb2rgb",
)


CONVERSION_FUNCTIONS = {
    "unit": {
        "yuv2yuv": lambda x, y, z: (x, y, z),
        "rgb2rgb": lambda x, y, z: (x, y, z),
    },
    "sdtv.basic": {
        "rgb2yuv": convert_rgb2yuv_sdtv_basic,
        "yuv2rgb": convert_yuv2rgb_sdtv_basic,
    },
    "yiq": {
        "rgb2yuv": convert_rgb2yuv_yiq,
        "yuv2rgb": convert_yuv2rgb_yiq,
    },
    "sdtv.analog": {
        "rgb2yuv": convert_rgb2yuv_ycbcr_sdtv_analog,
        "yuv2rgb": convert_yuv2rgb_ycbcr_sdtv_analog,
    },
    "sdtv.digital": {
        "rgb2yuv": convert_rgb2yuv_ycbcr_sdtv_digital,
        "yuv2rgb": convert_yuv2rgb_ycbcr_sdtv_digital,
    },
    "sdtv.computer": {
        "rgb2yuv": convert_rgb2yuv_ycbcr_sdtv_computer,
        "yuv2rgb": convert_yuv2rgb_ycbcr_sdtv_computer,
    },
    "hdtv.basic": {
        "rgb2yuv": convert_rgb2yuv_hdtv_basic,
        "yuv2rgb": convert_yuv2rgb_hdtv_basic,
    },
    "hdtv.analog": {
        "rgb2yuv": convert_rgb2yuv_ycbcr_hdtv_analog,
        "yuv2rgb": convert_yuv2rgb_ycbcr_hdtv_analog,
    },
    "hdtv.digital": {
        "rgb2yuv": convert_rgb2yuv_ycbcr_hdtv_digital,
        "yuv2rgb": convert_yuv2rgb_ycbcr_hdtv_digital,
    },
    "hdtv.computer": {
        "rgb2yuv": convert_rgb2yuv_ycbcr_hdtv_computer,
        "yuv2rgb": convert_yuv2rgb_ycbcr_hdtv_computer,
    },
    "ycocg": {
        "rgb2yuv": convert_rgb2yuv_ycocg,
        "yuv2rgb": convert_yuv2rgb_ycocg,
    },
    "ycocgr": {
        "rgb2yuv": convert_rgb2yuv_ycocgr,
        "yuv2rgb": convert_yuv2rgb_ycocgr,
    },
    "h273": {
        "rgb2yuv": convert_rgb2yuv_h273,
        "yuv2rgb": convert_yuv2rgb_h273,
    },
    "h273chromium": {
        "rgb2yuv": convert_rgb2yuv_h273_chromium,
        "yuv2rgb": convert_yuv2rgb_h273_chromium,
    },
}

# per-component range of the matrix output
default_values = {
    "func": "help",
    "pixel": None,
    "width": 1280,
    "height": 720,
    "ipix_fmt": "yuv420p",
    "opix_fmt": "rgba",
    "conversion_direction": None,
    "conversion_type": None,
    "matrix_coefficients": None,
    "color_range_yuv": "full",
    "color_range_rgb": "full",
    "yuv2yuv": "unit",
    "rgb2rgb": "unit",
    "rgb2yuv": "sdtv.computer",
    "yuv2rgb": "sdtv.computer",
}


def convert_pixel_wrapper(options):
    # prepare the input pixel
    idata = options.pixel
    width = 1
    height = 1
    if options.conversion_direction == "yuv2rgb":
        ipix_fmt = "yuv444p"
        opix_fmt = "rgb24"
    elif options.conversion_direction == "rgb2yuv":
        # add the alpha channel
        idata.append(255)
        ipix_fmt = "rgb24"
        opix_fmt = "yuv444p"
    # convert the input pixel
    odata = convert_image(
        idata,
        width,
        height,
        ipix_fmt,
        options.conversion_direction,
        options.conversion_type,
        options.matrix_coefficients,
        options.color_range_yuv,
        options.color_range_rgb,
        opix_fmt,
    )
    # print the output pixel
    print(",".join(str(i) for i in list(odata)))


def convert_image(
    idata,
    w,
    h,
    ipix_fmt,
    conversion_direction,
    conversion_type,
    matrix_coefficients,
    color_range_yuv,
    color_range_rgb,
    opix_fmt,
):
    # allocate output array
    odata = array("B")
    oframe_size = int(w * h * itools_yuvcommon.get_length_factor(opix_fmt))
    odata.extend([255] * oframe_size)

    # calculate the conversion direction
    if conversion_direction is None:
        if itools_yuvcommon.is_yuv(ipix_fmt) and itools_yuvcommon.is_yuv(opix_fmt):
            conversion_direction = "yuv2yuv"
        elif itools_yuvcommon.is_yuv(ipix_fmt) and not itools_yuvcommon.is_yuv(
            opix_fmt
        ):
            conversion_direction = "yuv2rgb"
        elif not itools_yuvcommon.is_yuv(ipix_fmt) and itools_yuvcommon.is_yuv(
            opix_fmt
        ):
            conversion_direction = "rgb2yuv"
        elif not itools_yuvcommon.is_yuv(ipix_fmt) and not itools_yuvcommon.is_yuv(
            opix_fmt
        ):
            conversion_direction = "rgb2rgb"

    if conversion_type is None:
        conversion_type = default_values[conversion_direction]
    conversion_function = CONVERSION_FUNCTIONS[conversion_type][conversion_direction]

    # convert arrays
    for j in range(0, h):
        for i in range(0, w):
            # get input components
            a, b, c = itools_yuvcommon.get_component_locations(i, j, w, h, ipix_fmt)
            # get output components
            d, e, f = itools_yuvcommon.get_component_locations(i, j, w, h, opix_fmt)
            # color conversion
            if conversion_type in ("h273", "h273chromium"):
                x, y, z = conversion_function(
                    idata[a],
                    idata[b],
                    idata[c],
                    matrix_coefficients,
                    color_range_yuv,
                    color_range_rgb,
                )
            else:
                x, y, z = conversion_function(idata[a], idata[b], idata[c])
            try:
                odata[d], odata[e], odata[f] = int(x), int(y), int(z)
            except OverflowError:
                print(
                    "error: overflow %s(%i, %i, %i)"
                    % (conversion_function, idata[a], idata[b], idata[c])
                )
                sys.exit(-1)

    return odata


def convert_image_wrapper(options):
    # open input file
    if options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # read input array
    idata = itools_yuvcommon.read_image(
        options.infile,
        options.width,
        options.height,
        options.ipix_fmt,
        options.frame_number,
    )
    # generate gradient file
    odata = convert_image(
        idata,
        options.width,
        options.height,
        options.ipix_fmt,
        options.conversion_direction,
        options.conversion_type,
        options.matrix_coefficients,
        options.color_range_yuv,
        options.color_range_rgb,
        options.opix_fmt,
    )
    # write the output file
    with open(options.outfile, "wb") as fout:
        odata.tofile(fout)


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
    parser = argparse.ArgumentParser(description="Generic runner argparser.")
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=0,
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
        "--function",
        action="store",
        type=str,
        dest="func",
        default=default_values["func"],
        choices=FUNC_CHOICES.keys(),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FUNC_CHOICES.items())),
    )

    class PixelAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.pixel = [int(v) for v in values[0].split(",")]

    parser.add_argument(
        "--pixel",
        dest="pixel",
        action=PixelAction,
        nargs=1,
        help="use <c1>,<c2>,<c3>",
    )
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
        help=("use HEIGHT height (default: %i)" % default_values["height"]),
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
        "--ipix_fmt",
        action="store",
        type=str,
        dest="ipix_fmt",
        default=default_values["ipix_fmt"],
        choices=PIX_FMTS,
        metavar="INPUT_PIX_FMT",
        help=(
            "input pixel format %r (default: %s)"
            % (PIX_FMTS, default_values["ipix_fmt"])
        ),
    )
    parser.add_argument(
        "--opix_fmt",
        action="store",
        type=str,
        dest="opix_fmt",
        default=default_values["opix_fmt"],
        choices=PIX_FMTS,
        metavar="OUTPUT_PIX_FMT",
        help=(
            "output pixel format %r (default: %s)"
            % (PIX_FMTS, default_values["opix_fmt"])
        ),
    )
    parser.add_argument(
        "--conversion",
        action="store",
        type=str,
        dest="conversion_type",
        default=default_values["conversion_type"],
        choices=list(CONVERSION_FUNCTIONS.keys()),
        metavar="[%s]"
        % (
            " | ".join(
                list(CONVERSION_FUNCTIONS.keys()),
            )
        ),
        help="conversion type",
    )
    parser.add_argument(
        "--matrix-coefficients",
        action="store",
        type=int,
        dest="matrix_coefficients",
        default=default_values["matrix_coefficients"],
        choices=list(H273_MATRIX_COEFFICIENTS_LIST),
        metavar="[%s]" % (" | ".join(str(i) for i in H273_MATRIX_COEFFICIENTS_LIST),),
        help="conversion type",
    )
    parser.add_argument(
        "--color-range",
        action="store",
        type=str,
        dest="color_range_yuv",
        default=default_values["color_range_yuv"],
        choices=list(COLOR_RANGES),
        metavar="[%s]"
        % (
            " | ".join(
                list(COLOR_RANGES),
            )
        ),
        help="range",
    )
    parser.add_argument(
        "--color-range-rgb",
        action="store",
        type=str,
        dest="color_range_rgb",
        default=default_values["color_range_rgb"],
        choices=list(COLOR_RANGES),
        metavar="[%s]"
        % (
            " | ".join(
                list(COLOR_RANGES),
            )
        ),
        help="range",
    )
    parser.add_argument(
        "--direction",
        action="store",
        type=str,
        dest="conversion_direction",
        default=default_values["conversion_direction"],
        choices=CONVERSION_DIRECTIONS,
        metavar="[%s]" % (" | ".join(CONVERSION_DIRECTIONS)),
        help="conversion direction",
    )
    parser.add_argument(
        "-n", "--frame_number", required=False, help="frame number", type=int, default=0
    )
    parser.add_argument(
        "-i",
        "--infile",
        dest="infile",
        type=str,
        default=None,
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        default=None,
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    # implement help
    if options.func == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # print results
    if options.debug > 0:
        print(options)
    # get in/out files
    if options.infile in (None, "-"):
        options.infile = sys.stdin
    if options.outfile in (None, "-"):
        options.outfile = sys.stdout
    if options.function == "image":
        convert_image_wrapper(options)
    elif options.function == "pixel":
        convert_pixel_wrapper(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
