#!/usr/bin/env python3

"""itools-bayer-enctool.py module description.

This is a tool to test Bayer image encoding.

"""


import argparse
import cv2
import importlib
import itertools
import json
import math
import numpy as np
import os
import pandas as pd
import re
import shutil
import sys
import tempfile

itools_version = importlib.import_module("itools-version")


DEFAULT_QUALITIES = [25, 75, 85, 95, 96, 97, 98, 99]
DEFAULT_QUALITY_LIST = sorted(set(list(range(0, 101, 10)) + DEFAULT_QUALITIES))

CODEC_LIST = ("jpeg/cv2",)

# dtype operation
# We use 2x dtype values
# * (1) st_dtype (storage dtype): This is uint8 for 8-bit Bayer, uint16 for
#   higher bit depths.
# * (2) op_dtype (operation dtype): This is int32 in all cases.
ST_DTYPE_8BIT = np.uint8
ST_DTYPE_16BIT = np.uint16
OP_DTYPE = np.int32


default_values = {
    "debug": 0,
    "dry_run": False,
    "cleanup": 1,
    "codec": "jpeg/cv2",
    "quality_list": ",".join(str(v) for v in DEFAULT_QUALITY_LIST),
    "workdir": tempfile.gettempdir(),
    "width": -1,
    "height": -1,
    "depth": 8,
    "infile_list": None,
    "outfile": None,
}


COLUMN_LIST = [
    "infile",
    "width",
    "height",
    "depth",
    "approach",
    "codec",
    "quality",
    "encoded_size",
    "encoded_size_breakdown",
    "encoded_bpp",
    "psnr_bayer",
    "psnr_rgb",
    "psnr_rgb_r",
    "psnr_rgb_g",
    "psnr_rgb_b",
    "psnr_yuv",
    "psnr_yuv_y",
    "psnr_yuv_u",
    "psnr_yuv_v",
]


def calculate_psnr(plane1, plane2):
    # Calculate the mean squared error (MSE)
    mse = np.mean((plane1 - plane2) ** 2)
    # Calculate the maximum possible value (peak)
    max_value = 255.0  # Assuming 8-bit unsigned integers
    # Calculate the PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((max_value**2) / mse)
    return float(psnr)


def get_opt_depth(depth):
    op_depth = 8 * (1 + (depth - 1) // 8)
    return op_depth


def remosaic_rgb_image(rgb_image, depth):
    op_depth = get_opt_depth(depth)
    st_dtype = np.uint8 if op_depth == 8 else np.uint16
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    height, width, _ = bgr_image.shape
    bayer_image = np.zeros((height, width), dtype=st_dtype)
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                bayer_image[i, j] = bgr_image[i, j, 0]  # Blue
            else:
                if i % 2 == 0:
                    bayer_image[i, j] = bgr_image[i, j, 1]  # Green
                else:
                    bayer_image[i, j] = bgr_image[i, j, 2]  # Red
    return bayer_image


def upsample_matrix(arr, shape):
    upsampled_array = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    rows, cols = shape
    height, width = shape
    return upsampled_array[:height, :width]


# matrix clippers
def clip_positive(arr, depth, check=True):
    op_depth = get_opt_depth(depth)
    max_value = 2**op_depth - 1
    st_dtype = np.uint8 if op_depth == 8 else np.uint16
    # check for values outside the valid range
    if check and np.any((arr < 0) | (arr > max_value)):
        raise ValueError("Array contains values outside the valid range")
    # round values to the closest integer and convert to storage dtype
    return np.clip(np.round(arr), 0, max_value).astype(st_dtype)


def clip_integer_and_scale(arr, depth, check=True):
    op_depth = get_opt_depth(depth)
    max_value = 2**op_depth - 1
    min_value = -max_value
    shift = 2 ** (op_depth - 1)
    st_dtype = np.uint8 if op_depth == 8 else np.uint16
    # check for values outside the valid range
    if check and np.any((arr < min_value) | (arr > max_value)):
        raise ValueError("Array contains values outside the valid range")
    # scale and round values to the storage dtype
    return np.round((arr >> 1) + shift).astype(st_dtype)


def unclip_positive(arr, depth):
    return arr.astype(OP_DTYPE)


def unclip_integer_and_unscale(arr, depth):
    op_depth = get_opt_depth(depth)
    shift = 2 ** (op_depth - 1)
    # unscale and convert
    return (arr.astype(OP_DTYPE) - shift) << 1


# Malvar Sullivan, "Progressive to Lossless Compression of Color Filter
# Array Images Using Macropixel Spectral Spatial Transformation", 2012
def convert_rg1g2b_to_ydgcocg(bayer_image, depth):
    # 1. separate RGGB components
    bayer_r = bayer_image[::2, ::2]
    bayer_g1 = bayer_image[::2, 1::2]
    bayer_g2 = bayer_image[1::2, ::2]
    bayer_b = bayer_image[1::2, 1::2]
    # 2. do the color conversion
    bayer_y, bayer_dg, bayer_co, bayer_cg = convert_rg1g2b_to_ydgcocg_components(
        bayer_r, bayer_g1, bayer_g2, bayer_b, depth
    )
    return bayer_y, bayer_dg, bayer_co, bayer_cg


def convert_rg1g2b_to_ydgcocg_components(bayer_r, bayer_g1, bayer_g2, bayer_b, depth):
    # 1. convert values
    bayer_r = bayer_r.astype(OP_DTYPE)
    bayer_g1 = bayer_g1.astype(OP_DTYPE)
    bayer_g2 = bayer_g2.astype(OP_DTYPE)
    bayer_b = bayer_b.astype(OP_DTYPE)
    # [ Y  ]   [ 1/4  1/4  1/4  1/4 ] [ G1 ]
    # [ Dg ] = [ -1    1    0    0  ] [ G4 ]
    # [ Co ]   [  0    0    1   -1  ] [ R2 ]
    # [ Cg ]   [ 1/2  1/2 -1/2 -1/2 ] [ B3 ]
    bayer_y = (bayer_g1 >> 2) + (bayer_g2 >> 2) + (bayer_r >> 2) + (bayer_b >> 2)
    bayer_dg = (-1) * bayer_g1 + (1) * bayer_g2
    bayer_co = (1) * bayer_r + (-1) * bayer_b
    bayer_cg = (bayer_g1 >> 1) + (bayer_g2 >> 1) - (bayer_r >> 1) - (bayer_b >> 1)
    # 2. clip matrices to storage type
    bayer_y = clip_positive(bayer_y, depth)
    bayer_dg = clip_integer_and_scale(bayer_dg, depth)
    bayer_co = clip_integer_and_scale(bayer_co, depth)
    bayer_cg = clip_integer_and_scale(bayer_cg, depth)
    return bayer_y, bayer_dg, bayer_co, bayer_cg


def convert_ydgcocg_to_rg1g2b(bayer_y, bayer_dg, bayer_co, bayer_cg, depth):
    # 1. do the color conversion
    bayer_r, bayer_g1, bayer_g2, bayer_b = convert_ydgcocg_to_rg1g2b_components(
        bayer_y, bayer_dg, bayer_co, bayer_cg, depth
    )
    # 2. merge RGGB components
    bayer_image = np.zeros(list(2 * dim for dim in bayer_r.shape), dtype=bayer_r.dtype)
    bayer_image[::2, ::2] = bayer_r
    bayer_image[::2, 1::2] = bayer_g1
    bayer_image[1::2, ::2] = bayer_g2
    bayer_image[1::2, 1::2] = bayer_b
    return bayer_image


def convert_ydgcocg_to_rg1g2b_components(bayer_y, bayer_dg, bayer_co, bayer_cg, depth):
    # 1. unclip matrices from storage dtype
    bayer_y = unclip_positive(bayer_y, depth)
    bayer_dg = unclip_integer_and_unscale(bayer_dg, depth)
    bayer_co = unclip_integer_and_unscale(bayer_co, depth)
    bayer_cg = unclip_integer_and_unscale(bayer_cg, depth)
    # 2. convert values
    # l = (1/4, 1/4, 1/4, 1/4, -1, 1, 0, 0, 0, 0, 1, -1, 1/2, 1/2, -1/2, -1/2)
    # matrix = np.array(l).reshape(4, 4)
    # np.linalg.inv(matrix)
    # array([[ 1. , -0.5, -0. ,  0.5],
    #        [ 1. ,  0.5,  0. ,  0.5],
    #        [ 1. ,  0. ,  0.5, -0.5],
    #        [ 1. ,  0. , -0.5, -0.5]])
    # [ G1 ]   [  1  -1/2   0   1/2 ] [ Y  ]
    # [ G4 ] = [  1   1/2   0   1/2 ] [ Dg ]
    # [ R2 ]   [  1    0   1/2 -1/2 ] [ Co ]
    # [ B3 ]   [  1    0  -1/2 -1/2 ] [ Cg ]
    bayer_g1 = bayer_y - (bayer_dg >> 1) + (bayer_cg >> 1)
    bayer_g2 = bayer_y + (bayer_dg >> 1) + (bayer_cg >> 1)
    bayer_r = bayer_y + (bayer_co >> 1) - (bayer_cg >> 1)
    bayer_b = bayer_y - (bayer_co >> 1) - (bayer_cg >> 1)
    # 3. round to storage dtype
    bayer_r = clip_positive(bayer_r, depth, check=False)
    bayer_g1 = clip_positive(bayer_g1, depth, check=False)
    bayer_g2 = clip_positive(bayer_g2, depth, check=False)
    bayer_b = clip_positive(bayer_b, depth, check=False)
    return bayer_r, bayer_g1, bayer_g2, bayer_b


# reads a bayer image as packed
def read_bayer_image_packed_mode(infile, width, height, depth):
    if depth == 8:
        bayer_image = np.fromfile(infile, dtype=np.uint8)
    elif depth == 16:
        bayer_image = np.fromfile(infile, dtype=np.uint16)  # little-endian
        # bayer_image = np.fromfile(infile, dtype=">u2")  # big-endian
    elif depth in (10, 12, 14):
        # cv2 assumes color to be 16-bit depth if dtype is uint16
        # For 10/12/14-bit color, let's expand to 16-bit before
        # further processing. This also helps unify all further
        # processing as 16-bit.
        # support expanded and/or packed bayer formats
        # if expanded:
        # a. read as little-endian
        bayer_image = np.fromfile(infile, dtype=np.uint16)
        # b. expand to 16 bits
        bayer_image <<= 16 - depth
        # elif packed:
    else:
        raise ValueError(f"Unsupported depth value: {depth}")
    # reshape image
    bayer_image = bayer_image.reshape(width, height)
    return bayer_image


# encoding processing
def codec_process(codec, quality, depth, arrays_list):
    if codec == "jpeg/cv2" and depth == 8:
        return jpeg_cv2_process(quality, arrays_list)


def jpeg_cv2_process(quality, arrays_list):
    arrays_prime_list = []
    encoded_size_list = []
    for array in arrays_list:
        # encode file
        success, array_encoded = cv2.imencode(
            ".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        encoded_size = len(array_encoded)
        encoded_size_list.append(encoded_size)
        # decode file
        array_prime = cv2.imdecode(array_encoded, cv2.IMREAD_GRAYSCALE)
        arrays_prime_list.append(array_prime)
    return arrays_prime_list, encoded_size_list


# Bayer processing stack
def process_file_bayer_ydgcocg(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_bayer_ydgcocg_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_bayer_ydgcocg_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    # 3. demosaic Bayer image to YDgCoCg
    bayer_y, bayer_dg, bayer_co, bayer_cg = convert_rg1g2b_to_ydgcocg(
        bayer_image, depth
    )

    for quality in quality_list:
        # 4. encode and decode the 4 planes
        (
            bayer_y_prime,
            bayer_dg_prime,
            bayer_co_prime,
            bayer_cg_prime,
        ), encoded_size_list = codec_process(
            codec, quality, depth, (bayer_y, bayer_dg, bayer_co, bayer_cg)
        )

        # 5. convert YDgCoCg image back to Bayer
        bayer_image_prime = convert_ydgcocg_to_rg1g2b(
            bayer_y_prime, bayer_dg_prime, bayer_co_prime, bayer_cg_prime, depth
        )

        # 6. demosaic raw image to RGB
        rgb_image_prime = cv2.cvtColor(bayer_image_prime, cv2.COLOR_BAYER_BG2RGB)
        rgb_r_prime, rgb_g_prime, rgb_b_prime = cv2.split(rgb_image_prime)

        # 7. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 8. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # psnr_bayer_y = calculate_psnr(bayer_y, bayer_y_prime)
        # psnr_bayer_dg = calculate_psnr(bayer_dg, bayer_dg_prime)
        # psnr_bayer_co = calculate_psnr(bayer_co, bayer_co_prime)
        # psnr_bayer_cg = calculate_psnr(bayer_cg, bayer_cg_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "bayer-ydgcocg",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


# Bayer processing stack
def process_file_bayer_ydgcocg_420(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_bayer_ydgcocg_420_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_bayer_ydgcocg_420_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    # 3. demosaic Bayer image to YDgCoCg
    bayer_y, bayer_dg, bayer_co, bayer_cg = convert_rg1g2b_to_ydgcocg(
        bayer_image, depth
    )

    for quality in quality_list:
        # 4. encode and decode the 4 planes
        # subsample the chromas
        bayer_co_subsampled = bayer_co[::2, ::2]
        bayer_cg_subsampled = bayer_cg[::2, ::2]
        # encode and decode the 4 planes
        (
            bayer_y_prime,
            bayer_dg_prime,
            bayer_co_subsampled_prime,
            bayer_cg_subsampled_prime,
        ), encoded_size_list = codec_process(
            codec,
            quality,
            depth,
            (bayer_y, bayer_dg, bayer_co_subsampled, bayer_cg_subsampled),
        )
        # upsample the chromas
        bayer_co_prime = upsample_matrix(bayer_co_subsampled_prime, bayer_co.shape)
        bayer_cg_prime = upsample_matrix(bayer_cg_subsampled_prime, bayer_cg.shape)

        # 5. convert YDgCoCg image back to Bayer
        bayer_image_prime = convert_ydgcocg_to_rg1g2b(
            bayer_y_prime, bayer_dg_prime, bayer_co_prime, bayer_cg_prime, depth
        )

        # 6. demosaic raw image to RGB
        rgb_image_prime = cv2.cvtColor(bayer_image_prime, cv2.COLOR_BAYER_BG2RGB)
        rgb_r_prime, rgb_g_prime, rgb_b_prime = cv2.split(rgb_image_prime)

        # 7. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 8. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # psnr_bayer_y = calculate_psnr(bayer_y, bayer_y_prime)
        # psnr_bayer_dg = calculate_psnr(bayer_dg, bayer_dg_prime)
        # psnr_bayer_co = calculate_psnr(bayer_co, bayer_co_prime)
        # psnr_bayer_cg = calculate_psnr(bayer_cg, bayer_cg_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "bayer-ydgcocg-420",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


# Bayer processing stack (single encoding)
def process_file_bayer_single(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_bayer_single_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_bayer_single_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    for quality in quality_list:
        # 3. encode and decode the single planes
        (bayer_image_prime,), encoded_size_list = codec_process(
            codec, quality, depth, (bayer_image,)
        )

        # 5. demosaic raw image to RGB
        rgb_image_prime = cv2.cvtColor(bayer_image_prime, cv2.COLOR_BAYER_BG2RGB)
        rgb_r_prime, rgb_g_prime, rgb_b_prime = cv2.split(rgb_image_prime)

        # 6. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 7. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "bayer-single",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


# Bayer processing stack (single encoding)
def process_file_bayer_rggb(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_bayer_rggb_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_bayer_rggb_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    # 3. separate RGGB components
    bayer_r = bayer_image[::2, ::2]
    bayer_g1 = bayer_image[::2, 1::2]
    bayer_g2 = bayer_image[1::2, ::2]
    bayer_b = bayer_image[1::2, 1::2]

    for quality in quality_list:
        # 4. encode and decode the 4 planes
        (
            bayer_r_prime,
            bayer_g1_prime,
            bayer_g2_prime,
            bayer_b_prime,
        ), encoded_size_list = codec_process(
            codec, quality, depth, (bayer_r, bayer_g1, bayer_g2, bayer_b)
        )

        # merge RGGB components
        bayer_image_prime = np.zeros(
            list(2 * dim for dim in bayer_r.shape), dtype=bayer_r.dtype
        )
        bayer_image_prime[::2, ::2] = bayer_r_prime
        bayer_image_prime[::2, 1::2] = bayer_g1_prime
        bayer_image_prime[1::2, ::2] = bayer_g2_prime
        bayer_image_prime[1::2, 1::2] = bayer_b_prime

        # 5. demosaic raw image to RGB
        rgb_image_prime = cv2.cvtColor(bayer_image_prime, cv2.COLOR_BAYER_BG2RGB)
        rgb_r_prime, rgb_g_prime, rgb_b_prime = cv2.split(rgb_image_prime)

        # 6. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 7. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "bayer-rggb",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


# traditional camera stack
def process_file_yuv444(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_yuv444_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_yuv444_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    for quality in quality_list:
        # 3. encode and decode the 3 planes
        (
            yuv_y_prime,
            yuv_u_prime,
            yuv_v_prime,
        ), encoded_size_list = codec_process(
            codec, quality, depth, (yuv_y, yuv_u, yuv_v)
        )
        yuv_image_prime = cv2.merge([yuv_y_prime, yuv_u_prime, yuv_v_prime])

        # 4. convert YUV image back to RGB
        rgb_image_prime = cv2.cvtColor(yuv_image_prime, cv2.COLOR_YUV2RGB)
        rgb_b_prime, rgb_g_prime, rgb_r_prime = cv2.split(rgb_image_prime)

        # 5. remosaic RGB image back to raw
        bayer_image_prime = remosaic_rgb_image(rgb_image_prime, depth)

        # 6. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "yuv444",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


def process_file_yuv420(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_yuv420_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_yuv420_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    for quality in quality_list:
        # 3. encode the 3 planes
        # subsample the chromas
        yuv_u_subsampled = yuv_u[::2, ::2]
        yuv_v_subsampled = yuv_v[::2, ::2]
        # encode and decode the 3 planes
        (
            yuv_y_prime,
            yuv_u_subsampled_prime,
            yuv_v_subsampled_prime,
        ), encoded_size_list = codec_process(
            codec, quality, depth, (yuv_y, yuv_u_subsampled, yuv_v_subsampled)
        )
        # upsample the chromas
        yuv_u_prime = upsample_matrix(yuv_u_subsampled_prime, yuv_u.shape)
        yuv_v_prime = upsample_matrix(yuv_u_subsampled_prime, yuv_v.shape)
        yuv_image_prime = cv2.merge([yuv_y_prime, yuv_u_prime, yuv_v_prime])

        # 5. convert YUV image back to RGB
        rgb_image_prime = cv2.cvtColor(yuv_image_prime, cv2.COLOR_YUV2RGB)
        rgb_b_prime, rgb_g_prime, rgb_r_prime = cv2.split(rgb_image_prime)

        # 6. remosaic RGB image back to raw
        bayer_image_prime = remosaic_rgb_image(rgb_image_prime, depth)

        # 7. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "yuv420",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


# RGB camera stack
def process_file_rgb(
    infile,
    width,
    height,
    depth,
    codec,
    quality_list,
    debug,
):
    bayer_image = read_bayer_image_packed_mode(infile, width, height, depth)
    return process_file_rgb_array(
        bayer_image,
        depth,
        infile,
        codec,
        quality_list,
        debug,
    )


def process_file_rgb_array(
    bayer_image,
    depth,
    infile,
    codec,
    quality_list,
    debug,
):
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.shape

    # 1. demosaic raw image to RGB
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
    rgb_r, rgb_g, rgb_b = cv2.split(rgb_image)

    # 2. convert RGB to YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    yuv_y, yuv_u, yuv_v = cv2.split(yuv_image)

    for quality in quality_list:
        # 3. encode and decode the 3 planes
        (
            rgb_r_prime,
            rgb_g_prime,
            rgb_b_prime,
        ), encoded_size_list = codec_process(
            codec, quality, depth, (rgb_r, rgb_g, rgb_b)
        )
        rgb_image_prime = cv2.merge([rgb_r_prime, rgb_g_prime, rgb_b_prime])

        # 4. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 5. remosaic RGB image back to raw
        bayer_image_prime = remosaic_rgb_image(rgb_image_prime, depth)

        # 6. calculate results
        # sizes
        encoded_size = sum(encoded_size_list)
        encoder_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values: YUV
        psnr_yuv_y = calculate_psnr(yuv_y, yuv_y_prime)
        psnr_yuv_u = calculate_psnr(yuv_u, yuv_u_prime)
        psnr_yuv_v = calculate_psnr(yuv_v, yuv_v_prime)
        psnr_yuv = np.mean([psnr_yuv_y, psnr_yuv_u, psnr_yuv_v])
        # psnr values: RGB
        psnr_rgb_r = calculate_psnr(rgb_r, rgb_r_prime)
        psnr_rgb_g = calculate_psnr(rgb_g, rgb_g_prime)
        psnr_rgb_b = calculate_psnr(rgb_b, rgb_b_prime)
        psnr_rgb = np.mean([psnr_rgb_r, psnr_rgb_g, psnr_rgb_b])
        # psnr values: Bayer
        psnr_bayer = calculate_psnr(bayer_image, bayer_image_prime)
        # add new element
        df.loc[df.size] = (
            infile,
            width,
            height,
            depth,
            "rgb",
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_list),
            encoder_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_r,
            psnr_rgb_g,
            psnr_rgb_b,
            psnr_yuv,
            psnr_yuv_y,
            psnr_yuv_u,
            psnr_yuv_v,
        )

    return df


def get_average_results(df):
    # import the results
    new_df = pd.DataFrame(columns=list(df.columns.values))
    for approach, quality in itertools.product(
        list(df["approach"].unique()),
        sorted(list(df["quality"].unique())),
    ):
        # select interesting data
        tmp_df = df[(df["approach"] == approach) & (df["quality"] == quality)]
        if tmp_df.size == 0:
            # no entries with this (approach, quality) combo
            continue
        # start with empty data
        derived_dict = {key: np.nan for key in list(df.columns.values)}
        derived_dict["infile"] = "average"
        derived_dict["approach"] = approach
        derived_dict["quality"] = quality
        # unused columns
        derived_dict["encoded_size_breakdown"] = ""
        # average a few columns
        COLUMNS_MEAN = (
            "width",
            "height",
            "depth",
            "encoded_size",
            "encoded_bpp",
            "psnr_bayer",
            "psnr_rgb",
            "psnr_rgb_r",
            "psnr_rgb_g",
            "psnr_rgb_b",
            "psnr_yuv",
            "psnr_yuv_y",
            "psnr_yuv_u",
            "psnr_yuv_v",
        )
        for key in COLUMNS_MEAN:
            derived_dict[key] = tmp_df[key].mean()
        new_df.loc[new_df.size] = list(derived_dict.values())
    return new_df


def process_data(
    infile_list,
    codec,
    quality_list,
    width,
    height,
    depth,
    workdir,
    outfile,
    cleanup,
    debug,
):
    df = None
    # 1. get infile/quality parameters
    quality_list = list(int(v) for v in quality_list.split(","))

    # 2. run the camera pipelines
    for infile in infile_list:
        # 2.1. run the Bayer-ydgcocg encoding pipeline
        tmp_df = process_file_bayer_ydgcocg(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.2. run the YUV444 encoding pipeline
        tmp_df = process_file_yuv444(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.3. run the RGB-encoding pipeline
        tmp_df = process_file_rgb(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.4. run the traditional YUV-encoding pipeline (4:2:0)
        tmp_df = process_file_yuv420(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.5. run the Bayer-single-encoding pipeline
        tmp_df = process_file_bayer_single(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.6. run the Bayer-subsampled-encoding pipeline
        tmp_df = process_file_bayer_ydgcocg_420(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.7. run the Bayer-rggb-encoding pipeline
        tmp_df = process_file_bayer_rggb(
            infile, width, height, depth, codec, quality_list, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)

    # 3. reindex per-file dataframe
    df = df.reindex()

    # 4. get average results
    derived_df = get_average_results(df)
    df = pd.concat([df, derived_df], ignore_index=True, axis=0)

    # 5. write the results
    df.to_csv(outfile, index=False)


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
        action="version",
        version=itools_version.__version__,
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
        "--cleanup",
        action="store_const",
        dest="cleanup",
        const=1,
        default=default_values["cleanup"],
        help="Cleanup Raw Files%s"
        % (" [default]" if default_values["cleanup"] == 1 else ""),
    )
    parser.add_argument(
        "--full-cleanup",
        action="store_const",
        dest="cleanup",
        const=2,
        default=default_values["cleanup"],
        help="Cleanup All Files%s"
        % (" [default]" if default_values["cleanup"] == 2 else ""),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_const",
        dest="cleanup",
        const=0,
        help="Do Not Cleanup Files%s"
        % (" [default]" if not default_values["cleanup"] == 0 else ""),
    )
    parser.add_argument(
        "--codec",
        action="store",
        type=str,
        dest="codec",
        default=default_values["codec"],
        choices=CODEC_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                CODEC_LIST,
            )
        ),
        help="codec",
    )
    parser.add_argument(
        "--quality-list",
        action="store",
        type=str,
        dest="quality_list",
        default=default_values["quality_list"],
        help="Quality list (comma-separated list)",
    )
    parser.add_argument(
        "--workdir",
        action="store",
        dest="workdir",
        type=str,
        default=default_values["workdir"],
        metavar="Work directory",
        help="work directory",
    )

    # 2-parameter setter using argparse.Action
    parser.add_argument(
        "--width",
        action="store",
        type=int,
        dest="width",
        default=default_values["width"],
        metavar="WIDTH",
        help=(f"use WIDTH width (default: {default_values['width']})"),
    )
    parser.add_argument(
        "--height",
        action="store",
        type=int,
        dest="height",
        default=default_values["height"],
        metavar="HEIGHT",
        help=(f"use HEIGHT height (default: {default_values['height']})"),
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
        "--depth",
        action="store",
        type=int,
        dest="depth",
        default=default_values["depth"],
        metavar="DEPTH",
        help=(f"use DEPTH depth (default: {default_values['depth']})"),
    )

    parser.add_argument(
        dest="infile_list",
        type=str,
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file-list",
        help="input file list",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # set workdir
    if options.workdir is not None:
        os.makedirs(options.workdir, exist_ok=True)
        tempfile.tempdir = options.workdir
    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")
    # process infile
    process_data(
        options.infile_list,
        options.codec,
        options.quality_list,
        options.width,
        options.height,
        options.depth,
        options.workdir,
        options.outfile,
        options.cleanup,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
