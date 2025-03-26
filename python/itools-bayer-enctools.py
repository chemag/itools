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


default_values = {
    "debug": 0,
    "dry_run": False,
    "cleanup": 1,
    "quality_list": ",".join(str(v) for v in DEFAULT_QUALITY_LIST),
    "workdir": tempfile.gettempdir(),
    "width": -1,
    "height": -1,
    "infile_list": None,
    "outfile": None,
}


COLUMN_LIST = [
    "infile",
    "width",
    "height",
    "approach",
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


def remosaic_rgb_image(rgb_image):
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    height, width, _ = bgr_image.shape
    bayer_image = np.zeros((height, width), dtype=np.uint8)
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


# matrix clippers
def clip_0_to_255(arr, check=True):
    # check for values outside the uint8 range
    if check and np.any((arr < 0) | (arr > 255)):
        raise ValueError("Array contains values outside the uint8 range")
    # round values to the closest integer and convert to uint8
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def round_to_uint8(arr):
    return clip_0_to_255(arr, False)


def unclip_0_to_255(arr):
    return arr


def clip_minus_255_to_255(arr, check=False):
    # check for values outside the int16 range
    if check and np.any((arr < -255) | (arr > 255)):
        raise ValueError("Array contains values outside the int16 range")
    # round values to the closest integer and scale to uint8
    return np.round((arr / 2) + 128).astype(np.uint8)


def unclip_minus_255_to_255(arr):
    # unscale from uint8
    return (arr.astype(np.int16) - 128) * 2


# Malvar Sullivan, "Progressive to Lossless Compression of Color Filter
# Array Images Using Macropixel Spectral Spatial Transformation", 2012
def convert_rg1g2b_to_ydgcocg(bayer_image):
    # 1. separate RGGB components
    bayer_r = bayer_image[::2, ::2]
    bayer_g1 = bayer_image[::2, 1::2]
    bayer_g2 = bayer_image[1::2, ::2]
    bayer_b = bayer_image[1::2, 1::2]
    # 2. do the color conversion
    bayer_y, bayer_dg, bayer_co, bayer_cg = convert_rg1g2b_to_ydgcocg_components(
        bayer_r, bayer_g1, bayer_g2, bayer_b
    )
    return bayer_y, bayer_dg, bayer_co, bayer_cg


def convert_rg1g2b_to_ydgcocg_components(bayer_r, bayer_g1, bayer_g2, bayer_b):
    # 1. convert values
    bayer_r = bayer_r.astype(np.int16)
    bayer_g1 = bayer_g1.astype(np.int16)
    bayer_g2 = bayer_g2.astype(np.int16)
    bayer_b = bayer_b.astype(np.int16)
    # [ Y  ]   [ 1/4  1/4  1/4  1/4 ] [ G1 ]
    # [ Dg ] = [ -1    1    0    0  ] [ G4 ]
    # [ Co ]   [  0    0    1   -1  ] [ R2 ]
    # [ Cg ]   [ 1/2  1/2 -1/2 -1/2 ] [ B3 ]
    bayer_y = (
        (1 / 4) * bayer_g1 + (1 / 4) * bayer_g2 + (1 / 4) * bayer_r + (1 / 4) * bayer_b
    )
    bayer_dg = (-1) * bayer_g1 + (1) * bayer_g2
    bayer_co = (1) * bayer_r + (-1) * bayer_b
    bayer_cg = (
        (1 / 2) * bayer_g1
        + (1 / 2) * bayer_g2
        + (-1 / 2) * bayer_r
        + (-1 / 2) * bayer_b
    )
    # 2. clip matrices to uint8
    bayer_y = clip_0_to_255(bayer_y)
    bayer_dg = clip_minus_255_to_255(bayer_dg)
    bayer_co = clip_minus_255_to_255(bayer_co)
    bayer_cg = clip_minus_255_to_255(bayer_cg)
    return bayer_y, bayer_dg, bayer_co, bayer_cg


def convert_ydgcocg_to_rg1g2b(bayer_y, bayer_dg, bayer_co, bayer_cg):
    # 1. do the color conversion
    bayer_r, bayer_g1, bayer_g2, bayer_b = convert_ydgcocg_to_rg1g2b_components(
        bayer_y, bayer_dg, bayer_co, bayer_cg
    )
    # 2. merge RGGB components
    bayer_image = np.zeros(list(2 * dim for dim in bayer_r.shape), dtype=bayer_r.dtype)
    bayer_image[::2, ::2] = bayer_r
    bayer_image[::2, 1::2] = bayer_g1
    bayer_image[1::2, ::2] = bayer_g2
    bayer_image[1::2, 1::2] = bayer_b
    return bayer_image


def convert_ydgcocg_to_rg1g2b_components(bayer_y, bayer_dg, bayer_co, bayer_cg):
    # 1. unclip matrices from uint8
    bayer_y = unclip_0_to_255(bayer_y)
    bayer_dg = unclip_minus_255_to_255(bayer_dg)
    bayer_co = unclip_minus_255_to_255(bayer_co)
    bayer_cg = unclip_minus_255_to_255(bayer_cg)
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
    bayer_g1 = (1) * bayer_y + (-1 / 2) * bayer_dg + (1 / 2) * bayer_cg
    bayer_g2 = (1) * bayer_y + (1 / 2) * bayer_dg + (1 / 2) * bayer_cg
    bayer_r = (1) * bayer_y + (1 / 2) * bayer_co + (-1 / 2) * bayer_cg
    bayer_b = (1) * bayer_y + (-1 / 2) * bayer_co + (-1 / 2) * bayer_cg
    # 3. round to uint8
    bayer_r = round_to_uint8(bayer_r)
    bayer_g1 = round_to_uint8(bayer_g1)
    bayer_g2 = round_to_uint8(bayer_g2)
    bayer_b = round_to_uint8(bayer_b)
    return bayer_r, bayer_g1, bayer_g2, bayer_b


def read_bayer_image(infile, width, height):
    # read the input file
    with open(infile, "rb") as fin:
        raw_contents = fin.read()
    assert width * height == len(
        raw_contents
    ), f"Image {infile} has size {len(raw_contents)} != {width * height} ({width=} * {height=})"
    # reshape the array as an MxN binary array
    binary_list = [int(byte) for byte in raw_contents]
    bayer_image = np.array(binary_list, dtype=np.uint8).reshape(width, height)
    return bayer_image


# Bayer processing stack
def process_file_bayer(
    infile,
    width,
    height,
    quality_list,
    workdir,
    cleanup,
    debug,
):
    bayer_image = read_bayer_image(infile, width, height)
    return process_file_bayer_array(
        bayer_image,
        infile,
        quality_list,
        workdir,
        cleanup,
        debug,
    )


def process_file_bayer_array(
    bayer_image,
    infile,
    quality_list,
    workdir,
    cleanup,
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
    bayer_y, bayer_dg, bayer_co, bayer_cg = convert_rg1g2b_to_ydgcocg(bayer_image)

    for quality in quality_list:
        # 4. encode the 4 planes
        success, bayer_y_encoded = cv2.imencode(
            ".jpg", bayer_y, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, bayer_dg_encoded = cv2.imencode(
            ".jpg", bayer_dg, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, bayer_co_encoded = cv2.imencode(
            ".jpg", bayer_co, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, bayer_cg_encoded = cv2.imencode(
            ".jpg", bayer_cg, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        # 5. decode the jpeg images
        bayer_y_prime = cv2.imdecode(bayer_y_encoded, cv2.IMREAD_GRAYSCALE)
        bayer_dg_prime = cv2.imdecode(bayer_dg_encoded, cv2.IMREAD_GRAYSCALE)
        bayer_co_prime = cv2.imdecode(bayer_co_encoded, cv2.IMREAD_GRAYSCALE)
        bayer_cg_prime = cv2.imdecode(bayer_cg_encoded, cv2.IMREAD_GRAYSCALE)

        # 6. convert YDgCoCg image back to Bayer
        bayer_image_prime = convert_ydgcocg_to_rg1g2b(
            bayer_y_prime, bayer_dg_prime, bayer_co_prime, bayer_cg_prime
        )

        # 7. demosaic raw image to RGB
        rgb_image_prime = cv2.cvtColor(bayer_image_prime, cv2.COLOR_BAYER_BG2RGB)
        rgb_r_prime, rgb_g_prime, rgb_b_prime = cv2.split(rgb_image_prime)

        # 8. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 9. calculate results
        # sizes
        encoded_size_bayer_y = len(bayer_y_encoded)
        encoded_size_bayer_dg = len(bayer_dg_encoded)
        encoded_size_bayer_co = len(bayer_co_encoded)
        encoded_size_bayer_cg = len(bayer_cg_encoded)
        encoded_size_list = [
            encoded_size_bayer_y,
            encoded_size_bayer_dg,
            encoded_size_bayer_co,
            encoded_size_bayer_cg,
        ]
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
            "bayer",
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
def process_file_yuv(
    infile,
    width,
    height,
    quality_list,
    workdir,
    cleanup,
    debug,
):
    bayer_image = read_bayer_image(infile, width, height)
    return process_file_yuv_array(
        bayer_image,
        infile,
        quality_list,
        workdir,
        cleanup,
        debug,
    )


def process_file_yuv_array(
    bayer_image,
    infile,
    quality_list,
    workdir,
    cleanup,
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
        success, yuv_y_encoded = cv2.imencode(
            ".jpg", yuv_y, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, yuv_u_encoded = cv2.imencode(
            ".jpg", yuv_u, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, yuv_v_encoded = cv2.imencode(
            ".jpg", yuv_v, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        # 4. decode the jpeg images
        yuv_y_prime = cv2.imdecode(yuv_y_encoded, cv2.IMREAD_GRAYSCALE)
        yuv_u_prime = cv2.imdecode(yuv_u_encoded, cv2.IMREAD_GRAYSCALE)
        yuv_v_prime = cv2.imdecode(yuv_v_encoded, cv2.IMREAD_GRAYSCALE)
        yuv_image_prime = cv2.merge([yuv_y_prime, yuv_u_prime, yuv_v_prime])

        # 5. convert YUV image back to RGB
        rgb_image_prime = cv2.cvtColor(yuv_image_prime, cv2.COLOR_YUV2RGB)
        rgb_b_prime, rgb_g_prime, rgb_r_prime = cv2.split(rgb_image_prime)

        # 6. remosaic RGB image back to raw
        bayer_image_prime = remosaic_rgb_image(rgb_image_prime)

        # 7. calculate results
        # sizes
        encoded_size_yuv_y = len(yuv_y_encoded)
        encoded_size_yuv_u = len(yuv_u_encoded)
        encoded_size_yuv_v = len(yuv_v_encoded)
        encoded_size_list = [encoded_size_yuv_y, encoded_size_yuv_u, encoded_size_yuv_v]
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
            "yuv",
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
    quality_list,
    workdir,
    cleanup,
    debug,
):
    bayer_image = read_bayer_image(infile, width, height)
    return process_file_rgb_array(
        bayer_image,
        infile,
        quality_list,
        workdir,
        cleanup,
        debug,
    )


def process_file_rgb_array(
    bayer_image,
    infile,
    quality_list,
    workdir,
    cleanup,
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
        success, rgb_r_encoded = cv2.imencode(
            ".jpg", rgb_r, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, rgb_g_encoded = cv2.imencode(
            ".jpg", rgb_g, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        success, rgb_b_encoded = cv2.imencode(
            ".jpg", rgb_b, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        # 4. decode the jpeg images
        rgb_r_prime = cv2.imdecode(rgb_r_encoded, cv2.IMREAD_GRAYSCALE)
        rgb_g_prime = cv2.imdecode(rgb_g_encoded, cv2.IMREAD_GRAYSCALE)
        rgb_b_prime = cv2.imdecode(rgb_b_encoded, cv2.IMREAD_GRAYSCALE)
        rgb_image_prime = cv2.merge([rgb_r_prime, rgb_g_prime, rgb_b_prime])

        # 5. convert RGB to YUV
        yuv_image_prime = cv2.cvtColor(rgb_image_prime, cv2.COLOR_RGB2YUV)
        yuv_y_prime, yuv_u_prime, yuv_v_prime = cv2.split(yuv_image_prime)

        # 6. remosaic RGB image back to raw
        bayer_image_prime = remosaic_rgb_image(rgb_image_prime)

        # 7. calculate results
        # sizes
        encoded_size_rgb_r = len(rgb_r_encoded)
        encoded_size_rgb_g = len(rgb_g_encoded)
        encoded_size_rgb_b = len(rgb_b_encoded)
        encoded_size_list = [encoded_size_rgb_r, encoded_size_rgb_g, encoded_size_rgb_b]
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
            "rgb",
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
    quality_list,
    width,
    height,
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
        # 2.1. run the Bayer-encoding pipeline
        tmp_df = process_file_bayer(
            infile, width, height, quality_list, workdir, cleanup, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.2. run the traditional-encoding pipeline
        tmp_df = process_file_yuv(
            infile, width, height, quality_list, workdir, cleanup, debug
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
        # 2.3. run the RGB-encoding pipeline
        tmp_df = process_file_rgb(
            infile, width, height, quality_list, workdir, cleanup, debug
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
        options.quality_list,
        options.width,
        options.height,
        options.workdir,
        options.outfile,
        options.cleanup,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
