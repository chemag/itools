#!/usr/bin/env python3

"""itools-bayer-enctool.py module description.

This is a tool to test Bayer image encoding.

"""


import argparse
import cv2
import importlib
import io
import itertools
import json
import math
import numpy as np
import os
import pandas as pd
import PIL
import pillow_heif
import re
import shutil
import sys
import tempfile

itools_version = importlib.import_module("itools-version")
itools_common = importlib.import_module("itools-common")
itools_bayer = importlib.import_module("itools-bayer")
itools_y4m = importlib.import_module("itools-y4m")


DEFAULT_QUALITIES = [25, 75, 85, 95, 96, 97, 98, 99]
DEFAULT_QUALITY_LIST = sorted(set(list(range(0, 101, 10)) + DEFAULT_QUALITIES))

CODEC_LIST = ("nocodec", "jpeg/cv2", "heic/libheif")

# dtype operation
# We use 2x dtype values
# * (1) st_dtype (storage dtype): This is uint8 for 8-bit Bayer, uint16 for
#   higher bit depths.
# * (2) op_dtype (operation dtype): This is int32 in all cases.
ST_DTYPE_8BIT = np.uint8
ST_DTYPE_16BIT = np.uint16
OP_DTYPE = np.int32


EXPERIMENT_DICT = {
    "bayer-ydgcocg": {
        "name": "Bayer-ydgcocg",
    },
    "bayer-ydgcocg-420": {
        "name": "Bayer-ydgcocg-subsampled",
    },
    "yuv444": {
        "name": "YUV444",
    },
    "yuv420": {
        "name": "YUV420",
    },
    "rgb": {
        "name": "RGB",
    },
    "bayer-single": {
        "name": "Bayer-single",
    },
    "bayer-rggb": {
        "name": "Bayer-rggb",
    },
}


default_values = {
    "debug": 0,
    "dry_run": False,
    "add_average": True,
    "psnr_infinity": False,
    "cleanup": 1,
    "codec": "jpeg/cv2",
    "quality_list": ",".join(str(v) for v in DEFAULT_QUALITY_LIST),
    "experiment_list": ",".join(str(k) for k in EXPERIMENT_DICT.keys()),
    "workdir": tempfile.gettempdir(),
    "width": -1,
    "height": -1,
    "pix_fmt": "bayer_rggb8",
    "infile_list": None,
    "outfile": None,
}


COLUMN_LIST = [
    "infile",
    "width",
    "height",
    "pix_fmt",
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


def ydgcocg_subsample_planar(ydgcocg_planar):
    ydgcocg_subsampled_planar = ydgcocg_planar.copy()
    ydgcocg_subsampled_planar["C"] = ydgcocg_subsampled_planar["C"][::2, ::2]
    ydgcocg_subsampled_planar["c"] = ydgcocg_subsampled_planar["c"][::2, ::2]
    return ydgcocg_subsampled_planar


def ydgcocg_upsample_planar(ydgcocg_subsampled_planar):
    ydgcocg_planar = ydgcocg_subsampled_planar.copy()
    original_shape = ydgcocg_planar["Y"].shape
    ydgcocg_planar["C"] = itools_bayer.upsample_matrix(
        ydgcocg_planar["C"], original_shape
    )
    ydgcocg_planar["c"] = itools_bayer.upsample_matrix(
        ydgcocg_planar["c"], original_shape
    )
    return ydgcocg_planar


# encoding processing
def codec_process(codec, quality, depth, planar, experiment, debug):
    if codec == "nocodec":
        return nocodec_process(quality, planar, depth, experiment, debug)
    elif codec == "jpeg/cv2" and depth == 8:
        return cv2_jpeg_process(quality, planar, depth, experiment, debug)
    elif codec == "heic/libheif" and depth in (8, 10):
        return libheif_heic_process(quality, planar, depth, experiment, debug)
    raise ValueError(f"Unimplemented {codec=}/{depth=} pair")


def nocodec_process(quality, planar, depth, experiment, debug):
    codec = "nocodec"
    planar_prime = {}
    encoded_size_dict = {}
    for key in planar:
        array = planar[key]
        if debug > 1:
            write_single_plane_to_y4m(
                array,
                f"experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}",
                depth,
            )
        # 1. encode array into blob
        height, width = array.shape
        encoded_size_dict[key] = width * height * (1 if depth == 8 else 2)
        # 2. decode blob into array
        array_prime = array.copy()
        planar_prime[key] = array_prime
    return planar_prime, encoded_size_dict


def cv2_jpeg_process(quality, planar, depth, experiment, debug):
    codec = "jpeg"
    planar_prime = {}
    encoded_size_dict = {}
    for key in planar:
        array = planar[key]
        if debug > 1:
            write_single_plane_to_y4m(
                array,
                f"experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}.before",
                depth,
            )
        # 1. encode array into blob
        success, array_encoded = cv2.imencode(
            ".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if debug > 1:
            filename = f"/tmp/itools.bayer.test.experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}.quality_{quality}.output.jpg"
            array_encoded.tofile(filename)

        encoded_size_dict[key] = len(array_encoded)
        # 2. decode blob into array
        array_prime = cv2.imdecode(array_encoded, cv2.IMREAD_GRAYSCALE)
        if debug > 1:
            write_single_plane_to_y4m(
                array_prime,
                f"experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}.after",
                depth,
            )
        planar_prime[key] = array_prime
    return planar_prime, encoded_size_dict


def libheif_heic_process(quality, planar, depth, experiment, debug):
    codec = "heic"
    planar_prime = {}
    encoded_size_dict = {}
    # explicitly register the HEIF format
    pillow_heif.register_heif_opener()
    for key in planar:
        array = planar[key]
        if debug > 1:
            write_single_plane_to_y4m(
                array,
                f"experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}.before",
                depth,
            )
        # 1. encode array into blob
        height, width = array.shape
        dtype = array.dtype
        if depth == 8 and dtype == np.uint8:
            # grayscale for 8-bit images
            mode = "L"
        elif depth == 10 and dtype == np.uint16:
            # grayscale for 10-bit imags
            mode = "I;16"
        else:
            raise ValueError(f"Invalid image type {depth=}/{dtype=}")
        # create a PIL Image from the NumPy array
        pil_image = PIL.Image.fromarray(array, mode=mode)
        # save the PIL Image to HEIF format in memory
        output = io.BytesIO()
        try:
            pil_image.save(output, format="HEIF", quality=quality)
            array_encoded = output.getvalue()
        except Exception as e:
            raise RuntimeError(f"Error during HEIF encoding: {e}")
        finally:
            output.close()
        if debug > 1:
            filename = f"/tmp/itools.bayer.test.experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}.quality_{quality}.output.heic"
            with open(filename, "wb") as fout:
                fout.write(array_encoded)
        encoded_size_dict[key] = len(array_encoded)
        # 2. decode blob into array
        img = PIL.Image.open(io.BytesIO(array_encoded))
        array_prime = np.array(img)
        if debug > 1:
            write_single_plane_to_y4m(
                array_prime,
                f"experiment_{experiment}.quality_{quality}.codec_{codec}.plane_{key}.after",
                depth,
            )
        planar_prime[key] = array_prime
    return planar_prime, encoded_size_dict


# read a Bayer image, and make sure the packed version is opencv-friendly
def read_bayer_image(infile, pix_fmt, width, height, debug):
    pix_fmt = itools_bayer.get_canonical_input_pix_fmt(pix_fmt)

    # read the input image
    bayer_image = itools_bayer.BayerImage.FromFile(infile, pix_fmt, width, height)
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # convert to a same-depth format that is opencv-friendly
    o_pix_fmt = itools_bayer.CV2_OPERATION_PIX_FMT_DICT[depth]
    o_pix_fmt = itools_bayer.get_canonical_output_pix_fmt(o_pix_fmt)
    bayer_image_cv2 = itools_bayer.BayerImage.FromBayerPlanar(
        bayer_image.GetBayerPlanar(),
        o_pix_fmt,
        bayer_image.infile,
        debug,
    )
    return bayer_image_cv2


# debug functions
def yuv_planar_to_opencv_yvu(yuv_planar):
    plane_yvu = np.stack((yuv_planar["y"], yuv_planar["v"], yuv_planar["u"]), axis=-1)
    return plane_yvu


def write_yuv_planar_to_y4m(
    yuv_planar, label, depth, colorrange=itools_common.ColorRange.full
):
    plane_yvu = yuv_planar_to_opencv_yvu(yuv_planar)
    y4mfile = tempfile.NamedTemporaryFile(
        prefix=f"itools-bayer-enctools.{label}.", suffix=".y4m"
    ).name
    colorspace = "444" if depth == 8 else "444p10"
    itools_y4m.write_y4m(
        y4mfile, plane_yvu, colorspace=colorspace, colorrange=colorrange
    )


def write_single_plane_to_y4m(
    array, label, depth, colorrange=itools_common.ColorRange.full
):
    y4mfile = tempfile.NamedTemporaryFile(
        prefix=f"itools-bayer-enctools.{label}.", suffix=".y4m"
    ).name
    colorspace = "mono" if depth == 8 else "mono10"
    itools_y4m.write_y4m(y4mfile, array, colorspace=colorspace, colorrange=colorrange)


# Bayer processing stack
def process_file_ydgcocg_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    # 3. demosaic Bayer image to YDgCoCg
    ydgcocg_planar = bayer_image.GetYDgCoCgPlanar()

    for quality in quality_list:
        if debug > 1:
            print(
                f"# ... {bayer_image.infile}: experiment: {experiment} quality: {quality}"
            )
        # 4. encode and decode the 4 planes
        ydgcocg_planar_prime, encoded_size_dict = codec_process(
            codec, quality, depth, ydgcocg_planar, experiment, debug
        )
        bayer_image_prime = itools_bayer.BayerImage.FromYDgCoCgPlanar(
            ydgcocg_planar_prime, bayer_image.pix_fmt, debug
        )

        # 6. demosaic raw image to RGB
        rgb_planar_prime = bayer_image_prime.GetRGBPlanar()

        # 7. convert RGB to YUV
        yuv_planar_prime = bayer_image_prime.GetYUVPlanar()
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 8. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


# Bayer processing stack (YDgCoCg 4:2:0)
def process_file_ydgcocg_420_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    # 3. demosaic Bayer image to YDgCoCg
    ydgcocg_planar = bayer_image.GetYDgCoCgPlanar()

    # 4. subsample the chromas
    ydgcocg_subsampled_planar = ydgcocg_subsample_planar(ydgcocg_planar)

    for quality in quality_list:
        if debug > 1:
            print(f"# ... {bayer_image.infile}: {experiment} {quality}")
        # 5. encode and decode the 4 planes
        # encode and decode the 4 planes
        ydgcocg_subsampled_planar_prime, encoded_size_dict = codec_process(
            codec, quality, depth, ydgcocg_subsampled_planar, experiment, debug
        )

        # 6. upsample the chromas
        ydgcocg_planar_prime = ydgcocg_upsample_planar(ydgcocg_subsampled_planar_prime)

        # 7. convert YDgCoCg image back to Bayer
        bayer_image_prime = itools_bayer.BayerImage.FromYDgCoCgPlanar(
            ydgcocg_planar_prime, bayer_image.pix_fmt, debug
        )

        # 8. demosaic raw image to RGB
        rgb_planar_prime = bayer_image_prime.GetRGBPlanar()

        # 9. convert RGB to YUV
        yuv_planar_prime = bayer_image_prime.GetYUVPlanar()
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 10. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


# Bayer processing stack (single plane encoding)
def process_file_bayer_single_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    for quality in quality_list:
        if debug > 1:
            print(f"# ... {bayer_image.infile}: {experiment} {quality}")
        # 3. encode and decode the single planes
        bayer_packed_dict = {"bayer": bayer_image.GetBayerPacked()}
        bayer_packed_prime_dict, encoded_size_dict = codec_process(
            codec, quality, depth, bayer_packed_dict, experiment, debug
        )
        bayer_packed_prime = bayer_packed_prime_dict["bayer"]
        bayer_image_prime = itools_bayer.BayerImage.FromBayerPacked(
            bayer_packed_prime, bayer_image.pix_fmt
        )

        # 4. demosaic raw image to RGB
        rgb_planar_prime = bayer_image_prime.GetRGBPlanar()

        # 5. convert RGB to YUV
        yuv_planar_prime = bayer_image_prime.GetYUVPlanar()
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 7. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


# Bayer processing stack (plane encoding)
def process_file_bayer_rggb_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    for quality in quality_list:
        if debug > 1:
            print(f"# ... {bayer_image.infile}: {experiment} {quality}")
        # 3. encode and decode the 4 Bayer planes
        bayer_planar_prime, encoded_size_dict = codec_process(
            codec, quality, depth, bayer_image.GetBayerPlanar(), experiment, debug
        )
        bayer_image_prime = itools_bayer.BayerImage.FromBayerPlanar(
            bayer_planar_prime, bayer_image.pix_fmt
        )

        # 4. demosaic raw image to RGB
        rgb_planar_prime = bayer_image_prime.GetRGBPlanar()

        # 5. convert RGB to YUV
        yuv_planar_prime = bayer_image_prime.GetYUVPlanar()
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 7. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


# YUV processing stack, 4:4:4
def process_file_yuv444_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    for quality in quality_list:
        if debug > 1:
            print(f"# ... {bayer_image.infile}: {experiment} {quality}")
        # 3. encode and decode the 3 planes
        yuv_planar_prime, encoded_size_dict = codec_process(
            codec, quality, depth, yuv_planar, experiment, debug
        )
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 4. convert YUV image back to RGB
        rgb_planar_prime = itools_bayer.yuv_planar_to_rgb_planar(
            yuv_planar_prime, depth
        )

        # 5. remosaic RGB image back to raw
        bayer_image_prime = itools_bayer.rgb_planar_to_bayer_image(
            rgb_planar_prime, bayer_image.pix_fmt
        )

        # 6. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


# YUV processing stack, 4:2:0 (traditional camera stack)
def process_file_yuv420_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    # 3. subsample the chromas
    yuv_subsampled_planar = itools_bayer.yuv_subsample_planar(yuv_planar)

    for quality in quality_list:
        if debug > 1:
            print(f"# ... {bayer_image.infile}: {experiment} {quality}")
        # 4. encode and decode the planes
        yuv_subsampled_planar_prime, encoded_size_dict = codec_process(
            codec, quality, depth, yuv_subsampled_planar, experiment, debug
        )

        # 5. upsample the chromas
        yuv_planar_prime = itools_bayer.yuv_upsample_planar(yuv_subsampled_planar_prime)
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 6. convert YUV image back to RGB
        rgb_planar_prime = itools_bayer.yuv_planar_to_rgb_planar(
            yuv_planar_prime, depth
        )

        # 7. remosaic RGB image back to raw
        bayer_image_prime = itools_bayer.rgb_planar_to_bayer_image(
            rgb_planar_prime, bayer_image.pix_fmt
        )

        # 8. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


# RGB camera stack
def process_file_rgb_array(
    experiment,
    bayer_image,
    codec,
    quality_list,
    debug,
):
    if debug > 0:
        print(f"# {bayer_image.infile}: {experiment}")
    df = pd.DataFrame(columns=COLUMN_LIST)
    width, height = bayer_image.width, bayer_image.height
    depth = itools_bayer.get_depth(bayer_image.pix_fmt)

    # 1. demosaic raw image to RGB
    rgb_planar = bayer_image.GetRGBPlanar()

    # 2. convert RGB to YUV
    yuv_planar = bayer_image.GetYUVPlanar()
    if debug > 1:
        write_yuv_planar_to_y4m(
            yuv_planar, f"experiment_{experiment}.yuv_planar", depth
        )

    for quality in quality_list:
        if debug > 1:
            print(f"# ... {bayer_image.infile}: {experiment} {quality}")
        # 3. encode and decode the 3 planes
        rgb_planar_prime, encoded_size_dict = codec_process(
            codec, quality, depth, rgb_planar, experiment, debug
        )

        # 4. convert RGB to YUV
        yuv_planar_prime = itools_bayer.rgb_planar_to_yuv_planar(
            rgb_planar_prime, depth
        )
        if debug > 1:
            write_yuv_planar_to_y4m(
                yuv_planar,
                f"experiment_{experiment}.yuv_planar_prime.quality_{quality}",
                depth,
            )

        # 5. remosaic RGB image back to raw
        bayer_image_prime = itools_bayer.rgb_planar_to_bayer_image(
            rgb_planar_prime, bayer_image.pix_fmt
        )

        # 6. calculate results
        # sizes
        encoded_size = sum(encoded_size_dict.values())
        encoded_bpp = (encoded_size * 8.0) / (width * height)
        # psnr values
        # psnr values: YUV
        global psnr_infinity
        psnr_yuv_dict = itools_common.calculate_psnr(
            yuv_planar, yuv_planar_prime, depth, psnr_infinity
        )
        psnr_yuv = np.mean(list(psnr_yuv_dict.values()))
        # psnr values: RGB
        psnr_rgb_dict = itools_common.calculate_psnr(
            rgb_planar, rgb_planar_prime, depth, psnr_infinity
        )
        psnr_rgb = np.mean(list(psnr_rgb_dict.values()))
        # psnr values: Bayer
        psnr_bayer = itools_common.calculate_psnr(
            bayer_image.GetBayerPacked(),
            bayer_image_prime.GetBayerPacked(),
            depth,
            psnr_infinity,
        )
        # add new element
        df.loc[df.size] = (
            bayer_image.infile,
            bayer_image.width,
            bayer_image.height,
            bayer_image.pix_fmt,
            bayer_image.depth,
            experiment,
            codec,
            quality,
            encoded_size,
            ":".join(str(size) for size in encoded_size_dict.values()),
            encoded_bpp,
            psnr_bayer,
            psnr_rgb,
            psnr_rgb_dict["r"],
            psnr_rgb_dict["g"],
            psnr_rgb_dict["b"],
            psnr_yuv,
            psnr_yuv_dict["y"],
            psnr_yuv_dict["u"],
            psnr_yuv_dict["v"],
        )

    return df


PROCESS_FILE_ARRAY_FUN = {
    "bayer-ydgcocg": process_file_ydgcocg_array,
    "bayer-ydgcocg-420": process_file_ydgcocg_420_array,
    "yuv444": process_file_yuv444_array,
    "yuv420": process_file_yuv420_array,
    "rgb": process_file_rgb_array,
    "bayer-single": process_file_bayer_single_array,
    "bayer-rggb": process_file_bayer_rggb_array,
}


def process_file_array(bayer_image, experiment, codec, quality_list, debug):
    # run the specific encoding pipeline
    df = PROCESS_FILE_ARRAY_FUN[experiment](
        experiment, bayer_image, codec, quality_list, debug
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
    experiment_list,
    codec,
    quality_list,
    width,
    height,
    pix_fmt,
    workdir,
    outfile,
    add_average,
    cleanup,
    debug,
):
    df = None

    # 1. run the camera pipelines
    for infile in infile_list:
        # 1.1. read input image
        bayer_image = read_bayer_image(infile, pix_fmt, width, height, debug)
        for experiment in experiment_list:
            # 1.2. run experiment
            tmp_df = process_file_array(
                bayer_image,
                experiment,
                codec,
                quality_list,
                debug,
            )
            df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)

    # 2. reindex per-file dataframe
    df = df.reindex()

    # 3. get average results
    if add_average:
        derived_df = get_average_results(df)
        df = pd.concat([df, derived_df], ignore_index=True, axis=0)

    # 4. write the results
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
        "--add-average",
        action="store_true",
        dest="add_average",
        default=default_values["add_average"],
        help="Add average%s" % (" [default]" if default_values["add_average"] else ""),
    )
    parser.add_argument(
        "--no-add-average",
        action="store_false",
        dest="add_average",
        help="Do not add average%s"
        % (" [default]" if not default_values["add_average"] else ""),
    )
    parser.add_argument(
        "--psnr-infinity",
        action="store_true",
        dest="psnr_infinity",
        default=default_values["psnr_infinity"],
        help="Use infinity in PSNR%s"
        % (" [default]" if default_values["psnr_infinity"] else ""),
    )
    parser.add_argument(
        "--no-psnr-infinity",
        action="store_false",
        dest="psnr_infinity",
        help="Do not use infinity in PSNR%s"
        % (" [default]" if not default_values["psnr_infinity"] else ""),
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
        "--codec-list",
        dest="show_codec_list",
        action="store_true",
        default=False,
        help="List available codecs and exit",
    )
    parser.add_argument(
        "--experiment",
        action="store",
        type=str,
        dest="experiment_list",
        default=default_values["experiment_list"],
        help="Experiment list (comma-separated list)",
    )
    parser.add_argument(
        "--experiment-list",
        action="store_true",
        dest="show_experiment_list",
        default=False,
        help="Show the full experiment list",
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

    input_choices_str = " | ".join(itools_bayer.I_PIX_FMT_LIST)
    parser.add_argument(
        "--pix_fmt",
        action="store",
        type=str,
        dest="pix_fmt",
        default=default_values["pix_fmt"],
        choices=itools_bayer.I_PIX_FMT_LIST,
        metavar=f"[{input_choices_str}]",
        help="input pixel format",
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
    # parse quick options
    if options.show_codec_list:
        print(f"list of valid codecs: {CODEC_LIST}")
        sys.exit()
    elif options.show_experiment_list:
        print(f"list of valid experiments: {list(EXPERIMENT_DICT.keys())}")
        sys.exit()
    return options


def main(argv):
    global psnr_infinity

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
    # fix comma-separated lists
    options.experiment_list = options.experiment_list.split(",")
    options.experiment_list.sort()
    options.quality_list = list(int(v) for v in options.quality_list.split(","))
    # process infile
    psnr_infinity = options.psnr_infinity
    process_data(
        options.infile_list,
        options.experiment_list,
        options.codec,
        options.quality_list,
        options.width,
        options.height,
        options.pix_fmt,
        options.workdir,
        options.outfile,
        options.add_average,
        options.cleanup,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
