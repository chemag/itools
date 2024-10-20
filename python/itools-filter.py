#!/usr/bin/env python3

"""filter.py module description.

Runs generic image transformation on input images.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import argparse
import cv2
import enum
import itertools
import math
import numpy as np
import os.path
import pandas as pd
import sys
import importlib

itools_common = importlib.import_module("itools-common")
itools_io = importlib.import_module("itools-io")
itools_rgb = importlib.import_module("itools-rgb")
itools_y4m = importlib.import_module("itools-y4m")


DEFAULT_NOISE_LEVEL = 50
PSNR_K = {
    itools_common.ColorDepth.depth_8: math.log10(2**8 - 1),
    itools_common.ColorDepth.depth_10: math.log10(2**10 - 1),
}
ROTATE_ANGLE_LIST = {
    0: 0,
    90: 1,
    180: 2,
    270: -1,
    -90: -1,
    360: 4,
}
DIFF_COMPONENT_LIST = ("y", "u", "v")
HIST_COMPONENT_LIST = ("y", "u", "v", "r", "g", "b")

FILTER_CHOICES = {
    "help": "show help options",
    "copy": "copy input to output",
    "gray": "convert image to GRAY scale",
    "xchroma": "swap U and V chromas",
    "mix-images": "mix image components between 2x files",
    "xrgb2yuv": "swap RGB components as YUV",
    "noise": "add noise",
    "diff": "diff 2 frames",
    "mse": "get the MSE/PSNR of a frame",
    "psnr": "get the MSE/PSNR of a frame",
    "histogram": "get a histogram of the values of a given component",
    "compose": "compose 2 frames",
    "rotate": "rotate a frame (90, 180, 270)",
    "match": "match 2 frames (needle and haystack problem -- only shift)",
    "affine": "run an affine transformation (defined as 2x matrices A and B) on the input",
    "affine-points": "run an affine transformation (defined as 2x set of 3x points) on the input",
}


PROC_COLOR_LIST = list(c.name for c in itools_common.ProcColor)

default_values = {
    "debug": 0,
    "dry_run": False,
    "proc_color": "bgr",
    "filter": "help",
    "noise_level": DEFAULT_NOISE_LEVEL,
    "mse_invert": False,
    "diff_invert": True,
    "diff_factor": 1.0,
    "diff_component": "y",
    "diff_color": False,
    "diff_color_factor": 5,
    "hist_component": "y",
    "rotate_angle": 90,
    "mix_definition": "y1y2y2",
    "x": 10,
    "y": 20,
    "iwidth": 0,
    "iheight": 0,
    "istride": None,
    "iscanline": None,
    "icolorrange": itools_common.ColorRange.get_default(),
    "width": 0,
    "height": 0,
    "a00": 1,
    "a01": 0,
    "a10": 0,
    "a11": 1,
    "b00": 0,
    "b10": 0,
    "s0x": 0,
    "s0y": 0,
    "s1x": 0,
    "s1y": 100,
    "s2x": 100,
    "s2y": 0,
    "d0x": 0,
    "d0y": 0,
    "d1x": 0,
    "d1y": 100,
    "d2x": 100,
    "d2y": 0,
    "infile": None,
    "infile2": None,
    "outfile": None,
    "logfile": None,
}


def image_to_gray(
    infile, outfile, iinfo, proc_color, config_dict, cleanup, logfd, debug
):
    assert (
        proc_color == itools_common.ProcColor.bgr
    ), f"error: image_to_gray unsupported in {proc_color}"
    # load the input image
    inbgr, status = itools_io.read_image_file(
        infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr is not None, f"error: cannot read {infile}"
    # convert to gray
    tmpgray = cv2.cvtColor(inbgr, cv2.COLOR_BGR2GRAY)
    outbgr = cv2.cvtColor(tmpgray, cv2.COLOR_GRAY2BGR)
    # store the output image
    itools_io.write_image_file(outfile, outbgr, **status)


def swap_xchroma(
    infile, outfile, iinfo, proc_color, config_dict, cleanup, logfd, debug
):
    assert (
        proc_color == itools_common.ProcColor.bgr
    ), f"error: swap_xchroma unsupported in {proc_color}"
    # load the input image
    inbgr, status = itools_io.read_image_file(
        infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr is not None, f"error: cannot read {infile}"
    # swap chromas
    tmpyvu = cv2.cvtColor(inbgr, cv2.COLOR_BGR2YCrCb)
    outyvu = tmpyvu[:, :, [0, 2, 1]]
    outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
    # store the output image
    itools_io.write_image_file(outfile, outbgr, **status)


def swap_xrgb2yuv(
    infile, outfile, iinfo, proc_color, config_dict, cleanup, logfd, debug
):
    assert (
        proc_color == itools_common.ProcColor.bgr
    ), f"error: swap_xrgb2yuv unsupported in {proc_color}"
    # load the input image
    inbgr, status = itools_io.read_image_file(
        infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr is not None, f"error: cannot read {infile}"
    inrgb = inbgr[:, :, [2, 1, 0]]
    # swap RGB to YUV
    outyuv = inrgb
    outyvu = outyuv[:, :, [0, 2, 1]]
    outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
    # store the output image
    itools_io.write_image_file(outfile, outbgr, **status)


MIX_DEFINITION_COMPONENTS = ("y1", "u1", "v1", "y2", "u2", "v2")


def get_component(inyvu1, inyvu2, component):
    if component == "y1":
        return inyvu1[:, :, [0]]
    elif component == "y2":
        return inyvu2[:, :, [0]]
    elif component == "u1":
        return inyvu1[:, :, [2]]
    elif component == "u2":
        return inyvu2[:, :, [2]]
    elif component == "v1":
        return inyvu1[:, :, [1]]
    elif component == "v2":
        return inyvu2[:, :, [1]]


def mix_images(
    infile1,
    infile2,
    mix_definition,
    outfile,
    iinfo,
    config_dict,
    cleanup,
    logfd,
    debug,
):
    # load the input images as YVU
    inyvu1, instatus1 = itools_io.read_image_file(
        infile1,
        config_dict,
        iinfo=iinfo,
        return_type=itools_common.ProcColor.yvu,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    inyvu2, instatus2 = itools_io.read_image_file(
        infile2,
        config_dict,
        iinfo=iinfo,
        return_type=itools_common.ProcColor.yvu,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inyvu1 is not None, f"error: cannot read {infile1}"
    assert inyvu2 is not None, f"error: cannot read {infile2}"
    # make sure the images are the same size
    assert inyvu1.shape == inyvu2.shape
    # check the mix definition
    assert len(mix_definition) == 6, f"error: invalid {mix_definition=}"
    components = mix_definition[:2], mix_definition[2:4], mix_definition[4:]
    for component in components:
        assert component in MIX_DEFINITION_COMPONENTS, f"error: invalid {component=}"
    # compose an output image using as the luma/chroma from the 2x input images
    outyvu = inyvu1.copy()
    outyvu[:, :, [0]] = get_component(inyvu1, inyvu2, components[0])
    outyvu[:, :, [2]] = get_component(inyvu1, inyvu2, components[1])
    outyvu[:, :, [1]] = get_component(inyvu1, inyvu2, components[2])
    # store the output image
    itools_io.write_image_file(
        outfile, outyvu, return_type=itools_common.ProcColor.yvu, **instatus1
    )


def add_noise(
    infile, outfile, iinfo, noise_level, proc_color, config_dict, cleanup, logfd, debug
):
    assert (
        proc_color == itools_common.ProcColor.bgr
    ), f"error: add_noise unsupported in {proc_color}"
    # load the input image
    inbgr, status = itools_io.read_image_file(
        infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr is not None, f"error: cannot read {infile}"
    # convert to gray
    noiseimg = np.random.randint(
        -noise_level, noise_level, size=inbgr.shape, dtype=np.int16
    )
    outbgr = inbgr + noiseimg
    outbgr[outbgr > np.iinfo(np.uint8).max] = np.iinfo(np.uint8).max
    outbgr[outbgr < np.iinfo(np.uint8).min] = np.iinfo(np.uint8).min
    outbgr = outbgr.astype(np.uint8)
    # store the output image
    itools_io.write_image_file(outfile, outbgr, **status)


def copy_image(infile, outfile, iinfo, proc_color, config_dict, cleanup, logfd, debug):
    # load the input image
    inabc, status = itools_io.read_image_file(
        infile,
        config_dict,
        iinfo=iinfo,
        return_type=proc_color,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inabc is not None, f"error: cannot read {infile}"
    # write the output image
    itools_io.write_image_file(outfile, inabc, return_type=proc_color, **status)


def force_range(val):
    return val if (val <= 255 and val >= 0) else (0 if val <= 0 else 255)


def diff_color_uval(ydiff):
    return force_range(int(255 - ((DIFF_COLOR_FACTOR * ydiff + 255) / 2)))


def diff_color_vval(ydiff):
    return force_range(int(((DIFF_COLOR_FACTOR * ydiff + 255) / 2)))


def diff_images(
    infile1,
    infile2,
    outfile,
    iinfo,
    diff_invert,
    diff_factor,
    diff_component,
    diff_color,
    diff_color_factor,
    config_dict,
    cleanup,
    logfd,
    debug,
):
    # load the input images as YVU
    inyvu1, instatus1 = itools_io.read_image_file(
        infile1,
        config_dict,
        iinfo=iinfo,
        return_type=itools_common.ProcColor.yvu,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    inyvu2, instatus2 = itools_io.read_image_file(
        infile2,
        config_dict,
        iinfo=iinfo,
        return_type=itools_common.ProcColor.yvu,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inyvu1 is not None, f"error: cannot read {infile1}"
    assert inyvu2 is not None, f"error: cannot read {infile2}"
    # diff them
    diff_yvu_sign = inyvu1.astype(np.int16) - inyvu2.astype(np.int16)
    diff_yvu = np.absolute(diff_yvu_sign).astype(np.uint8)
    # calculate the energy of the diff
    yd, vd, ud = diff_yvu[:, :, 0], diff_yvu[:, :, 1], diff_yvu[:, :, 2]
    ymean, ystddev = yd.mean(), yd.std()
    umean, ustddev = ud.mean(), ud.std()
    vmean, vstddev = vd.mean(), vd.std()
    # store values in a dataframe
    df = pd.DataFrame(
        columns=(
            "infile1",
            "incolorrange1",
            "inbroken1",
            "infile2",
            "incolorrange2",
            "inbroken2",
            "ymean",
            "ystddev",
            "umean",
            "ustddev",
            "vmean",
            "vstddev",
        )
    )
    df.loc[len(df.index)] = [
        infile1,
        instatus1.get("colorrange", ""),
        instatus1.get("y4m:broken", "") if instatus1 is not None else "",
        infile2,
        instatus2.get("colorrange", ""),
        instatus2.get("y4m:broken", "") if instatus2 is not None else "",
        ymean,
        ystddev,
        umean,
        ustddev,
        vmean,
        vstddev,
    ]
    # choose the visual output
    if diff_component == "y":
        # use the luma for diff luma
        yd = yd
    elif diff_component == "u":
        # use the u for diff luma
        yd = ud
    elif diff_component == "v":
        # use the v for diff luma
        yd = vd
    # apply the diff factor
    yd_float = yd * diff_factor
    yd_float = yd_float.clip(0, 255)
    yd_float = np.around(yd_float)
    yd = yd_float.astype(np.uint8)
    if diff_invert:
        # invert the luma values
        yd = 255 - yd
    if not diff_color:
        # use gray chromas for visualization
        width, height = yd.shape
        ud = np.full((width, height), 128, dtype=np.uint8)
        vd = np.full((width, height), 128, dtype=np.uint8)
    else:
        # use color chromas for visualization
        global DIFF_COLOR_FACTOR
        DIFF_COLOR_FACTOR = diff_color_factor
        apply_uval = np.vectorize(diff_color_uval)
        apply_vval = np.vectorize(diff_color_vval)
        ud = apply_uval(diff_yvu_sign).astype(np.uint8)[:, :, 0]
        vd = apply_vval(diff_yvu_sign).astype(np.uint8)[:, :, 0]

    # combine the diff color components
    outyvu = np.stack((yd, vd, ud), axis=2)
    outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
    # write the output image
    colorrange = itools_common.ColorRange.full
    itools_io.write_image_file(
        outfile, outyvu, return_type=itools_common.ProcColor.yvu, colorrange=colorrange
    )
    return df


def mse_image(infile, iinfo, mse_invert, config_dict, cleanup, logfd, debug):
    # load the input image
    inyvu, instatus = itools_io.read_image_file(
        infile,
        config_dict,
        iinfo=iinfo,
        return_type=itools_common.ProcColor.yvu,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inyvu is not None, f"error: cannot read {infile}"
    # TODO(chema): use instatus to deal with color ranges
    # calculate the (1 - luma) mse
    luma = inyvu[:, :, 0].astype(np.int32)
    width, height = luma.shape
    colordepth = instatus.get("colordepth", itools_common.ColorDepth.depth_8)
    if mse_invert:
        mse = ((colordepth.get_max() - luma) ** 2).mean()
    else:
        mse = (luma**2).mean()
    # calculate the PSNR
    psnr_k = PSNR_K[colordepth]
    psnr = (20 * psnr_k - 10 * math.log10(mse)) if mse != 0.0 else 100
    return mse, psnr


# calculates a histogram of the luminance values
def get_histogram(
    infile, outfile, iinfo, hist_component, config_dict, cleanup, logfd, debug
):
    # load the input image
    if hist_component in ("y", "v", "u"):
        inimg, instatus = itools_io.read_image_file(
            infile,
            config_dict,
            return_type=itools_common.ProcColor.yvu,
            iinfo=iinfo,
            cleanup=cleanup,
            logfd=logfd,
            debug=debug,
        )
    else:  # hist_component in ("r", "g", "b"):
        inimg, instatus = itools_io.read_image_file(
            infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
        )
    assert inimg is not None, f"error: cannot read {infile}"
    # TODO(chema): use status (_) to deal with color ranges
    # get the requested component: note that options are YVU or BGR
    if hist_component == "y" or hist_component == "b":
        component = inimg[:, :, 0]  # inyvu, inbgr
    elif hist_component == "v" or hist_component == "g":
        component = inimg[:, :, 1]  # inyvu, inbgr
    elif hist_component == "u" or hist_component == "r":
        component = inimg[:, :, 2]  # inyvu, inbgr

    # calculate the histogram
    colordepth = instatus.get("colordepth", itools_common.ColorDepth.depth_8)
    VALUE_RANGE = colordepth.get_max() + 1
    histogram = {k: 0 for k in range(VALUE_RANGE)}
    for v in component:
        for vv in v:
            histogram[vv] += 1
    # normalize histogram
    histogram_normalized = {k: (v / component.size) for (k, v) in histogram.items()}
    # store histogram as csv
    with open(outfile, "w") as fout:
        fout.write("value,hist,ratio\n")
        for k in range(VALUE_RANGE):
            fout.write(f"{k},{histogram[k]},{histogram_normalized[k]}\n")


# rotates infile
def rotate_image(
    infile, rotate_angle, outfile, iinfo, proc_color, config_dict, cleanup, logfd, debug
):
    # load the input image
    inabc, status = itools_io.read_image_file(
        infile,
        config_dict,
        iinfo=iinfo,
        return_type=proc_color,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inabc is not None, f"error: cannot read {infile}"
    # rotate it
    num_rotations = ROTATE_ANGLE_LIST[rotate_angle]
    outabc = np.rot90(inabc, k=num_rotations, axes=(0, 1))
    # write the output image
    itools_io.write_image_file(outfile, outabc, return_type=proc_color, **status)


# composes infile2 on top of infile1, at (xloc, yloc)
# uses alpha
def compose_images(
    infile1,
    infile2,
    iinfo,
    xloc,
    yloc,
    outfile,
    proc_color,
    config_dict,
    cleanup,
    logfd,
    debug,
):
    assert (
        proc_color == itools_common.ProcColor.bgr
    ), f"error: compose_images unsupported in {proc_color}"
    # load the input images
    inbgr1, _ = itools_io.read_image_file(
        infile1, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr1 is not None, f"error: cannot read {infile1}"
    inbgr2, _ = itools_io.read_image_file(
        infile2,
        config_dict,
        cv2.IMREAD_UNCHANGED,
        iinfo=iinfo,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inbgr2 is not None, f"error: cannot read {infile2}"
    # TODO(chema): use status (_) to deal with color ranges
    # compose them
    width1, height1, _ = inbgr1.shape
    width2, height2, _ = inbgr2.shape
    assert xloc + width2 < width1
    assert yloc + height2 < height1
    if inbgr2.shape[2] == 3:
        # no alpha channel: just use 50% ((im1 + im2) / 2)
        outbgr = inbgr1.astype(np.int16)
        outbgr[yloc : yloc + height2, xloc : xloc + width2] += inbgr2
        outbgr[yloc : yloc + height2, xloc : xloc + width2] /= 2

    elif inbgr2.shape[2] == 4:
        outbgr = inbgr1.astype(np.int16)
        # TODO(chema): replace this loop with alpha-channel line
        for x2, y2 in itertools.product(range(width2), range(height2)):
            x1 = xloc + x2
            y1 = yloc + y2
            alpha_value = inbgr2[y2][x2][3] / 256
            outbgr[y1][x1] = np.rint(
                outbgr[y1][x1] * (1 - alpha_value) + inbgr2[y2][x2][:3] * alpha_value
            )

    # store the output image
    outbgr = outbgr.astype(np.uint8)
    itools_io.write_image_file(outfile, outbgr)


def match_images(
    infile1, infile2, outfile, iinfo, proc_color, config_dict, cleanup, logfd, debug
):
    assert (
        proc_color == itools_common.ProcColor.bgr
    ), f"error: match_images unsupported in {proc_color}"
    # load the input images
    inbgr1, _ = itools_io.read_image_file(
        infile1, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr1 is not None, f"error: cannot read {infile1}"
    inbgr2, _ = itools_io.read_image_file(
        infile2,
        config_dict,
        cv2.IMREAD_UNCHANGED,
        iinfo=iinfo,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    assert inbgr2 is not None, f"error: cannot read {infile2}"
    # TODO(chema): use status (_) to deal with color ranges
    # we will do gray correlation image matching: Use only the lumas
    inluma1 = cv2.cvtColor(inbgr1, cv2.COLOR_BGR2GRAY)
    inluma2 = cv2.cvtColor(inbgr2, cv2.COLOR_BGR2GRAY)
    # support needles with alpha channels
    if inbgr2.shape[2] == 3:
        # no alpha channel: just use the luma for the search
        pass
    elif inbgr2.shape[2] == 4:
        # alpha channel: add noise to the non-alpha channel parts
        # https://stackoverflow.com/a/20461136
        # TODO(chema): replace random-composed luma with alpha-channel-based
        # matchTemplate() function.
        luma2rand = np.random.randint(256, size=inluma2.shape).astype(np.int16)
        width2, height2 = inluma2.shape
        alpha_channel2 = inbgr2[:, :, 3]
        # TODO(chema): replace this loop with alpha-channel line
        for x2, y2 in itertools.product(range(width2), range(height2)):
            alpha_value = alpha_channel2[y2][x2] / 256
            luma2rand[y2][x2] = np.rint(
                luma2rand[y2][x2] * (1 - alpha_value) + inluma2[y2][x2] * alpha_value
            )
        inluma2 = luma2rand.astype(np.uint8)
    # match infile2 (template, needle) in infile1 (image, haystack)
    # Note that matchTemplate() does not support rotation or scaling
    # https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    match = cv2.matchTemplate(inluma1, inluma2, cv2.TM_CCOEFF_NORMED)
    # get the location for the highest match[] value
    y0, x0 = np.unravel_index(match.argsort(axis=None)[-1], match.shape)
    if debug > 0:
        print(f"{x0 = } {y0 = }", file=logfd)
    # prepare the output
    outbgr = inbgr1.astype(np.int16)
    xwidth, ywidth, _ = inbgr2.shape
    x1, y1 = x0 + xwidth, y0 + ywidth
    # substract the needle from the haystack
    # this replaces black with black: Not very useful
    # outbgr[y0:y1, x0:x1] -= inbgr2[:,:,:3]
    # add an X in the origin (0,0) point
    # cv2.line(outbgr, (0, 0), (2, 2), color=(0,0,0), thickness=1)
    # add an X in the (x0, y0) point
    # cv2.line(outbgr, (x0 - 2, y0 - 2), (x0 + 2, y0 + 2), color=(0, 0, 0), thickness=1)
    # cv2.line(outbgr, (x0 + 2, y0 - 2), (x0 - 2, y0 + 2), color=(0, 0, 0), thickness=1)
    # add a square in the full needle location
    cv2.rectangle(outbgr, (x0, y0), (x1, y1), color=(0, 0, 0), thickness=1)

    # store the output image
    outbgr = np.absolute(outbgr).astype(np.uint8)
    itools_io.write_image_file(outfile, outbgr)


def affine_transformation_matrix(
    infile,
    iinfo,
    outfile,
    width,
    height,
    a00,
    a01,
    a10,
    a11,
    b00,
    b10,
    config_dict,
    cleanup,
    logfd,
    debug,
):
    # load the input image
    inbgr, status = itools_io.read_image_file(
        infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr is not None, f"error: cannot read {infile}"
    # process the image
    m0 = [a00, a01, b00]
    m1 = [a10, a11, b10]
    transform_matrix = np.array([m0, m1]).astype(np.float32)
    if debug > 0:
        print(f"{transform_matrix = }", file=logfd)
    width = width if width != 0 else inbgr.shape[1]
    height = height if height != 0 else inbgr.shape[0]
    outbgr = cv2.warpAffine(inbgr, transform_matrix, (width, height))
    # store the output image
    itools_io.write_image_file(outfile, outbgr, **status)


def affine_transformation_points(
    infile,
    iinfo,
    outfile,
    width,
    height,
    s0x,
    s0y,
    s1x,
    s1y,
    s2x,
    s2y,
    d0x,
    d0y,
    d1x,
    d1y,
    d2x,
    d2y,
    config_dict,
    cleanup,
    logfd,
    debug,
):
    # load the input image
    inbgr, status = itools_io.read_image_file(
        infile, config_dict, iinfo=iinfo, cleanup=cleanup, logfd=logfd, debug=debug
    )
    assert inbgr is not None, f"error: cannot read {infile}"
    # process the image
    s0 = [s0x, s0y]
    s1 = [s1x, s1y]
    s2 = [s2x, s2y]
    src_trio = np.array([s0, s1, s2]).astype(np.float32)
    d0 = [d0x, d0y]
    d1 = [d1x, d1y]
    d2 = [d2x, d2y]
    dst_trio = np.array([d0, d1, d2]).astype(np.float32)
    transform_matrix = cv2.getAffineTransform(src_trio, dst_trio)
    if debug > 0:
        print(f"{transform_matrix = }", file=logfd)
    width = width if width != 0 else inbgr.shape[1]
    height = height if height != 0 else inbgr.shape[0]
    outbgr = cv2.warpAffine(inbgr, transform_matrix, (width, height))
    # store the output image
    itools_io.write_image_file(outfile, outbgr, **status)


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
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
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
        "--proc-color",
        action="store",
        type=str,
        default=default_values["proc_color"],
        choices=PROC_COLOR_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                PROC_COLOR_LIST,
            )
        ),
        help="processing color",
    )
    parser.add_argument(
        "--noise-level",
        action="store",
        type=int,
        dest="noise_level",
        default=default_values["noise_level"],
        help="Noise Level",
    )
    parser.add_argument(
        "--mse-invert",
        action="store_true",
        dest="mse_invert",
        default=default_values["mse_invert"],
        help="Invert luminance before MSE calculation",
    )
    parser.add_argument(
        "--no-mse-invert",
        action="store_false",
        dest="mse_invert",
        help="Do not invert luminance before MSE calculation",
    )
    parser.add_argument(
        "--diff-invert",
        action="store_true",
        dest="diff_invert",
        default=default_values["diff_invert"],
        help="Invert diff colors (differences are black over white)",
    )
    parser.add_argument(
        "--no-diff-invert",
        action="store_false",
        dest="diff_invert",
        help="Normal diff colors (differences are white over black)",
    )
    parser.add_argument(
        "--diff-factor",
        action="store",
        type=float,
        dest="diff_factor",
        default=default_values["diff_factor"],
        help="Diff Multiplication Factor",
    )
    parser.add_argument(
        "--diff-component",
        action="store",
        type=str,
        default=default_values["diff_component"],
        choices=DIFF_COMPONENT_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                DIFF_COMPONENT_LIST,
            )
        ),
        help="diff component arg",
    )
    parser.add_argument(
        "--diff-color",
        action="store_true",
        dest="diff_color",
        default=default_values["diff_color"],
        help="Produce diff in color%s"
        % (" [default]" if default_values["diff_color"] else ""),
    )
    parser.add_argument(
        "--nodiff-color",
        action="store_false",
        dest="diff_color",
        help="Produce diff in grayscale%s"
        % (" [default]" if not default_values["diff_color"] else ""),
    )
    parser.add_argument(
        "--diff-color-factor",
        action="store",
        type=float,
        dest="diff_color_factor",
        default=default_values["diff_color_factor"],
        help="Diff Color Multiplication Factor",
    )
    parser.add_argument(
        "--hist-component",
        action="store",
        type=str,
        default=default_values["hist_component"],
        choices=HIST_COMPONENT_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                HIST_COMPONENT_LIST,
            )
        ),
        help="histogram component arg",
    )
    parser.add_argument(
        "-x",
        action="store",
        type=int,
        dest="x",
        default=default_values["x"],
        help="Composition X Coordinate",
    )
    parser.add_argument(
        "-y",
        action="store",
        type=int,
        dest="y",
        default=default_values["y"],
        help="Composition Y Coordinate",
    )
    parser.add_argument(
        "--iwidth",
        action="store",
        type=int,
        dest="iwidth",
        default=default_values["iwidth"],
        metavar="WIDTH",
        help=("input WIDTH (default: %i)" % default_values["iwidth"]),
    )
    parser.add_argument(
        "--iheight",
        action="store",
        type=int,
        dest="iheight",
        default=default_values["iheight"],
        metavar="HEIGHT",
        help=("input HEIGHT (default: %i)" % default_values["iheight"]),
    )

    class VideoSizeAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.iwidth, namespace.iheight = [int(v) for v in values[0].split("x")]

    parser.add_argument(
        "--isize",
        action=VideoSizeAction,
        nargs=1,
        help="use <width>x<height>",
    )

    parser.add_argument(
        "--istride",
        action="store",
        type=int,
        dest="istride",
        default=default_values["istride"],
        metavar="WIDTH",
        help=("input WIDTH stride"),
    )
    parser.add_argument(
        "--iscanline",
        action="store",
        type=int,
        dest="iscanline",
        default=default_values["iscanline"],
        metavar="HEIGHT",
        help=("input HEIGHT scanline"),
    )
    parser.add_argument(
        "--icolorrange",
        action="store",
        dest="icolorrange",
        default=default_values["icolorrange"],
        choices=itools_common.ColorRange.get_choices(),
        metavar="[%s]"
        % (
            " | ".join(
                itools_common.ColorRange.get_choices(),
            )
        ),
        help=("input COLORRANGE"),
    )
    parser.add_argument(
        "--width",
        action="store",
        type=int,
        dest="width",
        default=default_values["width"],
        help="Output Width",
    )
    parser.add_argument(
        "--height",
        action="store",
        type=int,
        dest="height",
        default=default_values["height"],
        help="Output height",
    )
    parser.add_argument(
        "--rotate-angle",
        action="store",
        type=int,
        dest="rotate_angle",
        default=default_values["rotate_angle"],
        choices=ROTATE_ANGLE_LIST.keys(),
        metavar="[%s]"
        % (
            " | ".join(
                [str(angle) for angle in ROTATE_ANGLE_LIST.keys()],
            )
        ),
        help="rotate angle arg",
    )
    parser.add_argument(
        "--mix-definition",
        action="store",
        type=str,
        dest="mix_definition",
        default=default_values["mix_definition"],
        help="Mix definition",
    )
    parser.add_argument(
        "--a00",
        action="store",
        type=float,
        dest="a00",
        default=default_values["a00"],
        metavar="a00",
        help=("a00 (default: %i)" % default_values["a00"]),
    )
    parser.add_argument(
        "--a01",
        action="store",
        type=float,
        dest="a01",
        default=default_values["a01"],
        metavar="a01",
        help=("a01 (default: %i)" % default_values["a01"]),
    )
    parser.add_argument(
        "--a10",
        action="store",
        type=float,
        dest="a10",
        default=default_values["a10"],
        metavar="a10",
        help=("a10 (default: %i)" % default_values["a10"]),
    )
    parser.add_argument(
        "--a11",
        action="store",
        type=float,
        dest="a11",
        default=default_values["a11"],
        metavar="a11",
        help=("a11 (default: %i)" % default_values["a11"]),
    )
    parser.add_argument(
        "--b00",
        action="store",
        type=float,
        dest="b00",
        default=default_values["b00"],
        metavar="b00",
        help=("b00 (default: %i)" % default_values["b00"]),
    )
    parser.add_argument(
        "--b10",
        action="store",
        type=float,
        dest="b10",
        default=default_values["b10"],
        metavar="b10",
        help=("b10 (default: %i)" % default_values["b10"]),
    )
    # affine transformation parameters
    parser.add_argument(
        "--s0x",
        action="store",
        type=float,
        dest="s0x",
        default=default_values["s0x"],
        metavar="s0.x",
        help=("s0x (default: %i)" % default_values["s0x"]),
    )
    parser.add_argument(
        "--s0y",
        action="store",
        type=float,
        dest="s0y",
        default=default_values["s0y"],
        metavar="s0.y",
        help=("s0y (default: %i)" % default_values["s0y"]),
    )
    parser.add_argument(
        "--s1x",
        action="store",
        type=float,
        dest="s1x",
        default=default_values["s1x"],
        metavar="s1.x",
        help=("s1x (default: %i)" % default_values["s1x"]),
    )
    parser.add_argument(
        "--s1y",
        action="store",
        type=float,
        dest="s1y",
        default=default_values["s1y"],
        metavar="s1.y",
        help=("s1y (default: %i)" % default_values["s1y"]),
    )
    parser.add_argument(
        "--s2x",
        action="store",
        type=float,
        dest="s2x",
        default=default_values["s2x"],
        metavar="s2.x",
        help=("s2x (default: %i)" % default_values["s2x"]),
    )
    parser.add_argument(
        "--s2y",
        action="store",
        type=float,
        dest="s2y",
        default=default_values["s2y"],
        metavar="s2.y",
        help=("s2y (default: %i)" % default_values["s2y"]),
    )
    parser.add_argument(
        "--d0x",
        action="store",
        type=float,
        dest="d0x",
        default=default_values["d0x"],
        metavar="d0.x",
        help=("d0x (default: %i)" % default_values["d0x"]),
    )
    parser.add_argument(
        "--d0y",
        action="store",
        type=float,
        dest="d0y",
        default=default_values["d0y"],
        metavar="d0.y",
        help=("d0y (default: %i)" % default_values["d0y"]),
    )
    parser.add_argument(
        "--d1x",
        action="store",
        type=float,
        dest="d1x",
        default=default_values["d1x"],
        metavar="d1.x",
        help=("d1x (default: %i)" % default_values["d1x"]),
    )
    parser.add_argument(
        "--d1y",
        action="store",
        type=float,
        dest="d1y",
        default=default_values["d1y"],
        metavar="d1.y",
        help=("d1y (default: %i)" % default_values["d1y"]),
    )
    parser.add_argument(
        "--d2x",
        action="store",
        type=float,
        dest="d2x",
        default=default_values["d2x"],
        metavar="d2.x",
        help=("d2x (default: %i)" % default_values["d2x"]),
    )
    parser.add_argument(
        "--d2y",
        action="store",
        type=float,
        dest="d2y",
        default=default_values["d2y"],
        metavar="d2.y",
        help=("d2y (default: %i)" % default_values["d2y"]),
    )
    parser.add_argument(
        "--filter",
        action="store",
        type=str,
        dest="filter",
        default=default_values["filter"],
        choices=FILTER_CHOICES.keys(),
        metavar="{%s}" % (" | ".join("{}".format(k) for k in FILTER_CHOICES.keys())),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FILTER_CHOICES.items())),
    )
    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-j",
        "--infile2",
        action="store",
        type=str,
        dest="infile2",
        default=default_values["infile2"],
        metavar="input-file-2",
        help="input file 2",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    parser.add_argument(
        "--logfile",
        action="store",
        dest="logfile",
        type=str,
        default=default_values["logfile"],
        metavar="log-file",
        help="log file",
    )
    itools_common.Config.set_parser_options(parser)
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # implement help
    if options.filter == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get logfile descriptor
    if options.logfile is None:
        logfd = sys.stdout
    else:
        logfd = open(options.logfile, "w")
    # get infile/outfile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # create configuration
    config_dict = itools_common.Config.Create(options)
    # print results
    if options.debug > 0:
        print(f"debug: {options}")

    iinfo = itools_common.ImageInfo(
        options.iwidth,
        options.iheight,
        options.istride,
        options.iscanline,
        itools_common.ColorRange.parse(options.icolorrange),
    )

    if options.filter == "copy":
        copy_image(
            options.infile,
            options.outfile,
            iinfo,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "diff":
        df = diff_images(
            options.infile,
            options.infile2,
            options.outfile,
            iinfo,
            options.diff_invert,
            options.diff_factor,
            options.diff_component,
            options.diff_color,
            options.diff_color_factor,
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )
        df.to_csv("/dev/fd/1", index=False)

    elif options.filter == "mse" or options.filter == "psnr":
        mse, psnr = mse_image(
            options.infile,
            iinfo,
            options.mse_invert,
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )
        print(f"{mse = }\n{psnr = }")

    elif options.filter == "histogram":
        get_histogram(
            options.infile,
            options.outfile,
            iinfo,
            options.hist_component,
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "rotate":
        rotate_image(
            options.infile,
            options.rotate_angle,
            options.outfile,
            iinfo,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "compose":
        compose_images(
            options.infile,
            options.infile2,
            iinfo,
            options.x,
            options.y,
            options.outfile,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "match":
        match_images(
            options.infile,
            options.infile2,
            options.outfile,
            iinfo,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "gray":
        image_to_gray(
            options.infile,
            options.outfile,
            iinfo,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "xchroma":
        swap_xchroma(
            options.infile,
            options.outfile,
            iinfo,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "mix-images":
        df = mix_images(
            options.infile,
            options.infile2,
            options.mix_definition,
            options.outfile,
            iinfo,
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "xrgb2yuv":
        swap_xrgb2yuv(
            options.infile,
            options.outfile,
            iinfo,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "noise":
        add_noise(
            options.infile,
            options.outfile,
            iinfo,
            options.noise_level,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "affine":
        affine_transformation_matrix(
            options.infile,
            iinfo,
            options.outfile,
            options.width,
            options.height,
            options.a00,
            options.a01,
            options.a10,
            options.a11,
            options.b00,
            options.b10,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )

    elif options.filter == "affine-points":
        affine_transformation_points(
            options.infile,
            iinfo,
            options.outfile,
            options.width,
            options.height,
            options.s0x,
            options.s0y,
            options.s1x,
            options.s1y,
            options.s2x,
            options.s2y,
            options.d0x,
            options.d0y,
            options.d1x,
            options.d1y,
            options.d2x,
            options.d2y,
            itools_common.ProcColor[options.proc_color],
            config_dict,
            options.cleanup,
            logfd,
            options.debug,
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
