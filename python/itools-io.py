#!/usr/bin/env python3

"""itools-io.py module description.

Generic I/O functions.
"""


import cv2
import os.path
import importlib
import sys

itools_common = importlib.import_module("itools-common")
itools_exiftool = importlib.import_module("itools-exiftool")
itools_heif = importlib.import_module("itools-heif")
itools_jpeg = importlib.import_module("itools-jpeg")
itools_jxl = importlib.import_module("itools-jxl")
itools_rgb = importlib.import_module("itools-rgb")
itools_y4m = importlib.import_module("itools-y4m")
itools_yuv = importlib.import_module("itools-yuv")


def read_image_file(
    infile,
    config_dict,
    flags=None,
    proc_color=itools_common.ProcColor.bgr,
    iinfo=None,
    cleanup=1,
    logfd=sys.stdout,
    debug=0,
):
    outyvu = None
    outbgr = None
    infile_extension = os.path.splitext(infile)[1]
    if infile_extension == ".y4m":
        outyvu, _, _, status = itools_y4m.read_y4m(
            infile,
            output_colorrange=itools_common.ColorRange.full,
            cleanup=cleanup,
            logfd=logfd,
            debug=debug,
        )
        if status is not None and status.get("broken", False):
            print(f"error: file {infile} is broken")

    elif infile_extension in (".yuv", ".YUV420NV12"):
        outyvu, status = itools_yuv.read_yuv(infile, iinfo, cleanup, logfd, debug)

    elif infile_extension == ".rgba":
        outbgr, status = itools_rgb.read_rgba(infile, iinfo, cleanup, logfd, debug)

    elif infile_extension in (".heic", ".avif", ".hif"):
        outyvu, status = itools_heif.read_heif(
            infile, config_dict, cleanup, logfd, debug
        )

    elif infile_extension == ".jxl":
        outyvu, status = itools_jxl.read_jxl(infile, config_dict, cleanup, logfd, debug)

    elif infile_extension in (".jpg", ".jpeg"):
        status = {}
        if proc_color == itools_common.ProcColor.bgr:
            outbgr = cv2.imread(cv2.samples.findFile(infile, flags))
        elif proc_color == itools_common.ProcColor.yvu:
            outyvu = itools_jpeg.read_jpeg(infile, config_dict, cleanup, logfd, debug)

    else:
        status = {}
        outbgr = cv2.imread(cv2.samples.findFile(infile, flags))

    # append the exiftool metadata
    exif_status = itools_exiftool.get_exiftool(
        infile,
        short=True,
        config_dict=config_dict,
        cleanup=cleanup,
        logfd=logfd,
        debug=debug,
    )
    status.update(exif_status)

    if infile_extension in (".jpg", ".jpeg"):
        status["colorrange"] = itools_common.ColorRange.full

    read_image_components = config_dict.get("read_image_components")
    if not read_image_components:
        if proc_color == itools_common.ProcColor.both:
            return None, None, status
        else:
            return None, status

    if proc_color == itools_common.ProcColor.yvu:
        if outyvu is None:
            outyvu = cv2.cvtColor(outbgr, cv2.COLOR_BGR2YCrCb)
            return outyvu, status
        else:
            return outyvu, status

    elif proc_color == itools_common.ProcColor.bgr:
        if outbgr is None:
            # TODO(chema): conversions only work in dt.uint8
            outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
            return outbgr, status
        else:
            return outbgr, status

    else:  # if proc_color == itools_common.ProcColor.both:
        if outyvu is None:
            outyvu = cv2.cvtColor(outbgr, cv2.COLOR_BGR2YCrCb)
        elif outbgr is None:
            outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
        return outbgr, outyvu, status


def read_metadata(infile, cleanup, logfd, debug):
    config_dict = {
        "read_image_components": True,
        "qpextract_bin": False,
        "h265nal_parser": True,
        "isobmff_parser": True,
    }
    _, status = read_image_file(
        infile, config_dict=config_dict, cleanup=cleanup, logfd=logfd, debug=debug
    )
    return status


def read_colorrange(infile, cleanup, logfd, debug):
    status = read_metadata(infile, cleanup, logfd, debug)
    return itools_common.ColorRange.parse(status["colorrange"])


def write_image_file(outfile, outimg, proc_color=itools_common.ProcColor.bgr, **kwargs):
    outfile_extension = os.path.splitext(outfile)[1]
    if outfile_extension == ".y4m":
        # y4m writer requires YVU
        if proc_color == itools_common.ProcColor.yvu:
            outyvu = outimg
        elif proc_color == itools_common.ProcColor.bgr:
            outyvu = cv2.cvtColor(outimg, cv2.COLOR_BGR2YCrCb)
        colorspace = kwargs.get("colorspace", "420")
        colorrange = kwargs.get("colorrange", itools_common.ColorRange.get_default())
        itools_y4m.write_y4m(outfile, outyvu, colorspace, colorrange)
    else:
        # cv2 writer requires BGR
        if proc_color == itools_common.ProcColor.yvu:
            outbgr = cv2.cvtColor(outimg, cv2.COLOR_YCrCb2BGR)
        elif proc_color == itools_common.ProcColor.bgr:
            outbgr = outimg
        cv2.imwrite(outfile, outbgr)
