#!/usr/bin/env python3

"""itools-io.py module description.

Generic I/O functions.
"""


import cv2
import os.path
import importlib

itools_common = importlib.import_module("itools-common")
itools_heif = importlib.import_module("itools-heif")
itools_rgb = importlib.import_module("itools-rgb")
itools_y4m = importlib.import_module("itools-y4m")


def read_image_file(
    infile,
    flags=None,
    return_type=itools_common.ProcColor.bgr,
    iwidth=None,
    iheight=None,
    read_exif_info=False,
    debug=0,
):
    outyvu = None
    outbgr = None
    status = None
    if os.path.splitext(infile)[1] == ".y4m":
        outyvu, _, _, status = itools_y4m.read_y4m(
            infile, colorrange="full", debug=debug
        )
        if status is not None and status.get("broken", False):
            print(f"error: file {infile} is broken")

    elif os.path.splitext(infile)[1] == ".rgba":
        outbgr = itools_rgb.read_rgba(infile, iwidth, iheight)

    elif os.path.splitext(infile)[1] in (".heic", ".avif"):
        outyvu, status = itools_heif.read_heif(infile, read_exif_info, debug)

    else:
        outbgr = cv2.imread(cv2.samples.findFile(infile, flags))

    if return_type == itools_common.ProcColor.yvu:
        if outyvu is None:
            outyvu = cv2.cvtColor(outbgr, cv2.COLOR_BGR2YCrCb)
            return outyvu, status
        else:
            return outyvu, status

    else:  # if return_type == itools_common.ProcColor.bgr:
        if outbgr is None:
            outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
            return outbgr, status
        else:
            return outbgr, status


def write_image_file(outfile, outimg, return_type=itools_common.ProcColor.bgr):
    if os.path.splitext(outfile)[1] == ".y4m":
        # y4m writer requires YVU
        if return_type == itools_common.ProcColor.yvu:
            outyvu = outimg
        elif return_type == itools_common.ProcColor.bgr:
            outyvu = cv2.cvtColor(outimg, cv2.COLOR_BGR2YCrCb)
        itools_y4m.write_y4m(outfile, outyvu)
    else:
        # cv2 writer requires BGR
        if return_type == itools_common.ProcColor.yvu:
            outbgr = cv2.cvtColor(outimg, cv2.COLOR_YCrCb2BGR)
        elif return_type == itools_common.ProcColor.bgr:
            outbgr = outimg
        cv2.imwrite(outfile, outbgr)
