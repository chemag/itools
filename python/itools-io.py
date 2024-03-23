#!/usr/bin/env python3

"""itools-io.py module description.

Generic I/O functions.
"""


import cv2
import os.path
import importlib

itools_common = importlib.import_module("itools-common")
itools_rgb = importlib.import_module("itools-rgb")
itools_y4m = importlib.import_module("itools-y4m")


def read_image_file(
    infile,
    flags=None,
    return_type=itools_common.ProcColor.bgr,
    iwidth=None,
    iheight=None,
    debug=0,
):
    if os.path.splitext(infile)[1] == ".y4m":
        outyvu, _, _, status = itools_y4m.read_y4m(
            infile, colorrange="full", debug=debug
        )
        if status is not None and status["broken"]:
            print(f"error: file {infile} is broken")
        if return_type == itools_common.ProcColor.yvu:
            return outyvu, status
        outbgr = cv2.cvtColor(outyvu, cv2.COLOR_YCrCb2BGR)
        return outbgr, status

    elif os.path.splitext(infile)[1] == ".rgba":
        outbgr = itools_rgb.read_rgba(infile, iwidth, iheight)

    else:
        outbgr = cv2.imread(cv2.samples.findFile(infile, flags))

    if return_type == itools_common.ProcColor.bgr:
        return outbgr, None
    elif return_type == itools_common.ProcColor.yvu:
        outyvu = cv2.cvtColor(outbgr, cv2.COLOR_BGR2YCrCb)
        return outyvu, None


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
