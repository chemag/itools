#!/usr/bin/env python3

"""itools-yuv.py module description.

Runs generic YUV I/O.
"""


import importlib
import numpy as np

itools_common = importlib.import_module("itools-common")


# YUV is packed, Y/U/V components
def read_yuv(infile, iinfo, cleanup, logfd, debug):
    with open(infile, "rb") as fin:
        data = fin.read()
    # assume nv12 for now
    # TODO(chemag): generalize this
    inyuv = np.frombuffer(data, dtype=np.uint8)
    # extract the 3x components
    if iinfo.stride is None:
        iinfo.stride = iinfo.width
    if iinfo.scanline is None:
        iinfo.scanline = iinfo.height
    # extract the luma
    luma_size = iinfo.stride * iinfo.scanline
    iny = inyuv[0:luma_size]
    try:
        outy = iny.reshape(iinfo.scanline, iinfo.stride)
    except ValueError as e:
        print(
            f"warn: pad before reshaping Y component: {len(iny)} != {(iinfo.scanline) * (iinfo.stride)} {iinfo.scanline}x{iinfo.stride}",
            file=logfd,
        )
        outy = np.resize(iny, (iinfo.scanline, iinfo.stride))
    # extract the chromas
    chroma_size = (iinfo.stride * iinfo.scanline) // 2
    inuv = inyuv[luma_size : luma_size + chroma_size]
    inu, inv = inuv[0::2], inuv[1::2]
    try:
        outu = inu.reshape(iinfo.scanline // 2, iinfo.stride // 2)
    except ValueError as e:
        print(
            f"warn: pad before reshaping U component: {len(inu)} != {(iinfo.scanline // 2) * (iinfo.stride // 2)} {iinfo.scanline // 2}x{iinfo.stride // 2}",
            file=logfd,
        )
        outu = np.resize(inu, (iinfo.scanline // 2, iinfo.stride // 2))
    try:
        outv = inv.reshape(iinfo.scanline // 2, iinfo.stride // 2)
    except ValueError as e:
        print(
            f"warn: pad before reshaping V component: {len(inv)} != {(iinfo.scanline // 2) * (iinfo.stride // 2)} {iinfo.scanline // 2}x{iinfo.stride // 2}",
            file=logfd,
        )
        outv = np.resize(inv, (iinfo.scanline // 2, iinfo.stride // 2))
    # undo the chroma subsample
    height, width = iinfo.height, iinfo.width
    dtype = iny.dtype
    outu_full = itools_common.chroma_subsample_reverse(
        outu, height, width, dtype, itools_common.ChromaSubsample.chroma_420
    )
    outv_full = itools_common.chroma_subsample_reverse(
        outv, height, width, dtype, itools_common.ChromaSubsample.chroma_420
    )
    # remove the stride and scanline
    oy = outy[0:height, 0:width]
    ou = outu_full[0:height, 0:width]
    ov = outv_full[0:height, 0:width]
    # stack components
    outyvu = itools_common.yuv_planar_to_yuv_cv2(oy, ou, ov)
    status = {
        "colorrange": iinfo.colorrange,
    }
    return outyvu, status
