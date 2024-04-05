#!/usr/bin/env python3

"""itools-yuv.py module description.

Runs generic YUV I/O.
"""


import importlib
import numpy as np

itools_common = importlib.import_module("itools-common")


# YUV is packed, Y/U/V components
def read_yuv(infile, iinfo):
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
    outy = iny.reshape(iinfo.scanline, iinfo.stride)
    # extract the chromas
    chroma_size = (iinfo.stride * iinfo.scanline) // 2
    inuv = inyuv[luma_size : luma_size + chroma_size]
    inu, inv = inuv[0::2], inuv[1::2]
    outu = inu.reshape(iinfo.scanline // 2, iinfo.stride // 2)
    outv = inv.reshape(iinfo.scanline // 2, iinfo.stride // 2)
    # undo the chroma subsample
    outu_full = itools_common.chroma_subsample_reverse(outu, "420")
    outv_full = itools_common.chroma_subsample_reverse(outv, "420")
    # remove the stride and scanline
    oy = outy[0 : iinfo.height, 0 : iinfo.width]
    ou = outu_full[0 : iinfo.height, 0 : iinfo.width]
    ov = outv_full[0 : iinfo.height, 0 : iinfo.width]
    # stack components
    outyvu = np.stack((oy, ov, ou), axis=2)
    return outyvu
