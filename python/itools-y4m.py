#!/usr/bin/env python3

"""itools-y4m.py module description.

Runs generic y4m I/O. Similar to python-y4m.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import numpy as np
import os.path
import y4m


# converts a chroma-subsampled matrix into a non-chroma subsampled one
# Algo is very simple (just dup values)
def chroma_subsample_reverse(inmatrix, colorspace):
    in_w, in_h = inmatrix.shape
    if colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
        out_w = in_w << 1
        out_h = in_h << 1
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix[::2, ::2] = inmatrix
        outmatrix[1::2, ::2] = inmatrix
        outmatrix[::2, 1::2] = inmatrix
        outmatrix[1::2, 1::2] = inmatrix
    elif colorspace in ("422",):
        out_w = in_w << 1
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix[::, ::2] = inmatrix
        outmatrix[::, 1::2] = inmatrix
    elif colorspace in ("444",):
        out_w = in_w
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix = inmatrix
    return outmatrix


# converts a non-chroma-subsampled matrix into a chroma subsampled one
# Algo is very simple (just average values)
def chroma_subsample_direct(inmatrix, colorspace):
    in_w, in_h = inmatrix.shape
    if colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
        out_w = in_w >> 1
        out_h = in_h >> 1
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint16)
        outmatrix += inmatrix[::2, ::2]
        outmatrix += inmatrix[1::2, ::2]
        outmatrix += inmatrix[::2, 1::2]
        outmatrix += inmatrix[1::2, 1::2]
        outmatrix = outmatrix / 4
        outmatrix = outmatrix.astype(np.uint8)
    elif colorspace in ("422",):
        out_w = in_w >> 1
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint16)
        outmatrix += inmatrix[::, ::2]
        outmatrix += inmatrix[::, 1::2]
        outmatrix = outmatrix / 2
        outmatrix = outmatrix.astype(np.uint8)
    elif colorspace in ("444",):
        out_w = in_w
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix = inmatrix
    return outmatrix


# TODO(chemag): avoid the global variable to read frames
def y4m_process_frame(frame):
    global y4m_read_frame
    y4m_read_frame = frame


def read_y4m(infile):
    # read the y4m frame
    with open(infile, "rb") as fin:
        data = fin.read()
    parser = y4m.Reader(y4m_process_frame, verbose=False)
    parser.decode(data)
    # convert it to a numpy array
    print(y4m_read_frame.headers)
    width = y4m_read_frame.headers["W"]
    height = y4m_read_frame.headers["H"]
    # framerate = y4m_read_frame.headers["F"]
    # interlaced = y4m_read_frame.headers["I"]
    # aspect = y4m_read_frame.headers["A"]
    colorspace = y4m_read_frame.headers["C"]
    # extra = y4m_read_frame.headers["X"]
    dt = np.dtype(np.uint8)
    luma_size = width * height
    ya = np.frombuffer(y4m_read_frame.buffer[:luma_size], dtype=dt).reshape(
        width, height
    )
    # read the chromas
    if colorspace in ("420jpeg", "420paldv", "420", "420mpeg2"):
        chroma_w = width >> 1
        chroma_h = height >> 1
    elif colorspace in ("422",):
        chroma_w = width >> 1
        chroma_h = height
    elif colorspace in ("444",):
        chroma_w = width
        chroma_h = height
    chroma_size = chroma_w * chroma_h
    ua = np.frombuffer(
        y4m_read_frame.buffer[luma_size : luma_size + chroma_size], dtype=dt
    ).reshape(chroma_w, chroma_h)
    va = np.frombuffer(
        y4m_read_frame.buffer[luma_size + chroma_size :], dtype=dt
    ).reshape(chroma_w, chroma_h)
    # combine the color components
    # undo chroma subsample in order to combine same-size matrices
    ua_full = chroma_subsample_reverse(ua, colorspace)
    va_full = chroma_subsample_reverse(va, colorspace)
    # note that OpenCV conversions use YCrCb (YVU) instead of YCbCr (YUV)
    outyvu = np.stack((ya, va_full, ua_full), axis=2)
    return outyvu


def write_y4m(outfile, outyvu, colorspace="420"):
    with open(outfile, "wb") as fout:
        # write luma
        width, height, _ = outyvu.shape
        header = f"YUV4MPEG2 W{width} H{height} F30000:1001 Ip C{colorspace}\n"
        fout.write(header.encode("utf-8"))
        # write frame line
        frame = "FRAME\n"
        fout.write(frame.encode("utf-8"))
        # write y
        ya = outyvu[:, :, 0]
        fout.write(ya.flatten())
        # write u (implementing chroma subsample)
        ua_full = outyvu[:, :, 2]
        ua = chroma_subsample_direct(ua_full, colorspace)
        fout.write(ua.flatten())
        # write v (implementing chroma subsample)
        va_full = outyvu[:, :, 1]
        va = chroma_subsample_direct(va_full, colorspace)
        fout.write(va.flatten())
