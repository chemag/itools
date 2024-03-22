#!/usr/bin/env python3

"""itools-rgb.py module description.

Runs generic RGB I/O.
"""


import numpy as np


# rgba is packed, R/G/B/A components
def read_rgba(infile, iwidth, iheight):
    with open(infile, "rb") as fin:
        data = fin.read()
    outrgba = np.frombuffer(data, dtype=np.uint8)
    # extract the 3x components (ignore alpha channel)
    outr, outg, outb = outrgba[0::4], outrgba[1::4], outrgba[2::4]
    # reshape them to the width and height
    outr = outr.reshape(iwidth, iheight)
    outg = outg.reshape(iwidth, iheight)
    outb = outb.reshape(iwidth, iheight)
    # stack components
    outbgr = np.stack((outb, outg, outr), axis=2)
    return outbgr
