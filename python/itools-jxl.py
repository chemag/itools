#!/usr/bin/env python3

"""itools-jxl.py module description.

Runs generic JXL analysis. Requires access to djxl (libjxl) and ffmpeg.
"""


import base64
import importlib
import io
import json
import os
import pandas as pd
import re
import struct
import sys
import tempfile

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")

JPEGXL_DEC = os.environ.get("JPEGXL_DEC", "djxl")


def read_jxl(infile, config_dict, debug=0):
    # jpegxl produces ppm
    tmpppm = tempfile.NamedTemporaryFile(prefix="itools.jpegxl.", suffix=".ppm").name
    # 1. decode the file
    command = f"{JPEGXL_DEC} {infile} {tmpppm}"
    returncode, out, err = itools_common.run(command, debug=debug)
    # 2. convert to y4m
    tmpy4m = tempfile.NamedTemporaryFile(prefix="itools.jpegxl.", suffix=".y4m").name
    # We are forcing the output of jpegxl to be interpreted as full range
    command = f"{itools_common.FFMPEG_SILENT} -i {tmpppm} -pix_fmt yuv420p -color_range full {tmpy4m}"
    returncode, out, err = itools_common.run(command, debug=debug)
    # 3. read the y4m
    outyvu, _, _, _ = itools_y4m.read_y4m(tmpy4m, colorrange="full", debug=debug)
    return outyvu
