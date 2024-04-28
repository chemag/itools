#!/usr/bin/env python3

"""itools-jpeg.py module description.

Runs generic JPEG analysis. Requires access to ffmpeg.
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


def read_jpeg(infile, config_dict, debug=0):
    # 1. decode to y4m
    tmpy4m = tempfile.NamedTemporaryFile(prefix="itools.jpeg.", suffix=".y4m").name
    decode_jpeg(infile, tmpy4m, debug)
    # 2. read the y4m
    outyvu, _, _, _ = itools_y4m.read_y4m(
        tmpy4m, output_colorrange=itools_common.ColorRange.full, debug=debug
    )
    return outyvu


def decode_jpeg(infile, outfile, debug):
    # ffmpeg (default) jpeg decoder does not annotate y4m
    command = f"{itools_common.FFMPEG_SILENT} -i {infile} -color_range full {outfile}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
