#!/usr/bin/env python3

"""itools-jxl.py module description.

Runs generic JXL (JPEG-XL) analysis. Requires access to djxl (libjxl) and ffmpeg.
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

JPEGXL_ENC = os.environ.get("JPEGXL_ENC", "cjxl")
JPEGXL_DEC = os.environ.get("JPEGXL_DEC", "djxl")


def read_jxl(infile, config_dict, logfd, debug=0):
    # 1. decode to y4m
    tmpy4m = tempfile.NamedTemporaryFile(prefix="itools.jxl.", suffix=".y4m").name
    decode_jxl(infile, tmpy4m, logfd, debug)
    # 2. read the y4m
    outyvu, _, _, _ = itools_y4m.read_y4m(
        tmpy4m,
        output_colorrange=itools_common.ColorRange.full,
        logfd=logfd,
        debug=debug,
    )
    # 3. set read colorrange
    status = {
        "colorrange": itools_common.ColorRange.full,
    }
    return outyvu, status


def decode_jxl(infile, outfile, logfd, debug):
    # 1. decode to ppm (jxl decoder produces ppm)
    tmpppm = tempfile.NamedTemporaryFile(prefix="itools.jxl.", suffix=".ppm").name
    # do the decoding
    command = f"{JPEGXL_DEC} {infile} {tmpppm}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
    # 2. convert to outfile
    # We are forcing the output of jxl to be interpreted as full range
    command = f"{itools_common.FFMPEG_SILENT} -i {tmpppm} -pix_fmt yuv420p -vf scale=in_range=limited:out_range=full {outfile}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"


def encode_jxl(infile, codec, preset, quality, outfile, logfd, debug):
    # 1. convert to ppm (jxl encoder wants ppm)
    tmpppm = tempfile.NamedTemporaryFile(prefix="itools.jxl.", suffix=".ppm").name
    command = f"{itools_common.FFMPEG_SILENT} -i {infile} {tmpppm}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. do the encoding
    if codec == "jxl" and preset is not None:
        preset_str = f" -e {preset}"
    else:
        preset_str = ""
    command = f"{JPEGXL_ENC} {tmpppm} {outfile} -q {quality} {preset_str}"
    returncode, out, err, stats = itools_common.run(
        command, logfd=logfd, debug=debug, gnu_time=True
    )
    assert returncode == 0, f"error: {out = } {err = }"
    return stats
