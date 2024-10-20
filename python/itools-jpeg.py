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

LIBJPEG_ENC = os.environ.get("LIBJPEG_ENC", "cjpeg")
JPEGLI_ENC = os.environ.get("JPEGLI_ENC", "cjpegli")


def read_jpeg(infile, config_dict, logfd, debug=0):
    # 1. decode to y4m
    tmpy4m = tempfile.NamedTemporaryFile(prefix="itools.jpeg.", suffix=".y4m").name
    decode_jpeg(infile, tmpy4m, logfd=logfd, debug=debug)
    # 2. read the y4m
    outyvu, _, _, _ = itools_y4m.read_y4m(
        tmpy4m,
        output_colorrange=itools_common.ColorRange.full,
        logfd=logfd,
        debug=debug,
    )
    return outyvu


def decode_jpeg(infile, outfile, logfd, debug):
    # ffmpeg (default) jpeg decoder does not annotate y4m
    command = f"{itools_common.FFMPEG_SILENT} -i {infile} -color_range full {outfile}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"


def encode_libjpeg(infile, codec, preset, quality, outfile, cleanup, logfd, debug):
    # 1. convert to ppm (libjpeg encoder wants ppm)
    tmpppm = tempfile.NamedTemporaryFile(prefix="itools.libjpeg.", suffix=".ppm").name
    command = f"{itools_common.FFMPEG_SILENT} -i {infile} {tmpppm}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. do the encoding
    command = f"{LIBJPEG_ENC} -quality {int(quality)} -outfile {outfile} {tmpppm}"
    returncode, out, err, stats = itools_common.run(
        command, logfd=logfd, debug=debug, gnu_time=True
    )
    assert returncode == 0, f"error: {out = } {err = }"
    # 3. cleanup
    if cleanup > 0:
        os.remove(tmpppm)
    return stats


def encode_jpegli(infile, codec, preset, quality, outfile, cleanup, logfd, debug):
    # 0. jpegli crashes with quality 0 or 100
    if quality == 0 or quality == 100:
        raise itools_common.EncoderException(
            f"jpegli: Invalid quality parameter {quality=}"
        )
    # 1. convert to ppm (jpegli encoder wants ppm)
    tmpppm = tempfile.NamedTemporaryFile(prefix="itools.jpegli.", suffix=".ppm").name
    command = f"{itools_common.FFMPEG_SILENT} -i {infile} {tmpppm}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. do the encoding
    command = f"{JPEGLI_ENC} {tmpppm} {outfile} -q {int(quality)}"
    returncode, out, err, stats = itools_common.run(
        command, logfd=logfd, debug=debug, gnu_time=True
    )
    assert returncode == 0, f"error: {out = } {err = }"
    # 3. cleanup
    if cleanup > 0:
        os.remove(tmpppm)
    return stats
