#!/usr/bin/env python3

"""itools-heif.py module description.

Runs generic HEIF analysis. Requires access to heif-convert (libheif) and ffmpeg.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import importlib
import tempfile

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")


def read_heif(infile, debug=0):
    tmpy4m1 = tempfile.NamedTemporaryFile(suffix=".y4m").name
    tmpy4m2 = tempfile.NamedTemporaryFile(suffix=".y4m").name
    if debug > 0:
        print(f"using {tmpy4m1} and {tmpy4m2}")
    # decode the file using libheif
    command = f"heif-convert {infile} {tmpy4m1}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    # fix the color range
    command = f"ffmpeg -y -i {tmpy4m1} -color_range full {tmpy4m2}"
    itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    # read the y4m frame
    outyvu, _, _, status = itools_y4m.read_y4m(tmpy4m2, colorrange="full", debug=debug)
    return outyvu, status
