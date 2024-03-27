#!/usr/bin/env python3

"""itools-heif.py module description.

Runs generic HEIF analysis. Requires access to heif-convert (libheif), MP4Box (gpac), and ffmpeg.
"""


import importlib
import json
import pandas as pd
import re
import tempfile

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")


def parse_mp4box_info(output):
    primary_id = None
    df = pd.DataFrame(columns=("item_id", "id", "type", "size", "rem", "primary"))
    for line in output.decode("ascii").split("\n"):
        if line.startswith("Primary Item - ID "):
            primary_id = int(line[len("Primary Item - ID ") :])
        elif line.startswith("Item #"):
            # 'Item #1: ID 10000 type hvc1 size 512x512 Hidden'
            pattern = (
                r"Item #(?P<item_id>\d+): ID (?P<id>\d+) type (?P<type>\w*)(?P<rem>.*)"
            )
            res = re.search(pattern, line)
            if not res:
                print(f'error: regexp does not work with line "{line}"')
            item_id = int(res.group("item_id"))
            the_id = int(res.group("id"))
            the_type = res.group("type")
            rem = res.group("rem")
            if rem:
                pattern = r" *size (?P<size>\S*)(?P<rem>.*)"
                res = re.search(pattern, line)
                size = res.group("size")
                rem = res.group("rem")
            else:
                size = None
            df.loc[df.size] = [
                item_id,
                the_id,
                the_type,
                size,
                rem,
                the_id == primary_id,
            ]
    return df


def get_item_list(infile, debug=0):
    # get the item list
    command = f"MP4Box -info {infile}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    return parse_mp4box_info(err)


COLOR_PARAMETER_LIST = {
    "colour_primaries": "cp",
    "transfer_characteristics": "tc",
    "matrix_coefficients": "mc",
    "video_full_range_flag": "fr",
}


def parse_ffmpeg_bsf_colorimetry(output):
    colorimetry = {}
    for line in output.decode("ascii").split("\n"):
        for color_parameter in COLOR_PARAMETER_LIST.keys():
            if color_parameter in line:
                rem = line[line.index(color_parameter) + len(color_parameter) :].strip()
                # prefix all keys
                colorimetry["hevc:" + COLOR_PARAMETER_LIST[color_parameter]] = int(
                    rem.split("=")[1]
                )
    return colorimetry


def get_heif_colorimetry(infile, get_exif_colorimetry, debug):
    df_item = get_item_list(infile, debug)
    # 1. get the HEVC (h265) SPS colorimetry
    # select the first hvc1 type
    hvc1_id = df_item[df_item.type == "hvc1"]["id"].iloc[0]
    # extract the 265 file of the first tile
    tmp265 = tempfile.NamedTemporaryFile(suffix=".265").name
    command = f"MP4Box -dump-item {hvc1_id}:path={tmp265} {infile}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    # extract the 265 file of the first tile
    command = f"ffmpeg -i {tmp265} -c:v copy -bsf:v trace_headers -f null -"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    colorimetry = parse_ffmpeg_bsf_colorimetry(err)
    # 2. get the Exif colorimetry
    if get_exif_colorimetry and len(df_item[df_item.type == "Exif"]["id"]) > 0:
        exif_id = df_item[df_item.type == "Exif"]["id"].iloc[0]
        # extract the exif file of the first tile
        tmpexif = tempfile.NamedTemporaryFile(suffix=".exif").name
        command = f"MP4Box -dump-item {exif_id}:path={tmpexif} {infile}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        # parse the exif file of the first tile
        command = f"exiftool -g -j -b {tmpexif}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        exif_info = json.loads(out)
        exif_dict = exif_info[0]["EXIF"]
        # prefix all keys
        exif_dict = {("exif:" + k): v for k, v in exif_dict.items()}
        colorimetry.update(exif_dict)
    return colorimetry


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
    # get the heif colorimetry
    colorimetry = get_heif_colorimetry(infile, get_exif_colorimetry=False, debug=debug)
    status.update(colorimetry)
    return outyvu, status
