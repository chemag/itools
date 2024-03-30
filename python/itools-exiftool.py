#!/usr/bin/env python3

"""itools-exiftool.py module description.

Runs exiftool on the file, and parses the output.
"""


import importlib
import json

itools_common = importlib.import_module("itools-common")


EXIFTOOL_KEYS = {
    "SourceFile": "ignore",
    "File": "ignore",
    "ExifTool": "ignore",
    "EXIF": "exif",
    "Photoshop": "ignore",
    "XMP": "ignore",
    "ICC_Profile": "icc",
    "APP14": "ignore",
    "Composite": "ignore",
}


def get_exiftool(infile, read_exif_info, read_icc_info, short, debug):
    # parse the exif file of the first tile
    command = f"exiftool -g -j -b {infile}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    exiftool_info = json.loads(out)
    exiftool_dict = exiftool_info[0]
    status = {}
    for key in exiftool_dict:
        assert key in EXIFTOOL_KEYS, f"error: exiftool key {key} not in known key list"
        val = EXIFTOOL_KEYS[key]
        if val == "ignore":
            # ignore this dictionary
            continue
        # prefix all keys
        if key == "EXIF" and not read_exif_info:
            # ignore this dictionary
            continue
        if key == "ICC_Profile" and not read_icc_info:
            # ignore this dictionary
            continue
        # prefix all keys
        subdict = {(val + ":" + k): v for k, v in exiftool_dict[key].items()}
        status.update(subdict)
    if short:
        status = reduce_info(status)
    return status


KEY_BLACKLIST = [
    "exif:ThumbnailOffset",
    "exif:ThumbnailLength",
    "exif:ThumbnailImage",
    "exif:MakerNoteUnknownText",
]


def reduce_info(status):
    status = {k: v for k, v in status.items() if k not in KEY_BLACKLIST}
    return status
