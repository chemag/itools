#!/usr/bin/env python3

"""itools-exiftool.py module description.

Runs exiftool on the file, and parses the output.
"""


import base64
import importlib
import json
import os
import sys

itools_common = importlib.import_module("itools-common")

# https://stackoverflow.com/a/7506029
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib", "icctool"))
icctool = importlib.import_module("icctool")


EXIFTOOL_KEYS = {
    "SourceFile": "ignore",
    "File": "ignore",
    "ExifTool": "ignore",
    "EXIF": "exif",
    "JFIF": "ignore",
    "Photoshop": "ignore",
    "XMP": "ignore",
    "ICC_Profile": "icc",
    "APP14": "ignore",
    "PNG": "ignore",
    "Composite": "ignore",
    "MakerNotes": "ignore",
    "QuickTime": "ignore",
    "MPF": "ignore",
}


def get_exiftool(infile, short, config_dict, cleanup, logfd, debug):
    # parse the exif file of the first tile
    command = f"exiftool -g -j -b {infile}"
    returncode, out, err = itools_common.run(command, logfd=logfd, debug=debug)
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
        read_exif_info = config_dict.get("read_exif_info")
        if key == "EXIF" and not read_exif_info:
            # ignore this dictionary
            continue
        read_icc_info = config_dict.get("read_icc_info")
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
    "icc:ProfileCMMType",
    "icc:ProfileDateTime",
    "icc:ProfileFileSignature",
    "icc:PrimaryPlatform",
    "icc:CMMFlags",
    "icc:DeviceManufacturer",
    "icc:DeviceModel",
    "icc:DeviceAttributes",
    "icc:RenderingIntent",
    "icc:ProfileCreator",
    "icc:ProfileID",
]


KEY_RENAME = {
    "icc:ProfileVersion": "icc:profile_version",
    "icc:ProfileClass": "icc:profile_class",
    "icc:ProfileCopyright": "icc:profile_copyright",
    "icc:ColorSpaceData": "icc:color_space",
    "icc:ProfileConnectionSpace": "icc:profile_connection_space",
    "icc:ProfileDescription": "icc:profile_description",
    "icc:ChromaticAdaptation": "icc:chromatic_adaptation",
    "icc:MediaWhitePoint": "icc:media_white_point",
    "icc:ConnectionSpaceIlluminant": "icc:xyz_illuminant",
    "icc:RedMatrixColumn": "icc:red_matrix_column",
    "icc:GreenMatrixColumn": "icc:green_matrix_column",
    "icc:BlueMatrixColumn": "icc:blue_matrix_column",
    "icc:RedTRC": "icc:red_trc",
    "icc:BlueTRC": "icc:green_trc",
    "icc:GreenTRC": "icc:blue_trc",
}


def base64_decode(string):
    t, v = string.split(":")
    assert t == "base64", f"error: unknown item '{string}'"
    blob = base64.b64decode(v)
    tag = icctool.ICCTag.parse(blob)
    return " ".join(str(f) for f in tag.todict()["parameters"])


def identity(string):
    return string


VAL_TRANSFORM = {
    "icc:green_trc": base64_decode,
    "icc:blue_trc": base64_decode,
    "icc:red_trc": base64_decode,
}


def reduce_info(status):
    status = {
        KEY_RENAME.get(k, k): VAL_TRANSFORM.get(KEY_RENAME.get(k, k), identity)(v)
        for k, v in status.items()
        if k not in KEY_BLACKLIST
    }
    return status
