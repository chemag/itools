#!/usr/bin/env python3

"""itools-heif.py module description.

Runs generic HEIF analysis. Requires access to heif-convert (libheif), MP4Box (gpac), and ffmpeg.
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
import xml.dom.minidom

itools_common = importlib.import_module("itools-common")
itools_exiftool = importlib.import_module("itools-exiftool")
itools_y4m = importlib.import_module("itools-y4m")

# https://stackoverflow.com/a/7506029
sys.path.append(os.path.join(os.path.dirname(__file__), "icctool"))
icctool = importlib.import_module("icctool.icctool")


def parse_mp4box_info(output):
    primary_id = None
    df = pd.DataFrame(columns=("item_id", "id", "type", "size", "rem", "primary"))
    for line in output.decode("ascii").split("\n"):
        if line.startswith("Primary Item - ID "):
            primary_id = int(line[len("Primary Item - ID ") :])
        elif line.startswith("Item #"):
            # 'Item #1: ID 10000 type hvc1 size 512x512 Hidden'
            # 'Item #51: ID 51 type Exif Hidden'
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
            size = None
            if rem:
                pattern = r" *size (?P<size>\S*)(?P<rem>.*)"
                res = re.search(pattern, line)
                if res:
                    size = res.group("size")
                    rem = res.group("rem")
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


NCLX_COLOR_PARAMETER_LIST = {
    "colour_primaries": "cp",
    "transfer_characteristics": "tc",
    "matrix_coefficients": "mc",
    "full_range_flag": "fr",
}


def parse_icc_profile(profile_text_bin):
    # parse the ICC profile into a dictionary
    profile = icctool.ICCProfile.parse(profile_text_bin, force_version_number=None)
    profile_dict = profile.todict(short=True)
    # prefix all keys
    colorimetry = {"icc:" + k: v for k, v in profile_dict.items()}
    return colorimetry


def parse_isomediafile_xml(tmpxml, config_dict):
    xml_doc = xml.dom.minidom.parse(tmpxml)
    try:
        colour_information_box = (
            xml_doc.getElementsByTagName("IsoMediaFile")[0]
            .getElementsByTagName("MetaBox")[0]
            .getElementsByTagName("ItemPropertiesBox")[0]
            .getElementsByTagName("ItemPropertyContainerBox")[0]
            .getElementsByTagName("ColourInformationBox")[0]
        )
    except:
        # no colr box
        return {}
    colour_type = colour_information_box.getAttribute("colour_type")
    colorimetry = {}
    colorimetry["colr:colour_type"] = colour_type
    read_icc_info = config_dict.get("read_icc_info")
    if colour_type == "nclx":
        # on-screen colours, per ISO/IEC 23091-2/h273
        # unsigned int(16) colour_primaries;
        # unsigned int(16) transfer_characteristics;
        # unsigned int(16) matrix_coefficients;
        # unsigned int(1) full_range_flag;
        # unsigned int(7) reserved = 0;
        colorimetry["colr:colour_type"] = colour_type
        for color_parameter in NCLX_COLOR_PARAMETER_LIST.keys():
            # prefix all keys
            colorimetry["colr:" + NCLX_COLOR_PARAMETER_LIST[color_parameter]] = int(
                colour_information_box.getAttribute(color_parameter)
            )
    elif colour_type == "prof" and read_icc_info:
        # unrestricted ICC profile (ISO 15076-1 or ICC.1:2010)
        profile = colour_information_box.getElementsByTagName("profile")[0]
        # parse CDATA
        profile_text = profile.toxml()
        CDStart = "<![CDATA["
        CDEnd = "]]>"
        if CDStart in profile_text and CDEnd in profile_text:
            profile_text_base64 = profile_text[
                profile_text.index(CDStart) + len(CDStart) : profile_text.index(CDEnd)
            ]
            profile_text_bin = base64.b64decode(profile_text_base64)
            icc_dict = parse_icc_profile(profile_text_bin)
            colorimetry.update(icc_dict)
    return colorimetry


# parse hvcC (hevc configuration record, HEVCDecoderConfigurationRecord) box,
# per ISO/IEC 14496-15:2022, Section 8.3.2.1.2
def parse_hvcC_box(infile, config_dict, debug):
    with open(infile, "rb") as fin:
        hvcC_bin = fin.read()
    i = 0
    # ensure this is a valid hvcC box
    hvcC_len = struct.unpack(">I", hvcC_bin[i : i + 4])[0]
    i += 4
    assert hvcC_len == len(hvcC_bin), f"error: invalid hvcC len in {infile}"
    hvcC_header = hvcC_bin[i : i + 4]
    i += 4
    assert hvcC_header == b"hvcC", f"error: invalid hvcC box in {infile}"
    # parse the hevc configuration record
    # unsigned int(8) configurationVersion = 1;
    configurationVersion = struct.unpack("B", hvcC_bin[i : i + 1])[0]
    i += 1
    assert (
        configurationVersion == 1
    ), f"error: invalid hvcC configurationVersion in {infile}"
    # unsigned int(2) general_profile_space;
    # unsigned int(1) general_tier_flag;
    # unsigned int(5) general_profile_idc;
    tmp_byte = struct.unpack("B", hvcC_bin[i : i + 1])[0]
    general_profile_space = tmp_byte >> 6
    general_tier_flag = (tmp_byte >> 5) & 0x01
    general_profile_idc = (tmp_byte) & 0x1F
    i += 1
    # unsigned int(32) general_profile_compatibility_flags;
    general_profile_compatibility_flags = struct.unpack(">I", hvcC_bin[i : i + 4])[0]
    i += 4
    # unsigned int(48) general_constraint_indicator_flags;
    tmp_short = struct.unpack(">H", hvcC_bin[i : i + 2])[0]
    i += 2
    tmp_int = struct.unpack(">I", hvcC_bin[i : i + 4])[0]
    i += 4
    general_constraint_indicator_flags = (tmp_short << 32) | tmp_int
    # unsigned int(8) general_level_idc;
    general_level_idc = struct.unpack("B", hvcC_bin[i : i + 1])[0]
    i += 1
    # bit(4) reserved = `1111'b;
    # unsigned int(12) min_spatial_segmentation_idc;
    tmp_short = struct.unpack(">H", hvcC_bin[i : i + 2])[0]
    i += 2
    reserved = tmp_short >> 12
    min_spatial_segmentation_idc = (tmp_short) & 0xFFF
    # bit(6) reserved = `111111'b;
    # unsigned int(2) parallelismType;
    tmp_byte = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
    i += 1
    reserved = tmp_byte >> 2
    parallelismType = (tmp_byte) & 0x3
    # bit(6) reserved = `111111'b;
    # unsigned int(2) chromaFormat;
    tmp_byte = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
    i += 1
    reserved = tmp_byte >> 2
    chromaFormat = (tmp_byte) & 0x3
    # bit(5) reserved = `11111'b;
    # unsigned int(3) bitDepthLumaMinus8;
    tmp_byte = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
    i += 1
    reserved = tmp_byte >> 3
    bitDepthLumaMinus8 = (tmp_byte) & 0x7
    # bit(5) reserved = `11111'b;
    # unsigned int(3) bitDepthChromaMinus8;
    tmp_byte = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
    i += 1
    reserved = tmp_byte >> 3
    bitDepthChromaMinus8 = (tmp_byte) & 0x7
    # bit(16) avgFrameRate;
    avgFrameRate = struct.unpack(">H", hvcC_bin[i : i + 2])[0]
    i += 2
    # bit(2) constantFrameRate;
    # bit(3) numTemporalLayers;
    # bit(1) temporalIdNested;
    # unsigned int(2) lengthSizeMinusOne;
    tmp_byte = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
    i += 1
    constantFrameRate = tmp_byte >> 6
    numTemporalLayers = (tmp_byte >> 3) & 0x3
    temporalIdNested = (tmp_byte >> 2) & 0x1
    lengthSizeMinusOne = (tmp_byte) & 0x3
    # unsigned int(8) numOfArrays;
    numOfArrays = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
    i += 1
    nal_units = {}
    for j in range(numOfArrays):
        # bit(1) array_completeness;
        # unsigned int(1) reserved = 0;
        # unsigned int(6) NAL_unit_type;
        tmp_byte = struct.unpack(">B", hvcC_bin[i : i + 1])[0]
        i += 1
        array_completeness = tmp_byte >> 7
        reserved = (tmp_byte >> 6) & 0x1
        NAL_unit_type = tmp_byte & 0x3F
        nal_units[NAL_unit_type] = []
        # unsigned int(16) numNalus;
        numNalus = struct.unpack(">H", hvcC_bin[i : i + 2])[0]
        i += 2
        for k in range(numNalus):
            # unsigned int(16) nalUnitLength;
            nalUnitLength = struct.unpack(">H", hvcC_bin[i : i + 2])[0]
            i += 2
            # bit(8*nalUnitLength) nalUnit;
            nalUnit = hvcC_bin[i : i + nalUnitLength]
            i += nalUnitLength
            nal_units[NAL_unit_type].append(nalUnit)
    if 33 not in nal_units.keys():
        # no SPS: punt here
        return {}
    for sps in nal_units[33]:
        tmpsps = tempfile.NamedTemporaryFile(
            prefix="itools.nalu.sps.", suffix=".265"
        ).name
        #
        with open(tmpsps, "wb") as fout:
            fout.write(sps)
        sps_dict = parse_hevc_sps(tmpsps, config_dict, debug)
    # prefix all keys
    sps_dict = {("hvcC:" + key): value for key, value in sps_dict.items()}
    return sps_dict


SPS_PARAMETERS = {
    "profile_idc": "profile",
    "general_level_idc": "level",
    "video_full_range_flag": "fr",
    "colour_primaries": "cp",
    "transfer_characteristics": "tc",
    "matrix_coeffs": "mc",
}


def parse_hevc_sps(tmpsps, config_dict, debug):
    h265nal_parser = config_dict.get("h265nal_parser")
    if h265nal_parser is None:
        return {}
    h265nal_parser = "~/proj/h265nal/build/tools/h265nal.nalu"
    command = f"{h265nal_parser} --no-as-one-line -i {tmpsps}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    # look for the colorimetry
    sps_dict = {}
    for line in out.decode("ascii").split("\n"):
        for vui_parameter in SPS_PARAMETERS.keys():
            if (vui_parameter + ":") in line:
                sps_dict[SPS_PARAMETERS[vui_parameter]] = int(line.split()[-1])
    return sps_dict


def get_heif_colorimetry(infile, config_dict, debug):
    df_item = get_item_list(infile, debug)
    file_type_list = df_item.type.unique()
    colorimetry = {}
    read_exif_info = config_dict.get("read_exif_info")
    read_icc_info = config_dict.get("read_icc_info")
    # 1. get the HEVC (h265) SPS colorimetry
    if "hvc1" in file_type_list:
        # select the first hvc1 type
        file_type = "hvc1"
        hvc1_id = df_item[df_item.type == file_type]["id"].iloc[0]
        # extract the 265 file of the first tile
        tmp265 = tempfile.NamedTemporaryFile(prefix="itools.tile.", suffix=".265").name
        command = f"MP4Box -dump-item {hvc1_id}:path={tmp265} {infile}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        # extract the 265 file of the first tile
        command = f"{itools_common.FFMPEG_SILENT} -i {tmp265} -c:v copy -bsf:v trace_headers -f null -"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        hevc_dict = parse_ffmpeg_bsf_colorimetry(err)
        hevc_dict["hevc:ntiles"] = len(df_item[df_item.type == file_type]["id"])
        colorimetry.update(hevc_dict)
    # 2. get the hvcC colorimetry
    isobmff_parser = config_dict.get("isobmff_parser")
    if file_type == "hvc1" and isobmff_parser is not None:
        # get the hvcC box
        hvcC_box = "/meta/iprp/ipco/hvcC"
        tmphvcC = tempfile.NamedTemporaryFile(prefix="itools.hvcC.", suffix=".bin").name
        command = f"{isobmff_parser} -i {infile} --func extract-box --path {hvcC_box} -o {tmphvcC}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        # parse the hvcC box
        hvcC_dict = parse_hvcC_box(tmphvcC, config_dict, debug)
        colorimetry.update(hvcC_dict)

    # 3. get the Exif colorimetry
    if read_exif_info and len(df_item[df_item.type == "Exif"]["id"]) > 0:
        exif_id = df_item[df_item.type == "Exif"]["id"].iloc[0]
        # extract the exif file of the first tile
        tmpexif = tempfile.NamedTemporaryFile(prefix="itools.exif.", suffix=".bin").name
        command = f"MP4Box -dump-item {exif_id}:path={tmpexif} {infile}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        # parse the exif file of the first tile
        exiftool_dict = itools_exiftool.get_exiftool(
            tmpexif, short=True, config_dict=config_dict, debug=debug
        )
        colorimetry.update(exiftool_dict)
    # 4. get the `colr` colorimetry
    tmpxml = tempfile.NamedTemporaryFile(prefix="itools.xml.", suffix=".xml").name
    command = f"MP4Box -std -dxml {infile} > {tmpxml}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error in {command}\n{err}"
    colr_dict = parse_isomediafile_xml(tmpxml, config_dict)
    colorimetry.update(colr_dict)
    return colorimetry


def parse_heif_convert_output(tmpy4m, output, debug):
    # count the number of images
    num_images = None
    output_files = []
    for line in output.decode("ascii").split("\n"):
        if line.startswith("File contains "):
            num_images = int(line[len("File contains ") : -len(" image")])
        elif line.startswith("Written to "):
            output_files.append(line[len("Written to ") :])
    if tmpy4m in output_files:
        return tmpy4m
    elif num_images > 1:
        return output_files[0]
    return tmpy4m


QPEXTRACT_FIELDS = ("qp_avg", "qp_stddev", "qp_num", "qp_min", "qp_max")
CTU_SIZE_VALUES = (8, 16, 32, 64)


def parse_qpextract_bin_output(output, mode):
    df = pd.read_csv(io.StringIO(output.decode("ascii")))
    if mode == "qp":
        return {key: df.iloc[0][key] for key in QPEXTRACT_FIELDS}
    elif mode == "ctu":
        # get statistics
        ctu_dict = {"mean": df["size"].mean(), "stddev": df["size"].std()}
        ctu_dict.update(
            {
                f"ratio_{ctu_size}": (len(df[df["size"] == ctu_size]) / len(df))
                for ctu_size in CTU_SIZE_VALUES
            }
        )
        return ctu_dict


def get_h265_values(infile, config_dict, debug):
    qpextract_bin = config_dict.get("qpextract_bin")
    if qpextract_bin is None:
        return {}
    qp_dict = {}
    df_item = get_item_list(infile, debug)
    file_type_list = df_item.type.unique()
    # 1. get the HEVC (h265) weighted QP distribution
    if "hvc1" in file_type_list:
        # select the first hvc1 type
        file_type = "hvc1"
        hvc1_id = df_item[df_item.type == file_type]["id"].iloc[0]
        # extract the 265 file of the first tile
        tmp265 = tempfile.NamedTemporaryFile(prefix="itools.hvc1.", suffix=".265").name
        command = f"MP4Box -dump-item {hvc1_id}:path={tmp265} {infile}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        # extract the QP-Y info for the first tile
        command = f"{qpextract_bin} --qpymode -w -i {tmp265}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        qp_dict_y = parse_qpextract_bin_output(out, "qp")
        qp_dict.update({f"qpwy:{k}": v for k, v in qp_dict_y.items()})
        # extract the QP-Cb info for the first tile
        command = f"{qpextract_bin} --qpcbmode -w -i {tmp265}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        qp_dict_cb = parse_qpextract_bin_output(out, "qp")
        qp_dict.update({f"qpwcb:{k}": v for k, v in qp_dict_cb.items()})
        # extract the QP-Cr info for the first tile
        command = f"{qpextract_bin} --qpcrmode -w -i {tmp265}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        qp_dict_cr = parse_qpextract_bin_output(out, "qp")
        qp_dict.update({f"qpwcr:{k}": v for k, v in qp_dict_cr.items()})
        # extract the CTU info for the first tile
        command = f"{qpextract_bin} --ctumode -w -i {tmp265}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        ctu_dict = parse_qpextract_bin_output(out, "ctu")
        qp_dict.update({f"ctu:{k}": v for k, v in ctu_dict.items()})
    return qp_dict


def read_heif(infile, config_dict, debug=0):
    read_image_components = config_dict.get("read_image_components")
    if read_image_components:
        tmpy4m = tempfile.NamedTemporaryFile(prefix="itools.raw.", suffix=".y4m").name
        if debug > 0:
            print(f"using {tmpy4m}")
        # decode the file using libheif
        command = f"heif-convert {infile} {tmpy4m}"
        returncode, out, err = itools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        tmpy4m = parse_heif_convert_output(tmpy4m, out, debug)
        # read the y4m frame ignoring the color range
        outyvu, _, _, status = itools_y4m.read_y4m(tmpy4m, colorrange=None, debug=debug)
    else:
        outyvu = None
        status = {}
    # get the heif colorimetry
    colorimetry = get_heif_colorimetry(infile, config_dict, debug=debug)
    status.update(colorimetry)
    # get the heif QP distribution
    qp_dict = get_h265_values(infile, config_dict, debug=debug)
    status.update(qp_dict)
    return outyvu, status
