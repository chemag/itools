#!/usr/bin/env python


import argparse
import collections
import struct
import sys

FUNC_CHOICES = {
    "parse": "parse JPEG markers",
    "extract": "extract JPEG marker by name",
}


default_values = {
    "debug": 0,
    "func": "parse",
    "marker": "",
    "infile": None,
    "outfile": None,
    "logfile": None,
}


def parse_unimplemented(marker_id, blob):
    return "unimplemented (0x%04x)" % marker_id, 0


def parse_app0(blob):
    contents = collections.OrderedDict()
    identifier_jfif = b"JFIF\x00"
    if blob[0 : len(identifier_jfif)] == identifier_jfif:
        contents["type"] = "JFIF\\x00"
        idx = len(identifier_jfif)
        # JFIF APP0 marker segment
        assert len(blob) >= 14
        version_major = blob[idx]
        contents["version_major"] = version_major
        idx += 1
        version_minor = blob[idx]
        contents["version_minor"] = version_minor
        idx += 1
        density_units = blob[idx]
        contents["density_units"] = density_units
        idx += 1
        xdensity = struct.unpack(">H", blob[idx : idx + 2])[0]
        contents["xdensity"] = xdensity
        idx += 2
        ydensity = struct.unpack(">H", blob[idx : idx + 2])[0]
        contents["ydensity"] = ydensity
        idx += 2
        xthumbnail = blob[idx]
        contents["xthumbnail"] = xthumbnail
        idx += 1
        ythumbnail = blob[idx]
        contents["ythumbnail"] = ythumbnail
        idx += 1
        # thumbnail_data = ...
        return contents

    else:
        print("error: parse_app0: invalid identifier (%s)" % blob[0:5], file=sys.stderr)
        sys.exit(-1)


# http://www.exif.org/Exif2-2.PDF
def parse_app1(blob):
    contents = collections.OrderedDict()
    identifier_exif = (b"Exif\x00\x00", b"Exif\x00\xff")
    identifier_xmp = b"http://ns.adobe.com/xap/1.0/\x00"
    identifier_extended_xmp = b"http://ns.adobe.com/xmp/extension/\x00"

    if (
        len(blob) >= len(identifier_exif[0])
        and blob[0 : len(identifier_exif[0])] in identifier_exif
    ):
        contents["type"] = "EXIF\\x00\\x00"
        idx = len(identifier_exif[0])
        assert len(blob) >= 14
        _padding = blob[idx]
        idx += 1
        ifd_tag = struct.unpack(">H", blob[idx : idx + 2])[0]
        ifd_type = struct.unpack(">H", blob[idx + 2 : idx + 4])[0]
        ifd_count = struct.unpack(">I", blob[idx + 4 : idx + 8])[0]
        ifd_value_offset = struct.unpack(">I", blob[idx + 8 : idx + 12])[0]
        # thumbnail_data = ...
        return contents

    elif (
        len(blob) >= len(identifier_xmp)
        and blob[0 : len(identifier_xmp)] == identifier_xmp
    ):
        contents["type"] = "Adobe XMP"
        idx = len(identifier_xmp)
        return contents

    elif (
        len(blob) >= len(identifier_extended_xmp)
        and blob[0 : len(identifier_extended_xmp)] == identifier_extended_xmp
    ):
        contents["type"] = "Adobe extended XMP"
        idx = len(identifier_extended_xmp)
        return contents

    else:
        print("error: parse_app1: invalid identifier (%s)" % blob[0:6], file=sys.stderr)
        sys.exit(-1)


# https://www.color.org/specification/ICC1v43_2010-12.pdf
def parse_app2(blob):
    contents = collections.OrderedDict()
    identifier_icc_profile = b"ICC_PROFILE\x00"
    identifier_mpf = b"MPF\x00"
    if blob[0 : len(identifier_icc_profile)] == identifier_icc_profile:
        contents["type"] = "ICC_PROFILE"
        idx = len(identifier_icc_profile)
        icc_chunk_count = blob[idx]
        idx += 1
        icc_total_chunks = blob[idx]
        idx += 1
        return contents

    elif blob[0:4] == identifier_mpf:
        contents["type"] = "MPF"
        return contents

    else:
        print("error: parse_app2: invalid identifier", file=sys.stderr)
        sys.exit(-1)


def parse_app3(blob):
    return parse_unimplemented(0xFFE3, blob)


def parse_app4(blob):
    return parse_unimplemented(0xFFE4, blob)


def parse_app5(blob):
    return parse_unimplemented(0xFFE5, blob)


def parse_app11(blob):
    return parse_unimplemented(0xFFEB, blob)


# https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_JPEG_files
def parse_app13(blob):
    return "Adobe Photoshop non-graphic information"


ADOBE_COLOR_TRANSFORM = {
    0: "Unknown (RGB or CMYK)",
    1: "YCbCr",
    2: "YCCK",
}


def parse_app14(blob):
    contents = collections.OrderedDict()
    identifier_adobe = b"Adobe\x00"
    assert blob[0 : len(identifier_adobe)] == identifier_adobe
    idx = len(identifier_adobe)
    DCTEncodeVersion = blob[idx]
    contents["DCTEncodeVersion"] = DCTEncodeVersion
    idx += 1
    APP14Flags0 = blob[idx]
    contents["APP14Flags0"] = APP14Flags0
    idx += 1
    APP14Flags1 = blob[idx]
    contents["APP14Flags1"] = APP14Flags1
    idx += 1
    ColorTransform = blob[idx]
    contents["ColorTransform"] = ColorTransform
    idx += 1
    ColorTransformStr = ADOBE_COLOR_TRANSFORM[ColorTransform]
    contents["ColorTransformStr"] = ColorTransformStr
    return contents


def parse_com(blob):
    contents = collections.OrderedDict()
    comment = 'comment: "%s"' % blob.decode("ascii", "ignore")
    comment = comment.replace("\x00", "\\x00")
    contents["comment"] = comment
    return contents


def parse_dqt(blob):
    contents = collections.OrderedDict()
    assert len(blob) == 65, f"invalid DQT length: {len(blob)} [should be 65]"
    idx = 0
    first_byte = blob[idx]
    idx += 1
    Pq = first_byte >> 4
    contents["Pq"] = Pq
    Tq = first_byte & 0x0F
    contents["Tq"] = Tq
    Q = []
    for _ in range(64):
        Q.append(blob[idx])
        idx += 1
    contents["Q"] = Q
    return contents


def parse_dri(blob):
    contents = collections.OrderedDict()
    assert len(blob) == 2
    size = struct.unpack(">H", blob[0:2])[0]
    contents["size"] = size
    return contents


def parse_sos(blob):
    contents = collections.OrderedDict()
    assert len(blob) >= 6
    idx = 0
    number_of_components = blob[idx]
    contents["number_of_components"] = number_of_components
    idx += 1
    assert len(blob) == 1 + number_of_components * 2 + 3
    components = []
    for _ in range(number_of_components):
        component = collections.OrderedDict()
        component["component_id"] = blob[idx]
        component["component_id"] = COMPONENT_ID_STR[component["component_id"]]
        idx += 1
        component["huffman_ac_table"] = blob[idx] & 0x0F
        component["huffman_dc_table"] = blob[idx] >> 4
        idx += 1
    contents["components"] = components
    start_of_spectral_selection = blob[idx]
    idx += 1
    end_of_spectral_selection = blob[idx]
    idx += 1
    contents["spectral_selection"] = (
        {start_of_spectral_selection},
        {end_of_spectral_selection},
    )
    approximation_bit_position_high = blob[idx] & 0x0F
    approximation_bit_position_low = blob[idx] >> 4
    idx += 1
    contents["approximation_bit"] = (
        {approximation_bit_position_high},
        {approximation_bit_position_low},
    )
    return contents


SOF_ID_STR = {
    0: "baseline DCT, non-differential, huffman-coding frames",
    1: "extended sequential DCT, non-differential, huffman-coding frames",
    2: "progressive DCT frame, non-differential, huffman-coding frames",
    3: "lossless (sequential), non-differential, huffman-coding frames",
    5: "sequential DCT, differential, huffman-coding frames",
    6: "progressive DCT, differential, huffman-coding frames",
    7: "lossless, differential, huffman-coding frames",
    9: "sequential DCT, non-differential, arithmetic-coding frames",
    10: "progressive DCT, non-differential, arithmetic-coding frames",
    11: "lossless (sequential), non-differential, arithmetic-coding frames",
    13: "sequential DCT, differential, arithmetic-coding frames",
    14: "progressive DCT, differential, arithmetic-coding frames",
    15: "lossless (sequential), differential, arithmetic-coding frames",
}


COMPONENT_ID_STR = {
    1: "Y",
    2: "Cb",
    3: "Cr",
    4: "I",
    5: "Q",
    # JCS_BG_YCC
    34: "C",
    35: "C",
    # RGB
    82: "R",
    71: "G",
    66: "B",
    # JCS_BG_RGB
    114: "r",
    103: "g",
    98: "b",
}


def parse_sof0(blob):
    return parse_sof(0, blob)


def parse_sof1(blob):
    return parse_sof(1, blob)


def parse_sof2(blob):
    return parse_sof(2, blob)


def parse_sof3(blob):
    return parse_sof(3, blob)


def parse_sof5(blob):
    return parse_sof(5, blob)


def parse_sof6(blob):
    return parse_sof(6, blob)


def parse_sof7(blob):
    return parse_sof(7, blob)


def parse_sof9(blob):
    return parse_sof(9, blob)


def parse_sof10(blob):
    return parse_sof(10, blob)


def parse_sof11(blob):
    return parse_sof(11, blob)


def parse_sof13(blob):
    return parse_sof(13, blob)


def parse_sof14(blob):
    return parse_sof(14, blob)


def parse_sof15(blob):
    return parse_sof(15, blob)


def parse_sof(sof_id, blob):
    contents = collections.OrderedDict()
    sof_str = SOF_ID_STR[sof_id]
    contents["sof_str"] = sof_str
    assert len(blob) >= 6
    idx = 0
    sample_precision = blob[idx]
    contents["sample_precision"] = sample_precision
    idx += 1
    number_of_lines = struct.unpack(">H", blob[idx : idx + 2])[0]
    contents["number_of_lines"] = number_of_lines
    idx += 2
    number_of_samples_per_line = struct.unpack(">H", blob[idx : idx + 2])[0]
    contents["number_of_samples_per_line"] = number_of_samples_per_line
    idx += 2
    number_of_components_in_frame = blob[idx]
    contents["number_of_components_in_frame"] = number_of_components_in_frame
    idx += 1
    # assert len(blob) == 6 + number_of_components_in_frame * 3
    components = []
    for _ in range(number_of_components_in_frame):
        component = collections.OrderedDict()
        component["component_id"] = blob[idx]
        component["component_id_str"] = COMPONENT_ID_STR[component["component_id"]]
        idx += 1
        the_byte = blob[idx]
        idx += 1
        component["horizontal_sampling_factor"] = the_byte >> 4
        component["vertical_sampling_factor"] = the_byte & 0x0F
        component["quantization_table_selector"] = blob[idx]
        idx += 1
        components.append(component)
    contents["components"] = components
    return contents


def parse_dht(blob):
    contents = collections.OrderedDict()
    assert len(blob) > 16, f"invalid DHT length: {len(blob)} [should be at least 17]"
    idx = 0
    first_byte = blob[idx]
    idx += 1
    Tc = first_byte >> 4
    contents["Tc"] = Tc
    Th = first_byte & 0x0F
    contents["Th"] = Th
    L = []
    for _ in range(16):
        L.append(blob[idx])
        idx += 1
    V = []
    for i in range(16):
        Vi = []
        for _ in range(L[i]):
            Vi.append(blob[idx])
            idx += 1
        V.append(Vi)
    contents["L"] = L
    for i, Vi in enumerate(V):
        if not Vi:
            continue
        contents[f"V{i}"] = Vi
    return contents


def parse_jpg(blob):
    return parse_unimplemented(0xFFC8, blob)


def parse_dac(blob):
    return parse_unimplemented(0xFFCC, blob)


# https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_JPEG_files
MARKER_MAP = {
    0xFFC0: ("SOF0", parse_sof0),
    0xFFC1: ("SOF1", parse_sof1),
    0xFFC2: ("SOF2", parse_sof2),
    0xFFC3: ("SOF3", parse_sof3),
    0xFFC4: ("DHT", parse_dht),
    0xFFC5: ("SOF5", parse_sof5),
    0xFFC6: ("SOF6", parse_sof6),
    0xFFC7: ("SOF7", parse_sof7),
    0xFFC8: ("JPG", parse_jpg),
    0xFFC9: ("SOF9", parse_sof9),
    0xFFCA: ("SOF10", parse_sof10),
    0xFFCB: ("SOF11", parse_sof11),
    0xFFCC: ("DAC", parse_dac),
    0xFFCD: ("SOF13", parse_sof13),
    0xFFCE: ("SOF14", parse_sof14),
    0xFFCF: ("SOF15", parse_sof15),
    0xFFD8: ("SOI", None),
    0xFFD9: ("EOI", None),
    0xFFDA: ("SOS", parse_sos),
    0xFFDB: ("DQT", parse_dqt),
    0xFFDD: ("DRI", parse_dri),
    # https://exiftool.org/TagNames/JPEG.html
    0xFFE0: ("APP0", parse_app0),
    0xFFE1: ("APP1", parse_app1),
    0xFFE2: ("APP2", parse_app2),
    0xFFE3: ("APP3", parse_app3),
    0xFFE4: ("APP4", parse_app4),
    0xFFE5: ("APP5", parse_app5),
    0xFFEB: ("APP11", parse_app11),
    0xFFED: ("APP13", parse_app13),
    0xFFEE: ("APP14", parse_app14),
    0xFFFE: ("COM", parse_com),
}


def parse_jfif_file(infile, logfd, debug):
    marker_list = []
    # parse input file
    with open(infile, "rb") as fin:
        while True:
            offset = fin.tell()
            marker_size = 2
            marker = struct.unpack(">H", fin.read(marker_size))[0]
            if marker == "":
                # eof
                break
            elif marker in (0xFFD8, 0xFFD9):
                length = 0
                length_size = 0
            else:
                length = struct.unpack(">H", fin.read(2))[0]
                length_size = 2
                # read the full blob
                blob = fin.read(length - length_size)
            if marker not in MARKER_MAP:
                print("error: invalid marker: 0x%04x" % marker, file=sys.stderr)
                sys.exit(-1)
            marker_str, marker_parser = MARKER_MAP[marker]
            if debug > 0:
                print("debug: marker: 0x%04x (%s)" % (marker, marker_str), file=logfd)
            if marker_parser is not None:
                contents = marker_parser(blob)

            else:
                contents = collections.OrderedDict()
            marker_list.append(
                [
                    marker,
                    marker_str,
                    offset,
                    length,
                    contents,
                ]
            )
            if marker == 0xFFDA:  # SOS
                # compressed data
                compressed_data_offset = fin.tell()
                compressed_data_length = 0
                while True:
                    byte = fin.read(1)
                    if not byte:
                        # end of file reached
                        break
                    # check if this is 0xff
                    if byte == b"\xff":
                        byte2 = fin.read(1)
                        if byte2 != b"\x00":
                            # new marker
                            break
                    compressed_data_length += 1
                if byte:
                    # there is another marker
                    fin.seek(fin.tell() - 2)
                contents = collections.OrderedDict()
                marker_list.append(
                    [
                        0,  # marker,
                        "compressed_data",  # marker_str,
                        fin.tell() - compressed_data_offset,
                        compressed_data_length,
                        contents,
                    ]
                )
            if marker == 0xFFD9:  # stop parsing after EOI
                break
    return marker_list


def print_marker_list(marker_list, outfile, debug):
    # dump contents into output file
    with open(outfile, "w") as fout:
        for (
            _marker,
            marker_str,
            offset,
            length,
            contents,
        ) in marker_list:
            fout.write(f"marker: {marker_str}\n")
            fout.write(f"  offset: 0x{offset:08x}\n")
            fout.write(f"  marker_id: 0x{_marker:04x}\n")
            fout.write(f"  length: {length}\n")
            for k, v in contents.items():
                if (
                    isinstance(v, (list, tuple))
                    and v
                    and isinstance(v[0], collections.OrderedDict)
                ):
                    for i, d in enumerate(v):
                        fout.write(f"    {i}:")
                        for k2, v2 in d.items():
                            fout.write(f" {k2}: {v2}")
                        fout.write("\n")
                    continue
                elif isinstance(v, (list, tuple)):
                    v_str = "[" + ",".join(list(str(element) for element in v)) + "]"
                else:
                    v_str = str(v)
                fout.write(f"  {k}: {v}\n")


def extract_marker(marker_list, marker_name, outfile, debug):
    for (
        _marker,
        marker_str,
        offset,
        length,
        contents_offset,
        contents_length,
        contents_str,
        contents_bin,
    ) in marker_list:
        if marker_str == marker_name:
            break
    else:
        print(f"error: unknown marker: {marker_name}", file=sys.stderr)
        sys.exit(-1)
    # dump the contents
    with open(outfile, "wb") as fout:
        fout.write(contents_bin)


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument(
        "--func",
        type=str,
        nargs="?",
        default=default_values["func"],
        choices=FUNC_CHOICES.keys(),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FUNC_CHOICES.items())),
    )
    for key, val in FUNC_CHOICES.items():
        parser.add_argument(
            f"--{key}",
            action="store_const",
            dest="func",
            const=f"{key}",
            help=val,
        )
    parser.add_argument(
        "--marker",
        type=str,
        dest="marker",
        default=default_values["marker"],
        metavar="marker-name",
        help="marker name",
    )
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    parser.add_argument(
        "--logfile",
        action="store",
        dest="logfile",
        type=str,
        default=default_values["logfile"],
        metavar="log-file",
        help="log file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get logfile descriptor
    if options.logfile is None:
        logfd = sys.stdout
    else:
        logfd = open(options.logfile, "w")
    # get infile/outfile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")
    # do something
    marker_list = parse_jfif_file(options.infile, logfd, options.debug)
    if options.func == "parse":
        print_marker_list(marker_list, options.outfile, options.debug)
    elif options.func == "extract":
        extract_marker(marker_list, options.marker, options.outfile, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
