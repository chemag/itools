#!/usr/bin/env python


import argparse
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
    identifier_jfif = b"JFIF\x00"
    if blob[0 : len(identifier_jfif)] == identifier_jfif:
        i = len(identifier_jfif)
        # JFIF APP0 marker segment
        assert len(blob) >= 14
        version_major = blob[i]
        i += 1
        version_minor = blob[i]
        i += 1
        density_units = blob[i]
        i += 1
        xdensity = struct.unpack(">H", blob[i : i + 2])[0]
        i += 2
        ydensity = struct.unpack(">H", blob[i : i + 2])[0]
        i += 2
        xthumbnail = blob[i]
        i += 1
        ythumbnail = blob[i]
        i += 1
        # thumbnail_data = ...
        blob_str = ""
        blob_str += 'type: "JFIF\\x00"'
        blob_str += " version_major: %i" % version_major
        blob_str += " version_minor: %i" % version_minor
        blob_str += " density_units: %i" % density_units
        blob_str += " xdensity: %i" % xdensity
        blob_str += " ydensity: %i" % ydensity
        blob_str += " xthumbnail: %i" % xthumbnail
        blob_str += " ythumbnail: %i" % ythumbnail
        return blob_str, len(identifier_jfif)

    else:
        print("error: parse_app0: invalid identifier (%s)" % blob[0:5], file=sys.stderr)
        sys.exit(-1)


# http://www.exif.org/Exif2-2.PDF
def parse_app1(blob):
    identifier_exif = (b"Exif\x00\x00", b"Exif\x00\xff")
    identifier_xmp = b"http://ns.adobe.com/xap/1.0/\x00"
    identifier_extended_xmp = b"http://ns.adobe.com/xmp/extension/\x00"

    if (
        len(blob) >= len(identifier_exif[0])
        and blob[0 : len(identifier_exif[0])] in identifier_exif
    ):
        i = len(identifier_exif[0])
        assert len(blob) >= 14
        _padding = blob[i]
        i += 1
        ifd_tag = struct.unpack(">H", blob[i : i + 2])[0]
        ifd_type = struct.unpack(">H", blob[i + 2 : i + 4])[0]
        ifd_count = struct.unpack(">I", blob[i + 4 : i + 8])[0]
        ifd_value_offset = struct.unpack(">I", blob[i + 8 : i + 12])[0]
        # thumbnail_data = ...
        return "EXIF", len(identifier_exif[0])

    elif (
        len(blob) >= len(identifier_xmp)
        and blob[0 : len(identifier_xmp)] == identifier_xmp
    ):
        i = len(identifier_xmp)
        return "Adobe XMP", len(identifier_xmp)

    elif (
        len(blob) >= len(identifier_extended_xmp)
        and blob[0 : len(identifier_extended_xmp)] == identifier_extended_xmp
    ):
        i = len(identifier_extended_xmp)
        return "Adobe extended XMP", len(identifier_extended_xmp)

    else:
        print("error: parse_app1: invalid identifier (%s)" % blob[0:6], file=sys.stderr)
        sys.exit(-1)


# https://www.color.org/specification/ICC1v43_2010-12.pdf
def parse_app2(blob):
    identifier_icc_profile = b"ICC_PROFILE\x00"
    identifier_mpf = b"MPF\x00"
    if blob[0 : len(identifier_icc_profile)] == identifier_icc_profile:
        i = len(identifier_icc_profile)
        icc_chunk_count = blob[i]
        i += 1
        icc_total_chunks = blob[i]
        i += 1
        return "ICC_PROFILE", i

    elif blob[0:4] == identifier_mpf:
        return "MPF", len(identifier_mpf)

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
    identifier_adobe = b"Adobe\x00"
    assert blob[0 : len(identifier_adobe)] == identifier_adobe
    i = len(identifier_adobe)
    DCTEncodeVersion = blob[i]
    i += 1
    APP14Flags0 = blob[i]
    i += 1
    APP14Flags1 = blob[i]
    i += 1
    ColorTransform = blob[i]
    i += 1
    ColorTransformStr = ADOBE_COLOR_TRANSFORM[ColorTransform]
    blob_str = "DCTEncodeVersion: %i" % DCTEncodeVersion
    blob_str = " APP14Flags0: %i" % APP14Flags0
    blob_str = " APP14Flags1: %i" % APP14Flags1
    blob_str = " ColorTransform: %i" % ColorTransform
    blob_str = ' ColorTransformStr: "%s"' % ColorTransformStr
    return blob_str, len(identifier_adobe)


def parse_com(blob):
    comment = 'comment: "%s"' % blob.decode("ascii", "ignore")
    comment = comment.replace("\x00", "\\x00")
    return comment, 0


def parse_dqt(blob):
    return parse_unimplemented(0xFFDB, blob)


def parse_dri(blob):
    assert len(blob) == 2
    size = struct.unpack(">H", blob[0:2])[0]
    blob_str = "size: %i" % (size)
    return blob_str, 0


def parse_sos(blob):
    assert len(blob) >= 6
    i = 0
    number_of_components = blob[i]
    i += 1
    assert len(blob) == 1 + number_of_components * 2 + 3
    components = []
    for _ in range(number_of_components):
        component_id = blob[i]
        i += 1
        huffman_ac_table = blob[i] & 0x0F
        huffman_dc_table = blob[i] >> 4
        i += 1
        components.append(
            [
                component_id,
                COMPONENT_ID_STR[component_id],
                huffman_ac_table,
                huffman_dc_table,
            ]
        )
    start_of_spectral_selection = blob[i]
    i += 1
    end_of_spectral_selection = blob[i]
    i += 1
    approximation_bit_position_high = blob[i] & 0x0F
    approximation_bit_position_low = blob[i] >> 4
    i += 1
    blob_str = "number_of_components: %i" % number_of_components
    blob_str = "components: %r" % components
    blob_str = "spectral_selection: (%i, %i)" % (
        start_of_spectral_selection,
        end_of_spectral_selection,
    )
    blob_str = "approximation_bit: (%i, %i)" % (
        approximation_bit_position_high,
        approximation_bit_position_low,
    )
    return blob_str, 0


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
    sof_str = SOF_ID_STR[sof_id]
    assert len(blob) >= 6
    i = 0
    sample_precision = blob[i]
    i += 1
    number_of_lines_precision = struct.unpack(">H", blob[i : i + 2])[0]
    i += 2
    number_of_samples_per_line = struct.unpack(">H", blob[i : i + 2])[0]
    i += 2
    number_of_components_in_frame = blob[i]
    i += 1
    # assert len(blob) == 6 + number_of_components_in_frame * 3
    components = []
    for i in range(number_of_components_in_frame):
        component_id = blob[6 + 3 * i + 0]
        vsampling = blob[6 + 3 * i + 1] & 0x0F
        hsampling = blob[6 + 3 * i + 1] >> 4
        qtable = blob[6 + 3 * i + 2]
        components.append(
            [component_id, COMPONENT_ID_STR[component_id], vsampling, hsampling, qtable]
        )

    blob_str = 'sof_str: "%s"' % sof_str
    blob_str = " sample_precision: %i" % sample_precision
    blob_str = " number_of_lines_precision: %i" % number_of_lines_precision
    blob_str = " number_of_samples_per_line: %i" % number_of_samples_per_line
    blob_str = " number_of_components_in_frame: %i" % (number_of_components_in_frame)
    blob_str = " components: %r" % components
    return blob_str, 0


def parse_dht(blob):
    return parse_unimplemented(0xFFC4, blob)


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
                contents_str, payload_offset = marker_parser(blob)
                contents_offset = offset + marker_size + length_size + payload_offset
                contents_length = length - length_size - payload_offset
                contents_bin = blob[payload_offset:]

            else:
                contents_str = ""
                payload_offset = 0
                contents_offset = offset + length_size + payload_offset
                contents_length = length - length_size - payload_offset
                contents_bin = ""
            marker_list.append(
                [
                    marker,
                    marker_str,
                    offset,
                    length,
                    contents_offset,
                    contents_length,
                    contents_str,
                    contents_bin,
                ]
            )
            if marker in (0xFFD9, 0xFFDA):  # stop parsing after EOI or SOS
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
            contents_offset,
            contents_length,
            contents_str,
            _contents_bin,
        ) in marker_list:
            contents_str_full = f"contents {{ "
            contents_str_full += f"offset: {contents_offset} "
            contents_str_full += f"length: {contents_length} "
            contents_str_full += f"data {{ {contents_str} }}"
            contents_str_full += "}"
            fout.write(f"marker: {marker_str}\n")
            fout.write(f"  header: 0x{offset:08x}\n")
            fout.write(f"  length: {length}\n")
            if contents_str_full:
                fout.write(f"  contents: {contents_str_full}\n")


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
