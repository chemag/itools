#!/usr/bin/env python3

"""alphacodec.py module description."""


import argparse
import bitstring
import importlib
import numpy as np
import os
import sys

itools_common = importlib.import_module("itools-common")
itools_y4m = importlib.import_module("itools-y4m")


__version__ = "0.1"

CODEC_LIST = [
    "warhol",
    "warhol2",
    "bitmap",
    "resolution-7",
    "resolution-6",
    "resolution-5",
    "resolution-4",
    "resolution-3",
    "resolution-2",
    "resolution-1",
]

FUNC_CHOICES = {
    "help": "show help options",
    "encode": "encode y4m into alpha binary file",
    "decode": "decode alpha binary file into y4m",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "codec": "warhol",
    "block_size": 8,
    "func": "help",
    "infile": None,
    "outfile": None,
}


BITS_PER_BYTE = 8


def write_header(height, width, colorspace, codec, block_size):
    header = f"ALPHA {height} {width} {colorspace} {codec} {block_size}\n"
    return header


def read_header(header_line):
    parameters = header_line.decode("ascii").split(" ")
    assert parameters[0] == "ALPHA", "invalid alpha file: starts with {parameters[0]}"
    assert len(parameters) == 6, f"error: invalid header: '{header_line}'"
    try:
        height = int(parameters[1])
        width = int(parameters[2])
        colorspace = parameters[3].strip()
        codec = parameters[4].strip()
        block_size = int(parameters[5].strip())
    except:
        print(f"error: invalid header: '{header_line}'")
        sys.exit(-1)
    return height, width, colorspace, codec, block_size


def block2string(block):
    block_str = (
        "["
        + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in block)
        + "]"
    )
    return block_str


# warhol encoder
def encode_warhol(yarray, colorspace, block_size, debug):
    # loop though each block_size block, and add the bits into
    # a BitStream
    height, width = yarray.shape
    stream = bitstring.BitStream()
    max_value = itools_common.COLORSPACES[colorspace]["depth"].get_max()
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = yarray[i : i + block_size, j : j + block_size]
            # encode the block
            if np.all(block == 0):
                stream.append("0b00")
            elif np.all(block == max_value):
                stream.append("0b01")
            else:
                stream.append("0b10")
                if debug > 0:
                    block_str = block2string(block)
                    print(f"debug: interesting block at {i},{j}: {block_str}")
                # flatten array to 1D and convert to <elements> bytes
                byte_data = block.flatten().tobytes()
                stream.append(byte_data)
    return stream


def decode_warhol(height, width, colorspace, block_size, stream, debug):
    # allocate space for the whole image
    yarray = np.zeros((height, width), dtype=np.uint8)
    i = 0
    j = 0
    elements = block_size * block_size
    max_value = itools_common.COLORSPACES[colorspace]["depth"].get_max()
    stats = {k: 0 for k in ["00", "01", "10", "11"]}
    while True:
        # decode the block
        try:
            val = stream.read("uint:2")
        except bitstring.ReadError:
            # end of stream or not enough bits to read 2 bits
            break
        if val == 0b00:
            stats["00"] += 1
            # create an all-zero block
            block = np.full((block_size, block_size), 0, dtype=np.uint8)
        elif val == 0b01:
            stats["01"] += 1
            # create an all-max_value block
            block = np.full((block_size, block_size), max_value, dtype=np.uint8)
        elif val == 0b10:
            stats["10"] += 1
            # read the full block
            bits = stream.read(f"bytes:{elements}")
            block = np.frombuffer(bits, dtype=np.uint8).reshape(block_size, block_size)
            if debug > 0:
                block_str = block2string(block)
                print(f"debug: interesting block at {i},{j}: {block_str}")
        else:
            print(f"error: invalid bitstream: 0b11")
            sys.exit(-1)
        # copy the block into the array
        yarray[i : i + block_size, j : j + block_size] = block
        j += block_size
        if j >= width:
            i += block_size
            if i >= height:
                i = height
                # ensure there are no bytes left
                if (stream.len - stream.pos) >= BITS_PER_BYTE:
                    print(f"warning: there are {stream.len - stream.pos} bits left")
                break
            j = 0
    # check that we have covered all the matrix
    if j < width or i < height:
        print(f"warning: only read {j}x{i} on a {width}x{height} image")
    return yarray, stats


# warhol2 encoder
def encode_warhol2(yarray, colorspace, block_size, debug):
    # preprocess the yarray to replace Y=254 with Y=255
    height, width = yarray.shape
    for i in range(0, height):
        for j in range(0, width):
            if yarray[i, j] == 254:
                yarray[i, j] = 255
            elif yarray[i, j] == 1:
                yarray[i, j] = 0
    return encode_warhol(yarray, colorspace, block_size, debug)


def decode_warhol2(height, width, colorspace, block_size, stream, debug):
    yarray, stats = decode_warhol(height, width, colorspace, block_size, stream, debug)
    return yarray, stats


# bitmap encoder
def encode_bitmap(yarray, colorspace, block_size, debug):
    # loop though each block_size block, and add the bits into
    # a BitStream
    height, width = yarray.shape
    max_value = itools_common.COLORSPACES[colorspace]["depth"].get_max()
    stream = bitstring.BitStream()
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = yarray[i : i + block_size, j : j + block_size]
            # encode the block
            if np.all(block == 0):
                stream.append("0b00")
            elif np.all(block == max_value):
                stream.append("0b01")
            elif np.all(np.isin(block, [0, max_value])):
                stream.append("0b11")
                # normalize block to 0/1
                block = (block // max_value).astype(np.uint8)
                binary_stream = "0b" + "".join(str(b) for b in block.flatten())
                stream.append(binary_stream)
            else:
                stream.append("0b10")
                if debug > 0:
                    block_str = block2string(block)
                    print(f"debug: interesting block at {i},{j}: {block_str}")
                # flatten array to 1D and convert to <element> bytes
                byte_data = block.flatten().tobytes()
                stream.append(byte_data)
    return stream


def decode_bitmap(height, width, colorspace, block_size, stream, debug):
    # allocate space for the whole image
    yarray = np.zeros((height, width), dtype=np.uint8)
    i = 0
    j = 0
    elements = block_size * block_size
    max_value = itools_common.COLORSPACES[colorspace]["depth"].get_max()
    stats = {k: 0 for k in ["00", "01", "10", "11"]}
    while True:
        # decode the block
        try:
            val = stream.read("uint:2")
        except bitstring.ReadError:
            # end of stream or not enough bits to read 2 bits
            break
        if val == 0b00:
            stats["00"] += 1
            # create an all-zero block
            block = np.full((block_size, block_size), 0, dtype=np.uint8)
        elif val == 0b01:
            stats["01"] += 1
            # create an all-max_value block
            block = np.full((block_size, block_size), max_value, dtype=np.uint8)
        elif val == 0b11:
            stats["11"] += 1
            bits = stream.read(f"bits:{elements}")
            bit_list = [int(bits.read("bool")) for _ in range(elements)]
            block = (np.array(bit_list, dtype=np.uint8) * max_value).reshape(
                (block_size, block_size)
            )
        elif val == 0b10:
            stats["10"] += 1
            # read the <elements>-byte block
            bits = stream.read(f"bytes:{elements}")
            block = np.frombuffer(bits, dtype=np.uint8).reshape(block_size, block_size)
            if debug > 0:
                block_str = block2string(block)
                print(f"debug: interesting block at {i},{j}: {block_str}")
        # copy the block into the array
        yarray[i : i + block_size, j : j + block_size] = block
        j += block_size
        if j >= width:
            i += block_size
            if i >= height:
                i = height
                # ensure there are no bytes left
                if (stream.len - stream.pos) >= BITS_PER_BYTE:
                    print(f"warning: there are {stream.len - stream.pos} bits left")
                break
            j = 0
    # check that we have covered all the matrix
    if j < width or i < height:
        print(f"warning: only read {j}x{i} on a {width}x{height} image")
    return yarray, stats


# resolution-* encoder
def encode_resolution(codec, yarray, colorspace, block_size, debug):
    codec_depth = int(codec.split("-")[1])
    # loop though each block_size block, and add the bits into
    # a BitStream
    height, width = yarray.shape
    stream = bitstring.BitStream()
    depth = itools_common.COLORSPACES[colorspace]["depth"].get_depth()
    max_encoded_value = (1 << codec_depth) - 1
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = yarray[i : i + block_size, j : j + block_size]
            # reduce the actual depth
            block = (block >> (depth - codec_depth)).astype(np.uint8)
            # encode the block
            if np.all(block == 0):
                stream.append("0b00")
            elif np.all(block == max_encoded_value):
                stream.append("0b01")
            elif np.all(np.isin(block, [0, max_encoded_value])):
                stream.append("0b11")
                # normalize block to 0/1
                block = (block // max_encoded_value).astype(np.uint8)
                binary_stream = "0b" + "".join(str(b) for b in block.flatten())
                stream.append(binary_stream)
            else:
                stream.append("0b10")
                for val in block.flatten():
                    stream.append(f"uint:{codec_depth}={val}")
    return stream


def decode_resolution(codec, height, width, colorspace, block_size, stream, debug):
    codec_depth = int(codec.split("-")[1])
    # allocate space for the whole image
    yarray = np.zeros((height, width), dtype=np.uint8)
    i = 0
    j = 0
    elements = block_size * block_size
    max_value = itools_common.COLORSPACES[colorspace]["depth"].get_max()
    depth = itools_common.COLORSPACES[colorspace]["depth"].get_depth()
    max_encoded_value = (1 << codec_depth) - 1
    stats = {k: 0 for k in ["00", "01", "10", "11"]}
    while True:
        # decode the block
        try:
            val = stream.read("uint:2")
        except bitstring.ReadError:
            # end of stream or not enough bits to read 2 bits
            break
        if val == 0b00:
            stats["00"] += 1
            # create an all-zero block
            block = np.full((block_size, block_size), 0, dtype=np.uint8)
        elif val == 0b01:
            stats["01"] += 1
            # create an all-<max_encoded_value> block
            block = np.full((block_size, block_size), max_encoded_value, dtype=np.uint8)
        elif val == 0b11:
            stats["11"] += 1
            bits = stream.read(f"bits:{elements}")
            bit_list = [int(bits.read("bool")) for _ in range(elements)]
            block = (np.array(bit_list, dtype=np.uint8) * max_encoded_value).reshape(
                (block_size, block_size)
            )
        elif val == 0b10:
            stats["10"] += 1
            bits_read = elements * codec_depth
            bits = stream.read(f"bits:{bits_read}")
            values = [bits.read(f"uint:{codec_depth}") for _ in range(elements)]
            block = np.array(values, dtype=np.uint8).reshape((block_size, block_size))
        # unnormalize block to the MSB
        block = block << (depth - codec_depth)
        # replace all the max values with max_value
        block[block == max_encoded_value] = max_value
        # copy the block into the array
        yarray[i : i + block_size, j : j + block_size] = block
        j += block_size
        if j >= width:
            i += block_size
            if i >= height:
                i = height
                # ensure there are no bytes left
                if (stream.len - stream.pos) >= BITS_PER_BYTE:
                    print(f"warning: there are {stream.len - stream.pos} bits left")
                break
            j = 0
    # check that we have covered all the matrix
    if j < width or i < height:
        print(f"warning: only read {j}x{i} on a {width}x{height} image")
    return yarray, stats


def encode_file(infile, outfile, codec, block_size, debug):
    # 1. read y4m input as numpy array
    outyvu = itools_y4m.read_y4m_image(
        infile,
        output_colorrange=itools_common.ColorRange.full,
        debug=debug,
    )
    # keep the luminance only
    yarray = outyvu[:, :, 0]
    height, width = yarray.shape
    colorspace = "mono"

    # 2. ensure block_size-alignment
    if height % block_size != 0:
        last_row = yarray[-1:, :]
        last_rows = np.repeat(last_row, (block_size - (height % block_size)), axis=0)
        yarray = np.concatenate((yarray, last_rows), axis=0)
    if width % block_size != 0:
        last_col = yarray[:, -1:]
        last_cols = np.repeat(last_col, (block_size - (width % block_size)), axis=1)
        yarray = np.concatenate((yarray, last_cols), axis=1)

    # 3. encode the luminance
    if codec == "warhol":
        stream = encode_warhol(yarray, colorspace, block_size, debug)
    elif codec == "warhol2":
        stream = encode_warhol2(yarray, colorspace, block_size, debug)
    elif codec == "bitmap":
        stream = encode_bitmap(yarray, colorspace, block_size, debug)
    elif codec.startswith("resolution-"):
        stream = encode_resolution(codec, yarray, colorspace, block_size, debug)

    # 4. write encoded alpha channel to file
    with open(outfile, "wb") as fout:
        # write a small header
        header = write_header(height, width, colorspace, codec, block_size)
        fout.write(header.encode("utf-8"))
        stream.tofile(fout)


def decode_file(infile, outfile, debug):
    # 1. read the encoded file
    with open(infile, "rb") as f:
        # read the header
        header_line = f.readline()
        height, width, colorspace, codec, block_size = read_header(header_line)
        # read the data
        data = f.read()
    stream = bitstring.ConstBitStream(data)

    # 2. decode the encoded file into a luminance plane
    effective_height = ((height + (block_size - 1)) // block_size) * block_size
    effective_width = ((width + (block_size - 1)) // block_size) * block_size
    if codec == "warhol":
        yarray, stats = decode_warhol(
            effective_height, effective_width, colorspace, block_size, stream, debug
        )
    elif codec == "warhol2":
        yarray, stats = decode_warhol2(
            effective_height, effective_width, colorspace, block_size, stream, debug
        )
    elif codec == "bitmap":
        yarray, stats = decode_bitmap(
            effective_height, effective_width, colorspace, block_size, stream, debug
        )
    elif codec.startswith("resolution-"):
        yarray, stats = decode_resolution(
            codec,
            effective_height,
            effective_width,
            colorspace,
            block_size,
            stream,
            debug,
        )

    # 3. chop the array if needed
    if effective_height > height:
        yarray = yarray[:height, :]
    if effective_width > width:
        yarray = yarray[:, :width]

    # 4. write the array into a y4m file
    itools_y4m.write_y4m_image(outfile, yarray, colorspace)
    return stats


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
        "-v",
        "--version",
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
    )
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
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=default_values["dry_run"],
        help="Dry run",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Do not dry run",
    )
    parser.add_argument(
        "--codec",
        action="store",
        type=str,
        dest="codec",
        default=default_values["codec"],
        choices=CODEC_LIST,
        metavar="[%s]"
        % (
            " | ".join(
                CODEC_LIST,
            )
        ),
        help="alpha codec",
    )
    parser.add_argument(
        "--block-size",
        action="store",
        type=int,
        dest="block_size",
        default=default_values["block_size"],
        metavar="BLOCK_SIZE",
        help=(f"use BLOCK_SIZE block_size (default: {default_values['block_size']})"),
    )

    parser.add_argument(
        "func",
        type=str,
        nargs="?",
        default=default_values["func"],
        choices=FUNC_CHOICES.keys(),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FUNC_CHOICES.items())),
    )
    parser.add_argument(
        "-i",
        "--infile",
        dest="infile",
        type=str,
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # implement help
    if options.func == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print(f"version: {__version__}")
        sys.exit(0)

    # get infile/outfile
    if options.infile is None or options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    if options.func == "encode":
        encode_file(
            options.infile,
            options.outfile,
            options.codec,
            options.block_size,
            options.debug,
        )
    elif options.func == "decode":
        decode_file(options.infile, options.outfile, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
