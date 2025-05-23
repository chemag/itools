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
    "default",
]

FUNC_CHOICES = {
    "help": "show help options",
    "encode": "encode y4m into alpha binary file",
    "decode": "decode alpha binary file into y4m",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "codec": "default",
    "block_size": 8,
    "func": "help",
    "infile": None,
    "outfile": None,
}


def write_header(width, height, colorspace, codec, block_size):
    header = f"ALPHA {width} {height} {colorspace} {codec} {block_size}\n"
    return header


def read_header(header_line):
    parameters = header_line.decode("ascii").split(" ")
    assert parameters[0] == "ALPHA", "invalid alpha file: starts with {parameters[0]}"
    assert len(parameters) == 6, f"error: invalid header: '{header_line}'"
    try:
        width = int(parameters[1])
        height = int(parameters[2])
        colorspace = parameters[3].strip()
        codec = parameters[4].strip()
        block_size = int(parameters[5].strip())
    except:
        print(f"error: invalid header: '{header_line}'")
        sys.exit(-1)
    return width, height, colorspace, codec, block_size


def block2string(block):
    block_str = (
        "["
        + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in block)
        + "]"
    )
    return block_str


def encode_default(yarray, colorspace, block_size, debug):
    # loop though each block_size block, and add the bits into
    # a BitStream
    height, width = yarray.shape
    stream = bitstring.BitStream()
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = yarray[i : i + block_size, j : j + block_size]
            # extend the block if needed
            bh, bw = block.shape
            if bh < block_size:
                last_col = block[-1:, :]
                repeats = np.repeat(last_col, (block_size - bh), axis=0)
                block = np.concatenate((block, repeats), axis=0)
            if bw < block_size:
                last_col = block[:, -1:]
                repeats = np.repeat(last_col, (block_size - bw), axis=1)
                block = np.concatenate((block, repeats), axis=1)
            # encode the block
            if np.all(block == 0):
                stream.append("0b00")
            elif np.all(block == 255):
                stream.append("0b01")
            else:
                stream.append("0b10")
                if debug > 0:
                    block_str = block2string(block)
                    print(f"debug: interesting block at {i},{j}: {block_str}")
                # flatten array to 1D and convert to 64 bytes
                byte_data = block.flatten().tobytes()
                stream.append(byte_data)
    return stream


def decode_default(width, height, block_size, stream, debug):
    # allocate space for the whole image
    yarray = np.zeros((height, width), dtype=np.uint8)
    i = 0
    j = 0
    while True:
        # decode the block
        try:
            val = stream.read("uint:2")
        except bitstring.ReadError:
            # end of stream or not enough bits to read 2 bits
            break
        if val == 0b00:
            # create an all-zero block
            block = np.full((block_size, block_size), 0, dtype=np.uint8)
        elif val == 0b01:
            # create an all-255 block
            block = np.full((block_size, block_size), 255, dtype=np.uint8)
        elif val == 0b10:
            # read the 64-byte block
            bits = stream.read("bytes:64")
            block = np.frombuffer(bits, dtype=np.uint8).reshape(block_size, block_size)
            if debug > 0:
                block_str = block2string(block)
                print(f"debug: interesting block at {i},{j}: {block_str}")
        else:
            print(f"error: invalid bitstream: 0b11")
            sys.exit(-1)
        # chop the block if needed
        if j + block_size > width:
            block = block[:, : width - j]
        if i + block_size > height:
            block = block[: height - i, :]
        yarray[i : i + 8, j : j + 8] = block
        j += 8
        if j >= width:
            i += 8
            if i >= height:
                i = height
                # ensure there are no bytes left
                if (stream.len - stream.pos) >= 8:
                    print(f"warning: there are {stream.len - stream.pos} bits left")
                break
            j = 0
    # check that we have covered all the matrix
    if j < width or i < height:
        print(f"warning: only read {j}x{i} on a {width}x{height} image")
    return yarray


def encode_file(infile, outfile, codec, block_size, debug):
    # 1. read y4m input as numpy array
    outyvu, _, _, _ = itools_y4m.read_y4m(
        infile,
        output_colorrange=itools_common.ColorRange.full,
        logfd=None,
        debug=debug,
    )
    # keep the luminance only
    yarray = outyvu[:, :, 0]
    height, width = yarray.shape
    colorspace = "mono"

    # 2. encode the luminance
    if codec == "default":
        stream = encode_default(yarray, colorspace, block_size, debug)
    # write encoded alpha channel to file
    with open(outfile, "wb") as fout:
        # write a small header
        header = write_header(width, height, colorspace, codec, block_size)
        fout.write(header.encode("utf-8"))
        stream.tofile(fout)


def decode_file(infile, outfile, debug):
    # 1. read the encoded file
    with open(infile, "rb") as f:
        # read the header
        header_line = f.readline()
        width, height, colorspace, codec, block_size = read_header(header_line)
        # read the data
        data = f.read()
    stream = bitstring.ConstBitStream(data)

    # 2. decode the encoded file into a luminance plane
    yarray = decode_default(width, height, block_size, stream, debug)
    # write the array into a y4m file
    itools_y4m.write_y4m(outfile, yarray, colorspace)


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
