#!/usr/bin/env python3

"""itools-common.py module description.


Module that contains common code.
"""


import enum
import numpy as np
import re
import subprocess
import sys


FFMPEG_SILENT = "ffmpeg -hide_banner -y"


class EncoderException(Exception):
    """Encoder issue."""


class ProcColor(enum.Enum):
    bgr = 0
    yvu = 1
    both = 2


PROC_COLOR_LIST = list(c.name for c in ProcColor)


class ColorRange(enum.Enum):
    limited = 0
    full = 1
    unspecified = 2
    TOTAL = 3

    @classmethod
    def get_choices(cls, name=True):
        return list(
            (c.name if name else c.value) for c in ColorRange if c.name != "TOTAL"
        )

    @classmethod
    def get_default(cls, name=True):
        default = cls.unspecified
        return default.name if name else default.value

    @classmethod
    def parse(cls, val):
        if val is None:
            return cls.unspecified
        # map value (int)/name (str) to object
        for data in cls:
            if (type(val) is int and val == data.value) or (
                type(val) is str and val.lower() == data.name
            ):
                return data
        # default value
        return cls.unspecified

    @classmethod
    def to_str(cls, val):
        # map object/value (int) to name (str)
        for data in cls:
            if val == data.value or cls(val).value == data.value:
                return data.name
        return "unspecified"

    @classmethod
    def to_int(cls, val):
        # map object/name (str) to value (int)
        val = val.lower() if type(val) is str else val
        for data in cls:
            if val == data.name or val == data:
                return data.value
        return -1


class ChromaSubsample(enum.Enum):
    chroma_420 = 0
    chroma_422 = 1
    chroma_444 = 2


class ColorDepth(enum.Enum):
    depth_8 = 0
    depth_10 = 1

    def get_max(self):
        if self == self.depth_8:
            return 255
        elif self == self.depth_10:
            return 1023


# "colorspace": (ChromaSubsample, ColorDepth),
COLORSPACES = {
    "420": (
        ChromaSubsample.chroma_420,
        ColorDepth.depth_8,
    ),
    "420jpeg": (
        ChromaSubsample.chroma_420,
        ColorDepth.depth_8,
    ),
    "420paldv": (
        ChromaSubsample.chroma_420,
        ColorDepth.depth_8,
    ),
    "420mpeg2": (
        ChromaSubsample.chroma_420,
        ColorDepth.depth_8,
    ),
    "422": (
        ChromaSubsample.chroma_422,
        ColorDepth.depth_8,
    ),
    "444": (
        ChromaSubsample.chroma_444,
        ColorDepth.depth_8,
    ),
    "420p10": (
        ChromaSubsample.chroma_420,
        ColorDepth.depth_10,
    ),
}


class ImageInfo:
    width = None
    height = None

    def __init__(self, width, height, stride=None, scanline=None, colorrange=None):
        self.width = width
        self.height = height
        self.stride = stride
        self.scanline = scanline
        self.colorrange = colorrange

    def __str__(self):
        return f"width: {self.width} height: {self.height} stride: {self.stride} scanline: {self.scanline} colorrange: {self.colorrange.name}"


def run(command, **kwargs):
    debug = kwargs.get("debug", 0)
    dry_run = kwargs.get("dry_run", False)
    env = kwargs.get("env", None)
    stdin = subprocess.PIPE if kwargs.get("stdin", False) else None
    bufsize = kwargs.get("bufsize", 0)
    universal_newlines = kwargs.get("universal_newlines", False)
    default_close_fds = True if sys.platform == "linux2" else False
    close_fds = kwargs.get("close_fds", default_close_fds)
    shell = kwargs.get("shell", type(command) in (type(""), type("")))
    logfd = kwargs.get("logfd", sys.stdout)
    if debug > 0:
        print(f"$ {command}", file=logfd)
    if dry_run:
        return 0, b"stdout", b"stderr"
    gnu_time = kwargs.get("gnu_time", False)
    if gnu_time:
        # GNU /usr/bin/time support
        command = f"/usr/bin/time -v {command}"

    p = subprocess.Popen(  # noqa: E501
        command,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=bufsize,
        universal_newlines=universal_newlines,
        env=env,
        close_fds=close_fds,
        shell=shell,
    )
    # wait for the command to terminate
    if stdin is not None:
        out, err = p.communicate(stdin)
    else:
        out, err = p.communicate()
    returncode = p.returncode
    # clean up
    del p
    if gnu_time:
        # make sure the stats are there
        GNU_TIME_BYTES = b"\n\tUser time"
        assert GNU_TIME_BYTES in err, "error: cannot find GNU time info in stderr"
        gnu_time_str = err[err.index(GNU_TIME_BYTES) :].decode("ascii")
        gnu_time_stats = gnu_time_parse(gnu_time_str, logfd, debug)
        err = err[0 : err.index(GNU_TIME_BYTES) :]
        return returncode, out, err, gnu_time_stats
    # return results
    return returncode, out, err


# converts a chroma-subsampled matrix into a non-chroma subsampled one
# Algo is very simple (just dup values)
def chroma_subsample_reverse(inmatrix, colorspace):
    in_w, in_h = inmatrix.shape
    chroma_subsample = COLORSPACES[colorspace][0]
    if chroma_subsample == ChromaSubsample.chroma_420:
        out_w = in_w << 1
        out_h = in_h << 1
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix[::2, ::2] = inmatrix
        outmatrix[1::2, ::2] = inmatrix
        outmatrix[::2, 1::2] = inmatrix
        outmatrix[1::2, 1::2] = inmatrix
    elif chroma_subsample == ChromaSubsample.chroma_422:
        out_w = in_w << 1
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix[::, ::2] = inmatrix
        outmatrix[::, 1::2] = inmatrix
    elif chroma_subsample == ChromaSubsample.chroma_444:
        out_w = in_w
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix = inmatrix
    return outmatrix


# converts a non-chroma-subsampled matrix into a chroma subsampled one
# Algo is very simple (just average values)
def chroma_subsample_direct(inmatrix, colorspace):
    in_w, in_h = inmatrix.shape
    chroma_subsample = COLORSPACES[colorspace][0]
    if chroma_subsample == ChromaSubsample.chroma_420:
        out_w = in_w >> 1
        out_h = in_h >> 1
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint16)
        outmatrix += inmatrix[::2, ::2]
        outmatrix += inmatrix[1::2, ::2]
        outmatrix += inmatrix[::2, 1::2]
        outmatrix += inmatrix[1::2, 1::2]
        outmatrix = outmatrix / 4
        outmatrix = outmatrix.astype(np.uint8)
    elif chroma_subsample == ChromaSubsample.chroma_422:
        out_w = in_w >> 1
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint16)
        outmatrix += inmatrix[::, ::2]
        outmatrix += inmatrix[::, 1::2]
        outmatrix = outmatrix / 2
        outmatrix = outmatrix.astype(np.uint8)
    elif chroma_subsample == ChromaSubsample.chroma_444:
        out_w = in_w
        out_h = in_h
        outmatrix = np.zeros((out_w, out_h), dtype=np.uint8)
        outmatrix = inmatrix
    return outmatrix


class Config:
    DEFAULT_VALUES = {
        "read_image_components": True,
        "read_exif_info": True,
        "read_icc_info": True,
        "average_results": True,
        "qpextract_bin": None,
        "isobmff_parser": None,
        "h265nal_parser": None,
    }

    def __init__(self):
        self.config_dict = {}

    def __str__(self):
        return "\n".join(f"{k}: {v}" for (k, v) in self.config_dict.items())

    @classmethod
    def Create(cls, options):
        config_dict = cls()
        for key, val in vars(options).items():
            if key in cls.DEFAULT_VALUES.keys():
                config_dict.set(key, val)
        return config_dict

    @classmethod
    def set_parser_options(cls, parser):
        parser.add_argument(
            "--read-image-components",
            dest="read_image_components",
            action="store_true",
            default=cls.DEFAULT_VALUES["read_image_components"],
            help="Read image components%s"
            % (" [default]" if cls.DEFAULT_VALUES["read_image_components"] else ""),
        )
        parser.add_argument(
            "--no-read-image-components",
            dest="read_image_components",
            action="store_false",
            help="Do not read image components%s"
            % (" [default]" if not cls.DEFAULT_VALUES["read_image_components"] else ""),
        )
        parser.add_argument(
            "--exif",
            dest="read_exif_info",
            action="store_true",
            default=cls.DEFAULT_VALUES["read_exif_info"],
            help="Parse EXIF Info%s"
            % (" [default]" if cls.DEFAULT_VALUES["read_exif_info"] else ""),
        )
        parser.add_argument(
            "--no-exif",
            dest="read_exif_info",
            action="store_false",
            help="Do not parse EXIF Info%s"
            % (" [default]" if not cls.DEFAULT_VALUES["read_exif_info"] else ""),
        )
        parser.add_argument(
            "--icc",
            dest="read_icc_info",
            action="store_true",
            default=cls.DEFAULT_VALUES["read_icc_info"],
            help="Parse ICC Info%s"
            % (" [default]" if cls.DEFAULT_VALUES["read_icc_info"] else ""),
        )
        parser.add_argument(
            "--no-icc",
            dest="read_icc_info",
            action="store_false",
            help="Do not parse ICC Info%s"
            % (" [default]" if not cls.DEFAULT_VALUES["read_icc_info"] else ""),
        )
        parser.add_argument(
            "--average-results",
            dest="average_results",
            action="store_true",
            default=cls.DEFAULT_VALUES["average_results"],
            help="Provide averaged results%s"
            % (" [default]" if cls.DEFAULT_VALUES["average_results"] else ""),
        )
        parser.add_argument(
            "--no-average-results",
            dest="average_results",
            action="store_false",
            help="Do not provide averaged results%s"
            % (" [default]" if not cls.DEFAULT_VALUES["average_results"] else ""),
        )
        parser.add_argument(
            "--qpextract-bin",
            action="store",
            type=str,
            dest="qpextract_bin",
            default=cls.DEFAULT_VALUES["qpextract_bin"],
            help="Path to the qpextract bin",
        )
        parser.add_argument(
            "--isobmff-parser",
            action="store",
            type=str,
            dest="isobmff_parser",
            default=cls.DEFAULT_VALUES["isobmff_parser"],
            help="Path to the isobmff-parser bin",
        )
        parser.add_argument(
            "--h265nal-parser",
            action="store",
            type=str,
            dest="h265nal_parser",
            default=cls.DEFAULT_VALUES["h265nal_parser"],
            help="Path to the h265nal NALU parser bin",
        )

    def get(self, key):
        return self.config_dict.get(key, self.DEFAULT_VALUES[key])

    def set(self, key, val):
        self.config_dict[key] = val


GNU_TIME_DEFAULT_KEY_DICT = {
    "Command being timed": "command",
    "User time (seconds)": "usertime",
    "System time (seconds)": "systemtime",
    "Percent of CPU this job got": "cpu",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed",
    "Average shared text size (kbytes)": "avgtext",
    "Average unshared data size (kbytes)": "avgdata",
    "Average stack size (kbytes)": "avgstack",
    "Average total size (kbytes)": "avgtotal",
    "Maximum resident set size (kbytes)": "maxrss",
    "Average resident set size (kbytes)": "avgrss",
    "Major (requiring I/O) page faults": "major_pagefaults",
    "Minor (reclaiming a frame) page faults": "minor_pagefaults",
    "Voluntary context switches": "voluntaryswitches",
    "Involuntary context switches": "involuntaryswitches",
    "Swaps": "swaps",
    "File system inputs": "fileinputs",
    "File system outputs": "fileoutputs",
    "Socket messages sent": "socketsend",
    "Socket messages received": "socketrecv",
    "Signals delivered": "signals",
    "Page size (bytes)": "page_size",
    "Exit status": "status",
}


GNU_TIME_DEFAULT_VAL_TYPE = {
    "int": [
        "avgtext",
        "avgdata",
        "avgstack",
        "avgtotal",
        "maxrss",
        "avgrss",
        "major_pagefaults",
        "minor_pagefaults",
        "voluntaryswitches",
        "involuntaryswitches",
        "swaps",
        "fileinputs",
        "fileoutputs",
        "socketsend",
        "socketrecv",
        "signals",
        "page_size",
        "status",
        "usersystemtime",
    ],
    "float": [
        "usertime",
        "systemtime",
    ],
    "timedelta": [
        "elapsed",
    ],
    "percent": [
        "cpu",
    ],
}


def gnu_time_parse(gnu_time_str, logfd, debug):
    gnu_time_stats = {}
    for line in gnu_time_str.split("\n"):
        if not line:
            # empty line
            continue
        # check if we know the line
        line = line.strip()
        for key1, key2 in GNU_TIME_DEFAULT_KEY_DICT.items():
            if line.startswith(key1):
                break
        else:
            # unknown key
            print(f"warn: unknown gnutime line: {line}", file=logfd)
            continue
        val = line[len(key1) + 1 :].strip()
        # fix val type
        if key2 in GNU_TIME_DEFAULT_VAL_TYPE["int"]:
            val = int(val)
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["float"]:
            val = float(val)
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["percent"]:
            val = float(val[:-1])
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["timedelta"]:
            # '0:00.02'
            timedelta_re = r"((?P<min>\d+):(?P<sec>\d+).(?P<centisec>\d+))"
            res = re.search(timedelta_re, val)
            timedelta_sec = int(res.group("min")) * 60 + int(res.group("sec"))
            timedelta_centisec = int(res.group("centisec"))
            timedelta_sec += timedelta_centisec / 100.0
            val = timedelta_sec
        gnu_time_stats[key2] = val
    gnu_time_stats["usersystemtime"] = (
        gnu_time_stats["usertime"] + gnu_time_stats["systemtime"]
    )
    return gnu_time_stats
