#!/usr/bin/env python3

"""itools-enctool.py module description.

This is a tool to test image encoders.

Image encoders supported:
* heif-enc with different formats: HEIC ("x265" and "kvazaar" engines),
  AVIF ("aom" and "svt" engines), and OpenJPEG ("jpeg2000" engine).

Features:
* Enforce the right alignment in the input (raw) images.
* Run experiments for different encoders.
* Test multiple input quality parameter values.
* Allow selection of input corpus.
* Calculate result values (size, bpp, quality using vmaf/psnr/ssim).
* Calculate averages for the different quality values.
"""


import argparse
import importlib
import json
import math
import os
import pandas as pd
import re
import subprocess
import sys
import tempfile

itools_common = importlib.import_module("itools-common")
itools_version = importlib.import_module("itools-version")


VMAF_DEF_MODEL = "/usr/share/model/vmaf_v0.6.1.json"
VMAF_NEG_MODEL = "/usr/share/model/vmaf_v0.6.1neg.json"
VMAF_4K_MODEL = "/usr/share/model/vmaf_4k_v0.6.1.json"


DEFAULT_QUALITIES = [25, 75, 85, 95, 96, 97, 97.5, 98, 99]
DEFAULT_QUALITY_LIST = sorted(set(list(range(0, 101, 10)) + DEFAULT_QUALITIES))
DEFAULT_HORIZONTAL_ALIGNMENT = 32
DEFAULT_VERTICAL_ALIGNMENT = 32


default_values = {
    "debug": 0,
    "dry_run": False,
    "horizontal_alignment": DEFAULT_HORIZONTAL_ALIGNMENT,
    "vertical_alignment": DEFAULT_VERTICAL_ALIGNMENT,
    "quality_list": ",".join(str(v) for v in DEFAULT_QUALITY_LIST),
    "codec": "heic",
    "tmpdir": tempfile.gettempdir(),
    "infile_list": None,
    "outfile": None,
}

COLUMN_LIST = [
    "infile",
    "width",
    "height",
    "codec",
    "quality",
    "encoded_size",
    "encoded_bpp",
    "psnr",
    "ssim",
]


def run(command, **kwargs):
    debug = kwargs.get("debug", 0)
    dry_run = kwargs.get("dry_run", False)
    env = kwargs.get("env", None)
    stdin = subprocess.PIPE if kwargs.get("stdin", False) else None
    bufsize = kwargs.get("bufsize", 0)
    universal_newlines = kwargs.get("universal_newlines", False)
    default_close_fds = True if sys.platform == "linux2" else False
    close_fds = kwargs.get("close_fds", default_close_fds)
    shell = type(command) in (type(""), type(""))
    if debug > 0:
        print("running $ %s" % command)
    if dry_run:
        return 0, b"stdout", b"stderr"
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
    # return results
    return returncode, out, err


def get_video_dimensions(infile, debug):
    command = f"ffprobe -v 0 -of csv='p=0' -select_streams v:0 -show_entries stream=width,height {infile}"
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    return [int(v) for v in out.decode("ascii").strip().split(",")]


VMAF_RE = "VMAF score: ([0-9.]*)"
PSNR_RE = "average:([0-9.]*)"
SSIM_RE = "SSIM Y:([0-9.]*)"

VMAF_METRIC_LIST = [
    "vmaf",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    "psnr_hvs",
    "psnr_hvs_y",
    "psnr_hvs_cb",
    "psnr_hvs_cr",
    "float_ssim",
    "float_ms_ssim",
    "cambi",
    "ciede2000",
]


def vmaf_parse(stdout, vmaf_file):
    """Parse log/output and return quality score"""
    assert os.path.isfile(vmaf_file), f"error: {stdout}"
    with open(vmaf_file, "rb") as fin:
        json_text = fin.read()
    json_dict = json.loads(json_text)
    out_dict = {}
    for metric in VMAF_METRIC_LIST:
        if metric in json_dict["pooled_metrics"]:
            out_dict[metric] = json_dict["pooled_metrics"][metric]["mean"]
    return out_dict


def ssim_parse(stdout):
    """Parse log/output and return quality score"""
    ssim = -1
    for line in stdout.decode("ascii").splitlines():
        match = re.search(SSIM_RE, line)
        if match:
            ssim = float(match.group(1))
            break
    return ssim


def psnr_parse(stdout):
    """Parse log/output and return quality score"""
    psnr = -1
    for line in stdout.decode("ascii").splitlines():
        match = re.search(PSNR_RE, line)
        if match:
            psnr = float(match.group(1))
            break
    return psnr


def vmaf_get(distorted, reference, debug, vmaf_model=None):
    vmaf_file = tempfile.NamedTemporaryFile(prefix="vmaf.", suffix=".json").name
    # 1. calculate the score
    command = (
        f"{itools_common.FFMPEG_SILENT} -i {distorted} -i {reference} "
        "-filter_complex "
        "'libvmaf="
    )
    if vmaf_model is not None:
        command += f"model=path={vmaf_model}:"
    command += (
        f"log_path={vmaf_file}:n_threads=16:log_fmt=json:"
        "feature=name=psnr|name=psnr_hvs|name=float_ssim|name=float_ms_ssim|name=cambi|name=ciede'"
        " -f null - 2>&1 "
    )
    returncode, out, err = run(command, debug=debug)
    # 2. parse the output
    return vmaf_parse(out, vmaf_file)


def psnr_get(distorted, reference, debug):
    psnr_file = "/tmp/psnr.txt"
    # 1. calculate the score
    command = (
        f"{itools_common.FFMPEG_SILENT} -i {distorted} -i {reference} "
        "-filter_complex "
        f'"psnr=stats_file={psnr_file}" '
        f"-f null - 2>&1"
    )
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. parse the output
    return psnr_parse(out)


def ssim_get(distorted, reference, debug):
    ssim_file = "/tmp/ssim.txt"
    # 1. calculate the score
    command = (
        f"{itools_common.FFMPEG_SILENT} -i {distorted} -i {reference} "
        "-filter_complex "
        f'"ssim=stats_file={ssim_file}" '
        f"-f null - 2>&1"
    )
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. parse the output
    return ssim_parse(out)


def escape_float(f):
    return str(int(f)) if f.is_integer() else str(f).replace(".", "_")


def heif_enc_encode_fun(
    infile_path, width, height, codec, quality, outfile_path, debug
):
    command = f"heif-enc {infile_path} -e {codec} -q {quality} {outfile_path}"
    returncode, out, err = run(command, debug=debug)


# TODO(chema): better value here
# codec: (extension, configure, init, main_fun, fini)  # (alignment, decoder)
CODEC_CHOICES = {
    "x265": ("heic", (None, None, heif_enc_encode_fun, None)),
    "kvazaar": ("heic", (None, None, heif_enc_encode_fun, None)),
    "aom": ("avif", (None, None, heif_enc_encode_fun, None)),
    "svt": ("avif", (None, None, heif_enc_encode_fun, None)),
    "openjpeg": ("jpeg2000", (None, None, heif_enc_encode_fun, None)),
}


def process_file(
    infile,
    codec,
    quality_list,
    horizontal_alignment,
    vertical_alignment,
    tmpdir,
    codec_choices,
    debug,
):
    df = None
    # 0. get input dimensions
    width, height = get_video_dimensions(infile, debug)
    # 1. crop input to alignment in vertical and horizontal
    width = (
        width
        if (width % horizontal_alignment == 0)
        else (horizontal_alignment * math.floor(width / horizontal_alignment))
    )
    height = (
        height
        if (height % vertical_alignment == 0)
        else (vertical_alignment * math.floor(height / vertical_alignment))
    )
    ref_basename = f"{os.path.basename(infile)}.{width}x{height}.y4m"
    ref_path = os.path.join(tmpdir, ref_basename)
    command = f'{itools_common.FFMPEG_SILENT} -i {infile} -vf "crop={width}:{height}:(iw-ow)/2:(ih-oh)/2" {ref_path}'
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. select a codec
    extension, (config_fun, init_fun, encode_fun, fini_fun) = codec_choices[codec]
    exp_path = init_fun(ref_path, tmpdir, debug) if init_fun is not None else ref_path
    exp_basename = os.path.basename(exp_path)
    for quality in quality_list:
        # 4. encode the file
        enc_path = os.path.join(
            tmpdir,
            f"{exp_basename}.codec_{codec}.quality_{escape_float(quality)}.{extension}",
        )
        encode_fun(exp_path, width, height, codec, quality, enc_path, debug)
        # 5. calculate the encoded size
        encoded_size = os.path.getsize(enc_path)
        encoded_bpp = (8 * encoded_size) / (width * height)
        # 6. decode encoded file
        distorted_path = f"{enc_path}.y4m"
        if codec in ("heic", "x265", "kvazaar", "aom", "svt", "openjpeg"):
            tmpy4m = tempfile.NamedTemporaryFile(prefix="xnova.", suffix=".y4m").name
            # decode the heic file
            command = f"heif-convert {enc_path} {tmpy4m}"
            returncode, out, err = run(command, debug=debug)
            assert returncode == 0, f"error: {out = } {err = }"
            # fix the color range
            command = f"{itools_common.FFMPEG_SILENT} -i {tmpy4m} -color_range full {distorted_path}"
            returncode, out, err = run(command, debug=debug)
            assert returncode == 0, f"error: {out = } {err = }"
        elif codec == "jpeg":
            command = f"{itools_common.FFMPEG_SILENT} -i {enc_path} {distorted_path}"
            # command = f"{itools_common.FFMPEG_SILENT} -i {enc_path} -pix_fmt yuv420p {distorted_path}"
            # command = f"{itools_common.FFMPEG_SILENT} -i {enc_path} -pix_fmt yuv420p -vf scale=out_range=full {distorted_path}"
            returncode, out, err = run(command, debug=debug)
            assert returncode == 0, f"error: {out = } {err = }"
        # 7. analyze encoded file
        vmaf_def = vmaf_get(distorted_path, ref_path, debug, VMAF_DEF_MODEL)
        vmaf_neg = vmaf_get(distorted_path, ref_path, debug, VMAF_NEG_MODEL)
        vmaf_4k = vmaf_get(distorted_path, ref_path, debug, VMAF_4K_MODEL)
        psnr = psnr_get(distorted_path, ref_path, debug)
        ssim = ssim_get(distorted_path, ref_path, debug)
        # 8. gather results
        if df is None:
            vmaf_column_list = list(
                f"vmaf:{key if key != 'vmaf' else 'neg'}" for key in vmaf_neg.keys()
            )
            df = pd.DataFrame(
                columns=COLUMN_LIST
                + [
                    "vmaf:def",
                    "vmaf:4k",
                ]
                + vmaf_column_list
            )
        df.loc[df.size] = (
            infile,
            width,
            height,
            codec,
            quality,
            encoded_size,
            encoded_bpp,
            psnr,
            ssim,
            vmaf_def["vmaf"],
            vmaf_4k["vmaf"],
            *list(vmaf_neg.values()),
        )

    # 9. clean up after yourself
    if fini_fun is not None:
        fini_fun(exp_basename, debug)
    return df


def get_derived_results(df):
    # import the results
    new_df = pd.DataFrame(columns=list(df.columns.values))
    for codec in list(df["codec"].unique()):
        for quality in sorted(list(df["quality"].unique())):
            # average a few values
            encoded_bpp = df[df["codec"] == codec][df["quality"] == quality][
                "encoded_bpp"
            ].mean()
            psnr = df[df["codec"] == codec][df["quality"] == quality]["psnr"].mean()
            ssim = df[df["codec"] == codec][df["quality"] == quality]["ssim"].mean()
            vmaf_keys = list(
                key for key in df.columns.values if key.startswith("vmaf:")
            )
            vmaf_dict = {
                key: df[df["codec"] == codec][df["quality"] == quality][key].mean()
                for key in vmaf_keys
            }
            infile = "average"
            width = df[df["codec"] == codec][df["quality"] == quality]["width"].mean()
            height = df[df["codec"] == codec][df["quality"] == quality]["height"].mean()
            encoded_size = df[df["codec"] == codec][df["quality"] == quality][
                "encoded_size"
            ].mean()
            new_df.loc[new_df.size] = (
                infile,
                width,
                height,
                codec,
                quality,
                encoded_size,
                encoded_bpp,
                psnr,
                ssim,
                *vmaf_dict.values(),
            )
    return new_df


def process_data(
    infile_list,
    codec,
    quality_list,
    horizontal_alignment,
    vertical_alignment,
    tmpdir,
    codec_choices,
    outfile,
    debug,
):
    df = None
    # 1. configure codec/device
    _, (config_fun, _, _, _) = codec_choices[codec]
    if config_fun is not None:
        config_fun(debug)
    # 2. run the experiments
    quality_list = list(float(v) for v in quality_list.split(","))
    results = ()
    for infile in infile_list:
        tmp_df = process_file(
            infile,
            codec,
            quality_list,
            horizontal_alignment,
            vertical_alignment,
            tmpdir,
            codec_choices,
            debug,
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
    # 3. get derived results
    tmp_df = get_derived_results(df)
    df = pd.concat([df, tmp_df], ignore_index=True)
    # 4. TODO(chema): aggregate results

    # 5. write the results
    df.to_csv(outfile, index=False)


def get_options(argv, codec_choices):
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
        action="version",
        version=itools_version.__version__,
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
        "--quality-list",
        action="store",
        type=str,
        dest="quality_list",
        default=default_values["quality_list"],
        help="Quality list (comma-separated list)",
    )
    parser.add_argument(
        "--codec",
        action="store",
        type=str,
        dest="codec",
        default=default_values["codec"],
        choices=codec_choices.keys(),
        metavar="[%s]"
        % (
            " | ".join(
                codec_choices.keys(),
            )
        ),
        help="codec arg",
    )
    parser.add_argument(
        "--horizontal-alignment",
        action="store",
        type=int,
        dest="horizontal_alignment",
        default=default_values["horizontal_alignment"],
        help='Horizontal alignment [default: {default_values["horizontal_alignment"]}]',
    )
    parser.add_argument(
        "--vertical-alignment",
        action="store",
        type=int,
        dest="vertical_alignment",
        default=default_values["vertical_alignment"],
        help='Vertical alignment [default: {default_values["vertical_alignment"]}]',
    )
    parser.add_argument(
        "--tmpdir",
        action="store",
        type=str,
        dest="tmpdir",
        default=default_values["tmpdir"],
        metavar="TMPDIR",
        help="temporal dir",
    )
    parser.add_argument(
        "infile_list",
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file-list",
        help="input file list",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="outfile",
        default=default_values["outfile"],
        metavar="OUTPUT",
        help="use OUTPUT filename",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    # parse options
    options = get_options(argv, CODEC_CHOICES)
    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # process infile
    process_data(
        options.infile_list,
        options.codec,
        options.quality_list,
        options.horizontal_alignment,
        options.vertical_alignment,
        options.tmpdir,
        CODEC_CHOICES,
        options.outfile,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
