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
* Calculate result values (size, bpp, quality using vmaf/psnr/ssim/ssimulacra2).
* Calculate averages for the different quality values.
"""


import argparse
import importlib
import itertools
import json
import math
import os
import pandas as pd
import re
import shutil
import sys
import tempfile

itools_analysis = importlib.import_module("itools-analysis")
itools_common = importlib.import_module("itools-common")
itools_filter = importlib.import_module("itools-filter")
itools_heif = importlib.import_module("itools-heif")
itools_io = importlib.import_module("itools-io")
itools_jpeg = importlib.import_module("itools-jpeg")
itools_jxl = importlib.import_module("itools-jxl")
itools_version = importlib.import_module("itools-version")

VMAF_DEF_MODEL = "/usr/share/model/vmaf_v0.6.1.json"
VMAF_NEG_MODEL = "/usr/share/model/vmaf_v0.6.1neg.json"
VMAF_4K_MODEL = "/usr/share/model/vmaf_4k_v0.6.1.json"


DEFAULT_QUALITIES = [25, 75, 85, 95, 96, 97, 97.5, 98, 99]
DEFAULT_QUALITY_LIST = sorted(set(list(range(0, 101, 10)) + DEFAULT_QUALITIES))


default_values = {
    "debug": 0,
    "dry_run": False,
    "horizontal_alignment": None,
    "vertical_alignment": None,
    "quality_list": ",".join(str(v) for v in DEFAULT_QUALITY_LIST),
    "codec": "x265",
    "analysis": False,
    "encoded_infile": None,
    "encoded_rotate": 0,
    "workdir": tempfile.gettempdir(),
    "infile_list": None,
    "outfile": None,
}

COLUMN_LIST = [
    "infile",
    "outfile",
    "width",
    "height",
    "codec",
    "quality",
    "encoded_size",
    "encoded_bpp",
    "psnr",
    "ssim",
    "ssimulacra2",
]


def get_video_dimensions(infile, debug):
    command = f"ffprobe -v 0 -of csv='p=0' -select_streams v:0 -show_entries stream=width,height {infile}"
    returncode, out, err = itools_common.run(command, debug=debug)
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
]


def vmaf_parse(stdout, stderr, vmaf_file):
    """Parse log/output and return quality score"""
    assert os.path.isfile(vmaf_file), f"error: {stdout=}\n{stderr=}"
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
    vmaf_file = tempfile.NamedTemporaryFile(prefix="itools.vmaf.", suffix=".json").name
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
        "feature=name=psnr|name=psnr_hvs|name=float_ssim|name=float_ms_ssim|name=cambi'"
        " -f null -"
    )
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. parse the output
    return vmaf_parse(out, err, vmaf_file)


def psnr_get(distorted, reference, debug):
    psnr_file = tempfile.NamedTemporaryFile(prefix="itools.psnr.", suffix=".txt").name
    # 1. calculate the score
    command = (
        f"{itools_common.FFMPEG_SILENT} -i {distorted} -i {reference} "
        "-filter_complex "
        f'"psnr=stats_file={psnr_file}" '
        f"-f null - 2>&1"
    )
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. parse the output
    return psnr_parse(out)


def ssim_get(distorted, reference, debug):
    ssim_file = tempfile.NamedTemporaryFile(prefix="itools.ssim.", suffix=".txt").name
    # 1. calculate the score
    command = (
        f"{itools_common.FFMPEG_SILENT} -i {distorted} -i {reference} "
        "-filter_complex "
        f'"ssim=stats_file={ssim_file}" '
        f"-f null - 2>&1"
    )
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. parse the output
    return ssim_parse(out)


def ssimulacra2_get(distorted, reference, debug):
    # 0. ssimulacra2 only accepts png as inputs
    reference_png = f"{reference}.png"
    command = f"{itools_common.FFMPEG_SILENT} -i {reference} {reference_png}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    distorted_png = f"{distorted}.png"
    command = f"{itools_common.FFMPEG_SILENT} -i {distorted} {distorted_png}"
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"
    # 1. calculate the score
    command = f"ssimulacra2 {reference_png} {distorted_png}"
    returncode, out, err = itools_common.run(command, debug=debug)
    if returncode == 127:
        # no ssimulacra2 available
        print(f"error: no ssimulacra2 available\n{err}")
        return 0.0
    assert returncode == 0, f"error: {out = } {err = }"
    # 2. parse the output
    return float(out.decode("ascii"))


def escape_float(f):
    return str(int(f)) if f.is_integer() else str(f).replace(".", "_")


# encoding backends
# 1. heif-enc
def heif_enc_encode_fun(infile, width, height, codec, quality, outfile, debug):
    return itools_heif.encode_heif(infile, codec, quality, outfile, debug)


# 2. libjpeg
def libjpeg_encode_fun(infile, width, height, codec, quality, outfile, debug):
    return itools_jpeg.encode_libjpeg(infile, quality, outfile, debug)


def jpegli_encode_fun(infile, width, height, codec, quality, outfile, debug):
    return itools_jpeg.encode_jpegli(infile, quality, outfile, debug)


# 3. jxl
def jxl_encode_fun(infile, width, height, codec, quality, outfile, debug):
    return itools_jxl.encode_jxl(infile, quality, outfile, debug)


def jxl_decode_fun(infile, outfile, debug):
    return itools_jxl.decode_jxl(infile, outfile, debug)


# TODO(chema): use better mechanism here
# codec: (extension, (config_fun, init_fun, encode_fun, fini_fun), (horizontal_alignment, vertical_alignment))
# TODO(chema): add (decoder)
CODEC_CHOICES = {
    "x265": (".heic", (None, None, heif_enc_encode_fun, None), (None, None)),
    "kvazaar": (".heic", (None, None, heif_enc_encode_fun, None), (None, None)),
    "aom": (".avif", (None, None, heif_enc_encode_fun, None), (None, None)),
    "svt": (".avif", (None, None, heif_enc_encode_fun, None), (None, None)),
    "libjpeg": (".jpeg", (None, None, libjpeg_encode_fun, None), (None, None)),
    "jxl": (".jxl", (None, None, jxl_encode_fun, None), (None, None)),
    "jpegli": (".jpeg", (None, None, jpegli_encode_fun, None), (None, None)),
    "openjpeg": (".jpeg2000", (None, None, heif_enc_encode_fun, None), (None, None)),
    "empty": (None, (None, None, None, None), (None, None)),
}


def process_file(
    infile,
    codec,
    quality_list,
    horizontal_alignment,
    vertical_alignment,
    workdir,
    codec_choices,
    config_dict,
    debug,
    encoded_infile=None,
    encoded_rotate=None,
):
    df = None
    # 1. select a codec
    extension, (config_fun, init_fun, encode_fun, fini_fun), (ha, va) = codec_choices[
        codec
    ]
    horizontal_alignment = (
        horizontal_alignment if horizontal_alignment is not None else ha
    )
    vertical_alignment = vertical_alignment if vertical_alignment is not None else va
    # 2. crop input to alignment in vertical and horizontal
    width, height = get_video_dimensions(infile, debug)
    width = (
        width
        if (horizontal_alignment is None or width % horizontal_alignment == 0)
        else (horizontal_alignment * math.floor(width / horizontal_alignment))
    )
    height = (
        height
        if (vertical_alignment is None or height % vertical_alignment == 0)
        else (vertical_alignment * math.floor(height / vertical_alignment))
    )
    ref_basename = f"{os.path.basename(infile)}.{width}x{height}.codec_{codec}.y4m"
    ref_path = os.path.join(workdir, ref_basename)
    command = f'{itools_common.FFMPEG_SILENT} -i {infile} -vf "crop={width}:{height}:(iw-ow)/2:(ih-oh)/2" {ref_path}'
    returncode, out, err = itools_common.run(command, debug=debug)
    assert returncode == 0, f"error: {out = } {err = }"

    # 3. prepare the input (reference) file
    if extension is None and encoded_infile is not None:
        extension = os.path.splitext(encoded_infile)[-1]
    exp_path = init_fun(ref_path, workdir, debug) if init_fun is not None else ref_path
    exp_basename = os.path.basename(exp_path)
    for quality in quality_list:
        # 4. encode the file
        enc_path = os.path.join(
            workdir,
            f"{exp_basename}.quality_{escape_float(quality)}{extension}",
        )
        if codec == "empty":
            # copy the encoded file to the encoded path
            if debug > 0:
                print(f"running $ cp {encoded_infile} {enc_path}")
            shutil.copyfile(encoded_infile, enc_path)
        else:
            stats = encode_fun(exp_path, width, height, codec, quality, enc_path, debug)
        # 5. calculate the encoded size
        encoded_size = os.path.getsize(enc_path)
        encoded_bpp = (8 * encoded_size) / (width * height)
        # 6. decode encoded file
        distorted_path = f"{enc_path}.y4m"
        enc_extension = os.path.splitext(enc_path)[-1]
        if enc_extension in (".heic", ".avif", ".jp2", ".j2k"):
            ref_colorrange = itools_io.read_colorrange(ref_path, debug)
            itools_heif.decode_heif(
                enc_path,
                distorted_path,
                config_dict,
                output_colorrange=ref_colorrange,
                debug=debug,
            )
        elif enc_extension in (".jpg", ".jpeg"):
            # copy the encoded file to the distorted path
            itools_jpeg.decode_jpeg(enc_path, distorted_path, debug)
        elif enc_extension in (".y4m",):
            # copy the encoded file to the distorted path
            if debug > 0:
                print(f"running $ cp {encoded_infile} {distorted_path}")
            shutil.copyfile(encoded_infile, distorted_path)
        elif enc_extension in (".jxl",):
            # copy the encoded file to the distorted path
            jxl_decode_fun(enc_path, distorted_path, debug)
        else:
            raise AssertionError(f"cannot decode file {enc_path}")

        if encoded_rotate is not None and encoded_rotate != 0:
            # rotate the encoded output
            iinfo = None
            proc_color = itools_common.ProcColor.yvu
            if debug > 0:
                print(
                    f"running $ itools-filter.py --filter rotate --rotate-angle {encoded_rotate} -i {distorted_path} -o {distorted_path}"
                )
            itools_filter.rotate_image(
                distorted_path,
                encoded_rotate,
                distorted_path,
                iinfo,
                proc_color,
                config_dict,
                debug,
            )
        # 7. analyze encoded file
        vmaf_def = vmaf_get(distorted_path, ref_path, debug, VMAF_DEF_MODEL)
        vmaf_neg = vmaf_get(distorted_path, ref_path, debug, VMAF_NEG_MODEL)
        vmaf_4k = vmaf_get(distorted_path, ref_path, debug, VMAF_4K_MODEL)
        psnr = psnr_get(distorted_path, ref_path, debug)
        ssim = ssim_get(distorted_path, ref_path, debug)
        ssimulacra2 = ssimulacra2_get(distorted_path, ref_path, debug)
        # 8. gather results
        if df is None:
            vmaf_column_list = list(
                f"vmaf:{key if key != 'vmaf' else 'neg'}" for key in vmaf_neg.keys()
            )
            stats_column_list = list(f"stats:{key}" for key in stats.keys())
            df = pd.DataFrame(
                columns=COLUMN_LIST
                + [
                    "vmaf:def",
                    "vmaf:4k",
                ]
                + vmaf_column_list
                + stats_column_list
            )
        df.loc[df.size] = (
            infile,
            enc_path,
            width,
            height,
            codec,
            quality,
            encoded_size,
            encoded_bpp,
            psnr,
            ssim,
            ssimulacra2,
            vmaf_def["vmaf"],
            vmaf_4k["vmaf"],
            *list(vmaf_neg.values()),
            *list(stats.values()),
        )

    # 9. clean up after yourself
    if fini_fun is not None:
        fini_fun(exp_basename, debug)
    return df


def get_average_results(df):
    # import the results
    new_df = pd.DataFrame(columns=list(df.columns.values))
    for codec, quality in itertools.product(
        list(df["codec"].unique()), sorted(list(df["quality"].unique()))
    ):
        # select interesting data
        tmp_fd = df[(df["codec"] == codec) & (df["quality"] == quality)]
        # start with empty data
        derived_dict = {key: None for key in list(df.columns.values)}
        derived_dict["infile"] = "average"
        # copy a few columns
        for col in ("quality", "codec"):
            derived_dict[col] = tmp_fd[col].values[0]
        # average a few columns
        vmaf_keys = list(key for key in df.columns.values if key.startswith("vmaf:"))
        qpwy_keys = list(key for key in df.columns.values if key.startswith("qpwy:"))
        qpwcb_keys = list(key for key in df.columns.values if key.startswith("qpwcb:"))
        qpwcr_keys = list(key for key in df.columns.values if key.startswith("qpwcr:"))
        ctu_keys = list(key for key in df.columns.values if key.startswith("ctu:"))
        stats_keys = list(key for key in df.columns.values if key.startswith("stats:"))
        mean_keys = (
            [
                "width",
                "height",
                "encoded_size",
                "encoded_bpp",
                "psnr",
                "ssim",
                "ssimulacra2",
            ]
            + vmaf_keys
            + qpwy_keys
            + qpwcb_keys
            + qpwcr_keys
            + ctu_keys
            + stats_keys
        )
        for key in mean_keys:
            derived_dict[key] = tmp_fd[key].mean()
        new_df.loc[new_df.size] = list(derived_dict.values())
    return new_df


def process_data(
    infile_list,
    codec,
    quality_list,
    horizontal_alignment,
    vertical_alignment,
    workdir,
    analysis,
    config_dict,
    codec_choices,
    outfile_csv,
    debug,
    encoded_infile=None,
    encoded_rotate=None,
):
    df = None
    # 1. get a codec list (if present)
    codec_list = codec.split(",") if codec != "all" else codec_choices.keys()
    codec_valid_list = list(codec_choices.keys())
    if not set(codec_list) <= set(codec_valid_list):
        unknown_codec_list = [
            codec for codec in codec_list if codec not in codec_valid_list
        ]
        raise AssertionError(f"unknown codec(s): {unknown_codec_list}")
    quality_list = list(float(v) for v in quality_list.split(","))
    results = ()
    for codec, infile in itertools.product(codec_list, infile_list):
        # 2. configure codec/device
        _, (config_fun, _, _, _), (_, _) = codec_choices[codec]
        if config_fun is not None:
            config_fun(debug)
        # 3. run the experiments
        tmp_df = process_file(
            infile,
            codec,
            quality_list,
            horizontal_alignment,
            vertical_alignment,
            workdir,
            codec_choices,
            config_dict,
            debug,
            encoded_infile=encoded_infile,
            encoded_rotate=encoded_rotate,
        )
        df = tmp_df if df is None else pd.concat([df, tmp_df], ignore_index=True)
    # 3. reindex per-file dataframe
    df = df.reindex()
    # 4. add per-output analysis
    if analysis:
        df_analysis = None
        for outfile in df.outfile:
            df_tmp_analysis = itools_analysis.get_components(
                outfile,
                roi=((None, None), (None, None)),
                roi_dump=None,
                config_dict=config_dict,
                debug=debug,
            )
            df_analysis = (
                df_tmp_analysis
                if df_analysis is None
                else pd.concat([df_analysis, df_tmp_analysis])
            )
        df = pd.merge(df, df_analysis, left_on="outfile", right_on="filename")
    average_results = config_dict.get("average_results")
    if average_results:
        # 5. get average results
        derived_df = get_average_results(df)
        # TODO(chemag): fix this warning
        # Likely cause is that some of the derived_df columns are dtype object
        # ('dtype("O")') but have only null values (e.g. "ymean" column).
        # \ref https://stackoverflow.com/a/60802429
        # df = pd.concat([df, derived_df], ignore_index=True)
        df = pd.concat([df, derived_df], ignore_index=True, axis=0)
    # 6. write the results
    df.to_csv(outfile_csv, index=False)


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
    codec_list = list(codec_choices.keys()) + [
        "all",
    ]
    parser.add_argument(
        "--codec",
        action="store",
        type=str,
        dest="codec",
        default=default_values["codec"],
        metavar="[%s]" % (" | ".join(codec_list)),
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
        "--analysis",
        action="store_true",
        dest="analysis",
        default=default_values["analysis"],
        help="Full analysis",
    )
    parser.add_argument(
        "--noanalysis",
        action="store_false",
        dest="analysis",
        help="No full analysis",
    )
    itools_common.Config.set_parser_options(parser)
    parser.add_argument(
        "--encoded-infile",
        action="store",
        dest="encoded_infile",
        default=default_values["encoded_infile"],
        metavar="ENCODED-INFILE",
        help="use encoded filename (for empty codec)",
    )
    parser.add_argument(
        "--encoded-rotate",
        action="store",
        type=int,
        dest="encoded_rotate",
        choices=itools_filter.ROTATE_ANGLE_LIST.keys(),
        default=default_values["encoded_rotate"],
        metavar="ENCODED-ROTATE",
        help="use encoded rotation (for empty codec)",
    )
    parser.add_argument(
        "--workdir",
        action="store",
        dest="workdir",
        type=str,
        default=default_values["workdir"],
        metavar="Work directory",
        help="work directory",
    )
    parser.add_argument(
        dest="infile_list",
        type=str,
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file-list",
        help="input file list",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main(argv, codec_choices=CODEC_CHOICES):
    # parse options
    options = get_options(argv, codec_choices)
    # set workdir
    if options.workdir is not None:
        os.makedirs(options.workdir, exist_ok=True)
        tempfile.tempdir = options.workdir
    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # create configuration
    config_dict = itools_common.Config.Create(options)
    # process infile
    process_data(
        options.infile_list,
        options.codec,
        options.quality_list,
        options.horizontal_alignment,
        options.vertical_alignment,
        options.workdir,
        options.analysis,
        config_dict,
        codec_choices,
        options.outfile,
        options.debug,
        encoded_infile=options.encoded_infile,
        encoded_rotate=options.encoded_rotate,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
