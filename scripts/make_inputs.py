#!/usr/bin/env python3
"""Write one prediction config per overlapping window across whole chromosomes.

Each window is WINDOW bp wide and starts STEP = WINDOW - 2*PADDING bp after the
previous one, so consecutive windows overlap by 2*PADDING. The overlap exists
because a model's prediction is unreliable near the edges of its input: the
middle STEP bp of each window ("trusted") tile the chromosome exactly, and the
overlapping edges are reconciled later by ``merge_bigwigs.py``.

Alongside the JSONs an intervals manifest is written, so the merge step knows
each prediction's coordinates without having to parse file names.

Usage::

    python make_inputs.py chrom.sizes -c 0 1 --output-dir predictions/WG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

#: Model input width, in bases. Fixed by the model architecture.
WINDOW = 196_608

#: Unreliable margin at each edge of a window, in bases.
PADDING = 40_960

#: Distance between consecutive window starts, so the trusted middles abut.
STEP = WINDOW - 2 * PADDING  # 114,688

#: Prediction settings that are the same for every window.
JSON_CONSTANTS = {
    # Peaks are only used for explanations (is_explained is False). With
    # get_peaks True, pipeline.py writes the bigwigs and then crashes in
    # find_peaks ("`x` must be a 1-D array"), so Snakemake would mark every
    # window as failed and delete its bigwigs. pipeline.py requires
    # additional_peaks whenever get_peaks is False, so pass an empty list.
    "get_peaks": False,
    "additional_peaks": [],
    "merged": False,
    "is_explained": False,
    "save_tensors": False,
    "save_sequences": False,
}

MANIFEST_COLUMNS = ["name", "chrom", "start", "end",
                    "trusted_start", "trusted_end", "json"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "chrom_sizes", type=Path,
        help="tab-separated file of chromosome name and length",
    )
    parser.add_argument(
        "-c", "--cell-types", type=int, nargs="+", required=True, metavar="ID",
        help="cell type ids to predict, e.g. -c 0 1 2",
    )
    parser.add_argument(
        "--chroms", nargs="*", default=None, metavar="CHROM",
        help="which chromosomes to cover (default: every one in chrom_sizes)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="where each window's prediction folder goes, as "
             "<output-dir>/<chrom>_<start>_<end>",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("08_Pipeline_predictions"),
        help="folder holding the model script and genome files "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "-m", "--model", default="GFIO/Human_Ery",
        help="HFmodel to predict with (default: %(default)s)",
    )
    parser.add_argument(
        "--json-dir", type=Path, default=Path("07_JSONs"),
        help="where the prediction configs go (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="intervals table for the merge step "
             "(default: <json-dir>/intervals.tsv)",
    )
    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = args.json_dir / "intervals.tsv"
    return args


def read_chrom_sizes(path: Path, chroms: list[str] | None) -> dict[str, int]:
    """Read name/length pairs, optionally restricted to `chroms`."""
    table = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1],
                        names=["chrom", "size"], dtype={"chrom": str})
    sizes = dict(zip(table["chrom"], table["size"].astype(int)))
    if chroms is None:
        return sizes

    missing = [c for c in chroms if c not in sizes]
    if missing:
        raise KeyError(f"not in {path}: {missing}")
    return {c: sizes[c] for c in chroms}


def get_intervals(chrom_size: int) -> list[tuple[int, int]]:
    """Overlapping windows covering one chromosome, as 1-based (start, end).

    `end = start + WINDOW` keeps the arithmetic of the original script. No window
    extends past the chromosome: the original's loop broke on the *trusted* end
    rather than the window end, so the last window could overshoot by up to
    PADDING, and a chromosome shorter than WINDOW produced a negative start.
    """
    if chrom_size < WINDOW:
        return []

    # NOTE on the convention: `end = start + WINDOW` is kept from the original
    # script, which makes each interval WINDOW + 1 bases long if `end` is treated
    # as inclusive, and consecutive overlaps 2*PADDING + 1 rather than 2*PADDING.
    # That is harmless if pipeline.py treats `end` as exclusive; if it treats it
    # as inclusive, change this to `start + WINDOW - 1` and the windows will be
    # exactly WINDOW bases with trusted middles that abut rather than sharing a base.
    intervals = []
    start = 1
    while start + WINDOW <= chrom_size:
        intervals.append((start, start + WINDOW))
        start += STEP

    # Cover the tail with one final window flush against the chromosome end.
    last_start = chrom_size - WINDOW
    if not intervals or intervals[-1][0] < last_start:
        intervals.append((last_start, chrom_size))
    return intervals


def trusted_region(start: int, end: int) -> tuple[int, int]:
    """The part of a window far enough from both edges to be relied on."""
    return start + PADDING, end - PADDING


def window_name(chrom: str, start: int, end: int) -> str:
    return f"{chrom}_{start}_{end}"


def build_config(chrom: str, start: int, end: int, args: argparse.Namespace) -> dict:
    """The prediction config for one window."""
    name = window_name(chrom, start, end)
    return {
        "chromosome": chrom,
        "cell_types": args.cell_types,
        "start": start,
        "end": end,
        **JSON_CONSTANTS,
        "output_folder": str(args.output_dir / name),
        "input_folder": str(args.input_dir),
        "tag": name,
        "HFmodel": args.model,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sizes = read_chrom_sizes(args.chrom_sizes, args.chroms)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.json_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for chrom, size in sizes.items():
        intervals = get_intervals(size)
        if not intervals:
            print(
                f"WARNING: {chrom} is {size:,} bp, shorter than the {WINDOW:,} bp "
                "model input, so it is skipped",
                file=sys.stderr,
            )
            continue

        trusted = [trusted_region(s, e) for s, e in intervals]
        # The outermost windows have no neighbour to cover their padding, so they
        # keep it: at a chromosome end there is no better prediction to be had.
        # Without this, a trusted-mode merge leaves PADDING bases empty at each end.
        trusted[0] = (1, trusted[0][1])
        trusted[-1] = (trusted[-1][0], size)

        for (start, end), (t_start, t_end) in zip(intervals, trusted):
            name = window_name(chrom, start, end)
            path = args.json_dir / f"{name}.json"
            path.write_text(json.dumps(build_config(chrom, start, end, args), indent=2))
            rows.append((name, chrom, start, end, t_start, t_end, str(path)))
        print(f"{chrom}: {len(intervals)} windows over {size:,} bp")

    if not rows:
        print("ERROR: no windows generated", file=sys.stderr)
        return 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(
        args.manifest, sep="\t", index=False
    )
    print(f"{len(rows)} windows total -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
