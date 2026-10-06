#!/usr/bin/env python3
"""Merge many prediction bigwigs into one, in a single chunked pass.

Two ways of reconciling the overlap between consecutive windows:

``--mode mean`` (default)
    Average wherever windows overlap. Matches the original pairwise script.

``--mode trusted``
    Take only each window's trusted middle, PADDING bases in from both edges.
    Those middles tile the chromosome exactly, so nothing has to be averaged and
    no unreliable edge is used. Needs a manifest.

Memory is bounded by the chunk size, not the chromosome: two float arrays of
``--chunk`` bases regardless of how many files are being merged. A pairwise
chain over a whole chromosome needs tens of GB and re-reads everything once per
merge.

Usage::

    python merge_bigwigs.py out.bw --inputs a.bw b.bw c.bw
    python merge_bigwigs.py out.bw --inputs *.bw --manifest intervals.tsv --mode trusted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output", type=Path, help="bigwig to write")
    parser.add_argument(
        "--inputs", type=Path, nargs="+", required=True, help="bigwigs to merge"
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="intervals.tsv from make_inputs.py. Restricts each file to the "
             "region it predicted, which avoids reading files that cannot "
             "contribute. Required by --mode trusted.",
    )
    parser.add_argument(
        "--mode", choices=["mean", "trusted"], default="mean",
        help="how to reconcile overlaps (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk", type=int, default=5_000_000, metavar="BASES",
        help="bases held in memory at once (default: %(default)s)",
    )
    parser.add_argument(
        "--chroms", nargs="*", default=None, help="only these chromosomes"
    )
    args = parser.parse_args(argv)
    if args.mode == "trusted" and args.manifest is None:
        parser.error("--mode trusted needs --manifest")
    return args


def read_manifest(path: Path, mode: str) -> dict[str, tuple[int, int]]:
    """Map window name to the 0-based half-open span each file contributes."""
    table = pd.read_csv(path, sep="\t")
    if mode == "trusted":
        starts, ends = table["trusted_start"], table["trusted_end"]
    else:
        starts, ends = table["start"], table["end"]
    # The manifest is 1-based inclusive; bigwig coordinates are 0-based.
    return {
        name: (int(s) - 1, int(e))
        for name, s, e in zip(table["name"], starts, ends)
    }


def span_for(path: Path, spans: dict[str, tuple[int, int]] | None) -> tuple[int, int] | None:
    """The span a file contributes, by matching its name against the manifest."""
    if spans is None:
        return None
    stem = path.stem
    if stem in spans:
        return spans[stem]
    # Predictions usually sit in <window name>/... rather than being named for it.
    for part in reversed(path.parts):
        if part in spans:
            return spans[part]
    return None


def collect_chrom_sizes(paths: list[Path], chroms: list[str] | None) -> dict[str, int]:
    """Longest length seen for each chromosome across all inputs.

    Files are opened one at a time: a chromosome can have ~2,000 windows, more
    than the usual limit of 1,024 open files per process.
    """
    sizes: dict[str, int] = {}
    for path in paths:
        handle = pyBigWig.open(str(path))
        try:
            for chrom, length in handle.chroms().items():
                sizes[chrom] = max(sizes.get(chrom, 0), int(length))
        finally:
            handle.close()
    if chroms is not None:
        missing = [c for c in chroms if c not in sizes]
        if missing:
            raise KeyError(f"not present in any input: {missing}")
        sizes = {c: sizes[c] for c in chroms}
    return sizes


def merge_chunk(
    sources: list[tuple], chrom: str, start: int, stop: int
) -> np.ndarray:
    """Mean across every source covering this block, NaN where none does.

    Accumulating a running sum and count avoids holding one array per file, which
    is what makes the memory independent of how many files are merged.

    Each file is opened only for the blocks it can reach and closed straight away,
    so the number of open files stays small however many windows are merged.
    """
    width = stop - start
    total = np.zeros(width, dtype=np.float64)
    count = np.zeros(width, dtype=np.int32)

    for path, span in sources:
        if span is not None:
            # Skip a file that cannot reach this block at all.
            if span[1] <= start or span[0] >= stop:
                continue
        handle = pyBigWig.open(str(path))
        try:
            if chrom not in handle.chroms():
                continue
            values = handle.values(chrom, start, stop, numpy=True)
        finally:
            handle.close()
        if span is not None:
            # Ignore anything outside the region this file is allowed to supply.
            keep = np.zeros(width, dtype=bool)
            lo, hi = max(span[0], start) - start, min(span[1], stop) - start
            keep[lo:hi] = True
            values = np.where(keep, values, np.nan)

        present = ~np.isnan(values)
        if not present.any():
            continue
        total[present] += values[present]
        count[present] += 1

    merged = np.full(width, np.nan, dtype=np.float64)
    seen = count > 0
    merged[seen] = total[seen] / count[seen]
    return merged


def write_runs(out, chrom: str, offset: int, values: np.ndarray) -> int:
    """Write the non-NaN parts of a block as runs of equal value.

    Predictions are piecewise constant, so collapsing runs keeps the output small
    and avoids building a Python list per base, which is what makes the original
    approach fall over on a whole chromosome.
    """
    present = ~np.isnan(values)
    if not present.any():
        return 0

    # Boundaries where coverage starts/stops or the value changes.
    changed = np.empty(len(values), dtype=bool)
    changed[0] = True
    changed[1:] = (values[1:] != values[:-1]) | (present[1:] != present[:-1])
    changed[1:] |= np.isnan(values[1:]) != np.isnan(values[:-1])

    edges = np.flatnonzero(changed)
    run_starts = edges
    run_ends = np.append(edges[1:], len(values))

    keep = present[run_starts]
    run_starts, run_ends = run_starts[keep], run_ends[keep]
    if not len(run_starts):
        return 0

    out.addEntries(
        [chrom] * len(run_starts),
        (run_starts + offset).tolist(),
        ends=(run_ends + offset).tolist(),
        values=values[run_starts].tolist(),
    )
    return len(run_starts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    spans = read_manifest(args.manifest, args.mode) if args.manifest else None
    # (path, span) pairs; files are opened only while they are being read
    sources = [(path, span_for(path, spans)) for path in args.inputs]

    if spans is not None:
        unmatched = sum(1 for _, span in sources if span is None)
        if unmatched:
            print(
                f"WARNING: {unmatched} input(s) are not in the manifest and "
                "will be read in full",
                file=sys.stderr,
            )

    try:
        sizes = collect_chrom_sizes(args.inputs, args.chroms)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = pyBigWig.open(str(args.output), "w")
    out.addHeader([(c, s) for c, s in sizes.items()])

    written = 0
    try:
        for chrom, length in sizes.items():
            for start in range(0, length, args.chunk):
                stop = min(start + args.chunk, length)
                merged = merge_chunk(sources, chrom, start, stop)
                written += write_runs(out, chrom, start, merged)
    finally:
        out.close()

    print(f"{len(args.inputs)} file(s) -> {args.output} ({written:,} intervals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
