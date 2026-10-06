"""Whole-genome predictions: tile each chromosome in overlapping windows, predict
each window on a GPU, then merge the windows back into one bigwig per cell type.

    snakemake -n                                     # dry run: list the jobs
    snakemake --profile profiles/slurm                # predict everything, then merge
    snakemake --profile profiles/slurm inputs_only    # just the configs
    snakemake --profile profiles/slurm per_chromosome # stop at one bigwig per chromosome

Consecutive windows overlap by 2 x padding, because a prediction is unreliable
near the edges of the model's input. `merge_mode` decides what happens there:
"mean" averages the two predictions, "trusted" keeps only the middle of each
window. Either way the chromosome ends up covered exactly once.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))
from make_inputs import PADDING, WINDOW, get_intervals, read_chrom_sizes, window_name

configfile: "config.yaml"

OUT = Path(config["output_dir"])
JSONS = Path(config["json_dir"])
MANIFEST = JSONS / "intervals.tsv"
CELLS = [str(c) for c in config["cell_types"]]

# Where one window's prediction lands. Check this against your pipeline's layout.
WINDOW_BIGWIG = OUT / "{name}" / config["bigwig_pattern"]

# Merged outputs. The per-chromosome ones sit under their own folder: with both
# at <out>/{something}/class_{cell}.bw, the {chrom} wildcard also matched the
# genome-wide file and whichever rule Snakemake picked first won.
CHROM_BIGWIG = OUT / "per_chromosome" / "{chrom}" / "class_{cell}.bw"
GENOME_BIGWIG = OUT / "class_{cell}.bw"   # final files sit directly in output_dir

# The window list is deterministic, so it is built here rather than discovered
# later: every rule's inputs are then known before anything runs.
SIZES = read_chrom_sizes(Path(config["chrom_sizes"]), config.get("chroms") or None)
WINDOWS = {
    chrom: [window_name(chrom, s, e) for s, e in get_intervals(size)]
    for chrom, size in SIZES.items()
}
WINDOWS = {chrom: names for chrom, names in WINDOWS.items() if names}

# Quick test: keep only `max_windows` consecutive windows from the middle of each
# chromosome (the middle avoids the N-filled telomeres; consecutive windows still
# exercise the overlap in the merge). 0 or unset = every window.
MAX_WINDOWS = int(config.get("max_windows") or 0)
if MAX_WINDOWS:
    for chrom, names in WINDOWS.items():
        first = max(0, len(names) // 2 - MAX_WINDOWS // 2)
        WINDOWS[chrom] = names[first:first + MAX_WINDOWS]
    print(f"TEST MODE: {MAX_WINDOWS} window(s) per chromosome", file=sys.stderr)

if not WINDOWS:
    raise ValueError(
        f"No chromosome in {config['chrom_sizes']} is at least {WINDOW:,} bp, "
        "so no window can be predicted."
    )
def check_existing_jsons():
    """Stop early if the JSONs on disk were written with different settings.

    Snakemake does not rewrite JSONs that already exist, but pipeline.py writes
    wherever its JSON says. After changing output_dir (or the model, cells, ...),
    stale JSONs would send every prediction to the old folder.
    """
    import json
    if not MANIFEST.exists():
        return
    import pandas as pd
    table = pd.read_csv(MANIFEST, sep="\t")
    if table.empty:
        return
    expected = {
        "output_folder": lambda name: str(OUT / name),
        "input_folder": lambda name: str(config["input_dir"]),
        "HFmodel": lambda name: config["model"],
        "cell_types": lambda name: [int(c) for c in CELLS],
        "get_peaks": lambda name: False,
    }
    # a few JSONs are enough: they are all written together by make_inputs.py
    for name in table["name"].iloc[[0, len(table) // 2, -1]]:
        path = JSONS / f"{name}.json"
        if not path.exists():
            continue
        on_disk = json.loads(path.read_text())
        for key, want in expected.items():
            if on_disk.get(key) != want(name):
                raise ValueError(
                    f"{path} has {key}={on_disk.get(key)!r}, but config.yaml now "
                    f"gives {want(name)!r}. The JSONs are from an earlier setup. "
                    f"Delete {JSONS} and run again; Snakemake will rewrite them."
                )


check_existing_jsons()

CHROMS = sorted(WINDOWS)
ALL_NAMES = [name for names in WINDOWS.values() for name in names]

skipped = sorted(set(SIZES) - set(WINDOWS))
if skipped:
    print(
        f"WARNING: shorter than the {WINDOW:,} bp model input, skipped: {skipped}",
        file=sys.stderr,
    )


wildcard_constraints:
    # Only real chromosome and cell names, so no rule can match something else.
    chrom="|".join(re.escape(c) for c in CHROMS),
    cell="|".join(re.escape(c) for c in CELLS),


rule all:
    input: expand(str(GENOME_BIGWIG), cell=CELLS)


rule inputs_only:
    input: MANIFEST


rule per_chromosome:
    input: expand(str(CHROM_BIGWIG), chrom=CHROMS, cell=CELLS)


rule make_inputs:
    """Tile the chromosomes: one config per window, plus the intervals manifest."""
    input:
        chrom_sizes=config["chrom_sizes"],
    output:
        # Only the manifest is declared. The ~27,000 JSONs are written by the same
        # script, but listing them all as outputs made Snakemake compare every
        # predict job's JSON against the whole list (time grows with the square of
        # the number of windows: >10 min just to build the job graph).
        manifest=MANIFEST,
    params:
        cells=" ".join(CELLS),
        chroms=" ".join(CHROMS),
        model=config["model"],
        input_dir=config["input_dir"],
        output_dir=OUT,
        json_dir=JSONS,
    log:
        OUT / "logs" / "make_inputs.log",
    localrule: True
    shell:
        "python {workflow.basedir}/scripts/make_inputs.py {input.chrom_sizes} "
        "-c {params.cells} --chroms {params.chroms} "
        "--output-dir {params.output_dir} --input-dir {params.input_dir} "
        "--model {params.model} --json-dir {params.json_dir} "
        "--manifest {output.manifest} > {log} 2>&1"


rule predict:
    """One window, one GPU job."""
    input:
        # make_inputs writes this window's JSON together with the manifest, so
        # depending on the manifest guarantees the JSON exists
        manifest=MANIFEST,
    output:
        expand(str(WINDOW_BIGWIG), cell=CELLS, allow_missing=True),
    params:
        json=lambda w: str(JSONS / f"{w.name}.json"),
        script=config["pipeline_script"],
        # the window's output_folder, as written into its JSON by make_inputs.py
        outdir=lambda w: str(OUT / w.name),
        # pipeline.py writes here while it runs (outside the window folder, which
        # pipeline.py requires to be empty); moved to the log below when it ends
        running_log=lambda w: str(OUT / "logs" / "predict_running" / f"{w.name}.log"),
    log:
        OUT / "{name}" / "out.log",
    resources:
        mem_mb=config["resources"]["predict"]["mem_mb"],
        runtime=config["resources"]["predict"]["runtime"],
        slurm_partition=config["resources"]["predict"]["partition"],
        gpu=config["resources"]["predict"]["gpu"],
    shell:
        """
        # out.log from an earlier failed attempt would make the folder non-empty
        rm -f {log}
        # Snakemake creates the (empty) folders of the outputs before the job starts,
        # but pipeline.py refuses an output_folder that is not empty. Remove empty
        # folders only; any real file is left alone.
        find {params.outdir} -mindepth 1 -type d -empty -delete
        mkdir -p $(dirname {params.running_log})
        set +e
        python {params.script} --json_file {params.json} > {params.running_log} 2>&1
        status=$?
        set -e
        # keep the log in the window folder whether pipeline.py succeeded or failed
        mkdir -p {params.outdir}
        mv {params.running_log} {log}
        exit $status
        """


rule merge_chromosome:
    """One chromosome's windows into one bigwig, in a single chunked pass.

    Merging all of them at once rather than pairwise is what keeps this feasible:
    memory depends on --chunk, not on the number of windows or the chromosome
    length, and every input is read once.
    """
    input:
        bigwigs=lambda w: [
            str(WINDOW_BIGWIG).format(name=n, cell=w.cell) for n in WINDOWS[w.chrom]
        ],
        manifest=MANIFEST,
    output:
        CHROM_BIGWIG,
    params:
        mode=config["merge_mode"],
        chunk=config["merge_chunk"],
    log:
        OUT / "logs" / "merge" / "{chrom}_class_{cell}.log",
    resources:
        mem_mb=config["resources"]["merge"]["mem_mb"],
        runtime=config["resources"]["merge"]["runtime"],
        slurm_partition=config["resources"]["merge"]["partition"],
    shell:
        "python {workflow.basedir}/scripts/merge_bigwigs.py {output} "
        "--inputs {input.bigwigs} --manifest {input.manifest} "
        "--mode {params.mode} --chunk {params.chunk} --chroms {wildcards.chrom} "
        "> {log} 2>&1"


rule merge_genome:
    """The per-chromosome bigwigs into one file per cell type.

    Nothing overlaps at this point, so there is nothing to reconcile.
    """
    input:
        expand(str(CHROM_BIGWIG), chrom=CHROMS, allow_missing=True),
    output:
        GENOME_BIGWIG,
    params:
        chunk=config["merge_chunk"],
    log:
        OUT / "logs" / "merge" / "WG_class_{cell}.log",
    resources:
        mem_mb=config["resources"]["merge"]["mem_mb"],
        runtime=config["resources"]["merge"]["runtime"],
        slurm_partition=config["resources"]["merge"]["partition"],
    shell:
        "python {workflow.basedir}/scripts/merge_bigwigs.py {output} "
        "--inputs {input} --chunk {params.chunk} > {log} 2>&1"
