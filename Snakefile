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

# The window list is deterministic, so it is built here rather than discovered
# later: every rule's inputs are then known before anything runs.
SIZES = read_chrom_sizes(Path(config["chrom_sizes"]), config.get("chroms") or None)
WINDOWS = {
    chrom: [window_name(chrom, s, e) for s, e in get_intervals(size)]
    for chrom, size in SIZES.items()
}
WINDOWS = {chrom: names for chrom, names in WINDOWS.items() if names}
if not WINDOWS:
    raise ValueError(
        f"No chromosome in {config['chrom_sizes']} is at least {WINDOW:,} bp, "
        "so no window can be predicted."
    )
CHROMS = sorted(WINDOWS)
ALL_NAMES = [name for names in WINDOWS.values() for name in names]

skipped = sorted(set(SIZES) - set(WINDOWS))
if skipped:
    print(
        f"WARNING: shorter than the {WINDOW:,} bp model input, skipped: {skipped}",
        file=sys.stderr,
    )


rule all:
    input: expand(str(OUT / "genome" / "class_{cell}.bw"), cell=CELLS)


rule inputs_only:
    input: MANIFEST


rule per_chromosome:
    input: expand(str(OUT / "{chrom}" / "class_{cell}.bw"), chrom=CHROMS, cell=CELLS)


rule make_inputs:
    """Tile the chromosomes: one config per window, plus the intervals manifest."""
    input:
        chrom_sizes=config["chrom_sizes"],
    output:
        manifest=MANIFEST,
        jsons=expand(str(JSONS / "{name}.json"), name=ALL_NAMES),
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
        json=JSONS / "{name}.json",
    output:
        expand(str(WINDOW_BIGWIG), cell=CELLS, allow_missing=True),
    params:
        script=config["pipeline_script"],
    log:
        OUT / "logs" / "predict" / "{name}.log",
    resources:
        mem_mb=config["resources"]["predict"]["mem_mb"],
        runtime=config["resources"]["predict"]["runtime"],
        slurm_partition=config["resources"]["predict"]["partition"],
        slurm_extra=config["resources"]["predict"]["slurm_extra"],
    shell:
        "python {params.script} --json_file {input.json} > {log} 2>&1"


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
        OUT / "{chrom}" / "class_{cell}.bw",
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
        expand(str(OUT / "{chrom}" / "class_{cell}.bw"),
               chrom=CHROMS, allow_missing=True),
    output:
        OUT / "genome" / "class_{cell}.bw",
    params:
        chunk=config["merge_chunk"],
    log:
        OUT / "logs" / "merge" / "genome_class_{cell}.log",
    resources:
        mem_mb=config["resources"]["merge"]["mem_mb"],
        runtime=config["resources"]["merge"]["runtime"],
        slurm_partition=config["resources"]["merge"]["partition"],
    shell:
        "python {workflow.basedir}/scripts/merge_bigwigs.py {output} "
        "--inputs {input} --chunk {params.chunk} > {log} 2>&1"
