# Whole-genome HYPER predictions

This pipeline uses a HYPER model to reconstruct epigenetic tracks in the whole
genome. Briefly, this script tiles each chromosome into overlapping 196,608 bp 
windows, predicts each window on a GPU, and merges the windows back into one 
bigwig per cell type — per chromosome and then genome-wide. Consecutive windows 
overlap by 2 × 40,960 bp, because a prediction is unreliable near the edges of 
the model's input.

## Usage

Edit the input parameters in `config.yaml`, then:

```bash
snakemake -n                                      # dry run: list the jobs
snakemake --profile profiles/slurm                # predict everything, then merge
snakemake --profile profiles/slurm inputs_only    # just the configs and manifest
snakemake --profile profiles/slurm per_chromosome # stop at one bigwig per chromosome
```

Re-run the same command to resume. Logs are stored in `<output_dir>/logs/`.

Running `snakemake -n` before the rest informs how many GPU jobs are about to queue 
(~27,000 for the 24 human chromosomes).

`make_inputs.py` also runs standalone if you only want the configs:

```bash
python scripts/make_inputs.py hg38.chrom.sizes \
    -c 0 1 \
    --output-dir 08_Pipeline_predictions/WG \
    --input-dir 08_Pipeline_predictions \
    -m GFIO/Human_Ery \
    --json-dir 07_JSONs
```

## Parameters (`config.yaml`)

* `chrom_sizes`: tab-separated file of chromosome name and length
* `cell_types`: list of cell type ids to predict, e.g. `[0, 1, 2, 3]`. The number of cells or
the model has is described in the model's HuggingFace README
* `chroms`: restrict to these chromosomes; `[]` uses every one in `chrom_sizes`.
  Set it explicitly for a full assembly, or you will also get windows for every
  alt and unplaced scaffold longer than the model input
* `output_dir`: one folder per window, named `<chrom>_<start>_<end>`; also holds
  the merged per-chromosome and genome-wide bigwigs
* `input_dir`: folder containing the model script and genome files
* `json_dir`: where the per-window prediction configs and `intervals.tsv` go
* `pipeline_script`: the prediction script, called as `--json_file <config>`
* `model`: HFmodel to predict with
* `bigwig_pattern`: where one window's prediction lands inside its folder.
  **Check this against your pipeline's output layout.**
* `merge_mode`: `mean` averages the two predictions in an overlap; `trusted`
  keeps only each window's middle, so no edge value is used and nothing is
  averaged. Either way the chromosome is covered exactly once
* `merge_chunk`: bases held in memory per merge step; this alone sets the merge's
  memory use, not the chromosome length or the number of windows
* `resources`: memory, walltime, partition and `slurm_extra` (for `--gpus=1`)
  for the `predict` and `merge` steps

Window size (196,608 bp) and padding (40,960 bp) are fixed by the model and live
as constants in `scripts/make_inputs.py`.

SLURM settings are in `profiles/slurm/config.yaml`:

* `jobs`: how many jobs in flight; keep within your GPU queue limit
* `retries`: resubmit a job that dies
* `keep-going`: one failed window does not stop the rest

## Outputs

* `<output_dir>/<chrom>_<start>_<end>/…`: one prediction per window
* `<output_dir>/<chrom>/class_<cell>.bw`: merged per chromosome
* `<output_dir>/genome/class_<cell>.bw`: merged genome-wide
* `<json_dir>/intervals.tsv` — every window with its coordinates and trusted
  region; the merge step reads this instead of parsing file names

## Requirements

* Python
* Snakemake
* snakemake-executor-plugin-slurm
* pyBigWig
* NumPy
* Pandas

A GPU partition is needed for the prediction step; the merge steps are CPU only.
