#!/usr/bin/env python3
"""
Collects the training runs in runs/ into one page: a table of the runs and charts of how each one trained.

    python summarize_runs.py                                  # every run in runs/ -> runs/summary.html
    python summarize_runs.py --runs <directory> --out <file.html>

The table is also printed to the terminal. The script only reads the runs. It needs PyYAML and no GPU.

Where the numbers come from. Every checkpoint-<step>/trainer_state.json holds the history of its run from the
first step, so the checkpoint with the highest step has all of it:
    training loss                      every `logging_steps` steps
    validation loss, WER and CER       after every evaluation (only when the data has a validation split)

The settings of a run are read from <run>/settings.yaml, which train.py writes. A run trained before train.py
did that has no such file: its settings are then read from configs/ as they are now (base.yaml and the
experiment file with the name of the run), and the page marks the run.

To keep in mind when reading the page:
    - The lowest WER of a run is found on the validation split and measured on that same split, so it is
      a little too good, and the split is synthetic. Compare the final models on the test sets:
      evaluation/evaluate.py --model <run>/final for every run, then evaluation/compare.py.
    - With gradient_accumulation_steps the trainer logs the training loss summed over the batches of one
      update. It is divided back here, so in every run the training loss is the loss of one batch.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PIPELINE_DIR / "configs"
BASE_CONFIG = CONFIGS_DIR / "base.yaml"
RUNS_DIR = PIPELINE_DIR / "runs"
TEMPLATE = PIPELINE_DIR / "runs_template.html"      # the page without numbers
BASELINE = "base"       # name of the run of base.yaml as it is (python train.py); the others are compared with it


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def number(value, digits=4):
    """A number for the page: rounded, and None for a missing or non-finite value (JSON has no NaN)."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def percent(fraction):
    """0.2664 -> 26.64. compute_metrics returns error rates as fractions, the page shows percent."""
    return number(fraction * 100, 3) if fraction is not None else None


def load_settings(run_dir):
    """Settings a run was trained with -> (settings, True when they were saved with the run)."""
    saved = run_dir / "settings.yaml"
    if saved.exists():
        return read_yaml(saved), True
    settings = read_yaml(BASE_CONFIG)
    experiment = CONFIGS_DIR / "experiments" / f"{run_dir.name}.yaml"
    if experiment.exists():
        settings.update(read_yaml(experiment))
    return settings, False


def load_state(run_dir):
    """trainer_state.json of the newest checkpoint of a run; {} when the run has no checkpoint yet."""
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                    key=lambda path: int(path.parent.name.split("-")[1]))
    return json.loads(states[-1].read_text(encoding="utf-8")) if states else {}


def summarize(run_dir, base_settings, failed):
    """Everything the page shows about one run."""
    settings, settings_saved = load_settings(run_dir)
    state = load_state(run_dir)
    history = state.get("log_history", [])

    # Training loss of one batch at every logging point, as (epoch, loss). The trainer logs the loss summed
    # over the batches of one update, so with gradient accumulation it is divided back.
    batches_per_update = settings.get("gradient_accumulation_steps") or 1
    train = [(entry["epoch"], entry["loss"] / batches_per_update)
             for entry in history if "loss" in entry and math.isfinite(entry["loss"])]

    # One row per evaluation. Its training loss is the mean since the evaluation before: with an evaluation
    # after every epoch that is the mean of the epoch.
    evaluations = []
    previous = 0
    for entry in history:
        if "eval_loss" not in entry:
            continue
        losses = [loss for epoch, loss in train if previous < epoch <= entry["epoch"]]
        evaluations.append({
            "epoch": number(entry["epoch"], 2),
            "step": entry["step"],
            "train_loss": number(sum(losses) / len(losses)) if losses else None,
            "eval_loss": number(entry["eval_loss"]),
            "wer": percent(entry.get("eval_wer")),
            "cer": percent(entry.get("eval_cer")),
        })
        previous = entry["epoch"]

    # The evaluation with the lowest WER. With load_best_model_at_end and metric_for_best_model: wer
    # (base.yaml) this is the checkpoint that train.py saves to final/.
    scored = [evaluation for evaluation in evaluations if evaluation["wer"] is not None]
    best = min(scored, key=lambda evaluation: evaluation["wer"]) if scored else None

    if (run_dir / "final").is_dir():
        status = "finished"
    elif run_dir.name in failed:
        status = "failed"
    else:
        status = "unfinished"       # still training, or stopped before the end

    return {
        "name": run_dir.name,
        "model": settings.get("model"),
        "status": status,
        "settings_saved": settings_saved,
        "changed": [f"{key}: {value}" for key, value in settings.items() if base_settings.get(key) != value],
        "epochs_done": number(state.get("epoch"), 2),
        "epochs_planned": state.get("num_train_epochs", settings.get("num_train_epochs")),
        "best_epoch": best["epoch"] if best else None,
        "evaluations": evaluations,
        "train": [[number(epoch, 3), number(loss)] for epoch, loss in train],
    }


def print_table(runs):
    """The table of the page, in the terminal."""
    def show(value, pattern):
        return pattern.format(value) if value is not None else "-"

    rows = [("run", "differs from base.yaml", "epochs", "best WER", "at epoch", "CER", "val. loss", "status")]
    for run in runs:
        best = next((e for e in run["evaluations"] if e["epoch"] == run["best_epoch"]), {})
        rows.append((
            run["name"],
            ", ".join(run["changed"]) or "-",
            f"{show(run['epochs_done'], '{:g}')} of {show(run['epochs_planned'], '{:g}')}",
            show(best.get("wer"), "{:.2f}%"),
            show(best.get("epoch"), "{:g}"),
            show(best.get("cer"), "{:.2f}%"),
            show(best.get("eval_loss"), "{:.3f}"),
            run["status"],
        ))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="directory with one folder per run (default: runs/)")
    parser.add_argument("--out", type=Path, help="the page to write (default: <runs>/summary.html)")
    args = parser.parse_args()
    out = args.out or args.runs / "summary.html"

    # The baseline first, then by name. The experiment files are numbered, so that is the order they ran in.
    run_dirs = sorted((path for path in args.runs.glob("*") if path.is_dir()),
                      key=lambda path: (path.name != BASELINE, path.name))
    if not run_dirs:
        raise SystemExit(f"No runs in {args.runs}: train first (python train.py, or multiple_experiments.sh).")

    # multiple_experiments.sh writes the name of every run that stopped with an error to failed.txt.
    failed_file = args.runs / "failed.txt"
    failed = set(failed_file.read_text(encoding="utf-8").split()) if failed_file.exists() else set()

    base_settings = read_yaml(BASE_CONFIG)
    runs = [summarize(run_dir, base_settings, failed) for run_dir in run_dirs]
    print_table(runs)

    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runs_dir": args.runs.name,
        "baseline": BASELINE if any(run["name"] == BASELINE for run in runs) else None,
        "runs": runs,
    }
    # "<" is escaped so that no text from the data (a run is named by its folder) can end the <script> element.
    data_json = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data_json, 1), encoding="utf-8")
    print(f"\nPage written to {out}")


if __name__ == "__main__":
    main()
