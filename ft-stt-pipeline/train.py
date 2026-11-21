#!/usr/bin/env python3
"""
Fine-tunes Whisper on the synthetic conversations in data/training/.

    python train.py                                                   # configs/base.yaml as it is
    python train.py --config configs/experiments/01_6_epochs.yaml     # base.yaml with the changes of one experiment
    python train.py --config <experiment file> --output_dir runs/<name>

multiple_experiments.sh runs the last form for every file in configs/experiments/.

The settings of a run come from configs/base.yaml. An experiment file repeats only the settings it changes.
`model` is read by this script; every other key is an argument of Seq2SeqTrainingArguments under its own name,
so any of them can be set in a config without a change here.

Reads data/training/train/ and, when it exists, data/training/validation/. With a validation split the model
is evaluated during training (WER, CER) and the checkpoint with the lowest WER wins.

Writes to --output_dir (default: runs/<name of the config file>/ next to this script; kept out of git):
    checkpoint-<step>/   the state at a save; only the last ones are kept
    final/               the model after training, with its feature extractor and tokenizer:
                         the best checkpoint when there is a validation split, otherwise the last one

final/ can be evaluated on the test sets: evaluation/evaluate.py --model <path to final/>.
"""
import argparse
import dataclasses
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from training_metrics import IGNORE_INDEX, compute_metrics

PIPELINE_DIR = Path(__file__).resolve().parent

DATA_DIR = PIPELINE_DIR.parent / "data" / "training"        # made by synthetic-data-creation/export_dataset.py
BASE_CONFIG = PIPELINE_DIR / "configs" / "base.yaml"        # the settings every run starts from
RUNS_DIR = PIPELINE_DIR / "runs"                            # results of a run, when --output_dir is not given


def load_config(experiment=None):
    """Settings of one run: configs/base.yaml, with the settings of the experiment file written over it."""
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    if experiment is not None:
        config.update(yaml.safe_load(Path(experiment).read_text(encoding="utf-8")) or {})
    return config


def training_arguments(settings, output_dir, has_validation):
    """Settings from the config -> Seq2SeqTrainingArguments: how long and how fast the model learns,
    when it is evaluated and what is saved. Every key has to be the name of one of its arguments.

    Three things are decided here and not in the config:
        output_dir - comes from the command line
        fp16       - half precision (less memory, faster steps) is on when there is a GPU, unless the config sets it
        evaluation - switched off when the data has no validation split
    """
    arguments = {field.name: field for field in dataclasses.fields(Seq2SeqTrainingArguments)}
    del arguments["output_dir"]
    unknown = sorted(set(settings) - set(arguments))
    if unknown:
        sys.exit(f"Config: {', '.join(unknown)}: not a setting that can be given here "
                 "(the arguments of Seq2SeqTrainingArguments, except output_dir)")

    # PyYAML reads 1e-5 as text (a number needs the dot: 1.0e-5). Where a number is expected, text is converted.
    settings = {key: float(value) if arguments[key].type is float and isinstance(value, str) else value
                for key, value in settings.items()}
    settings.setdefault("fp16", torch.cuda.is_available())
    if not has_validation:
        print("No validation split: the model is only trained and the last checkpoints are kept.", flush=True)
        settings |= {"eval_strategy": "no", "load_best_model_at_end": False}
    return Seq2SeqTrainingArguments(output_dir=str(output_dir), **settings)


def prepare_dataset(sample, feature_extractor, tokenizer):
    """One dataset row -> what the model learns from: the spectrogram of the audio and the token IDs of the text.

    feature_extractor: turns raw audio into a log-mel spectrogram, filled up to 30 s
    tokenizer:         turns the transcription into token IDs, with the special tokens around it
                       (<|startoftranscript|><|pl|><|transcribe|><|notimestamps|> ... <|endoftext|>)
    """
    audio = sample["audio"]
    return {
        "input_features": feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0],
        "labels": tokenizer(sample["text"]).input_ids,
    }


def prepare_batch(samples, decoder_start_id):
    """Join single samples into one batch of tensors for the model.

    samples: list of dataset rows after `prepare_dataset`. Only two fields are read:
        "input_features" - log-mel spectrogram, 80 x 3000
        "labels"         - token IDs of the transcription
    decoder_start_id: ID of <|startoftranscript|>, the token every decoder input begins with.

    Returns a dict that can be passed straight to the model:
        "input_features" - float32 tensor, shape (batch, 80, 3000)
        "labels"         - int64 tensor, shape (batch, longest transcription in the batch)
    """
    # 1. Spectrograms. The feature extractor already filled every recording up to 30 s,
    #    so all spectrograms have the same shape and can be stacked as they are.
    #    (Going through numpy is much faster than building a tensor from nested lists.)
    input_features = torch.stack(
        [torch.from_numpy(np.asarray(sample["input_features"], dtype=np.float32)) for sample in samples]
    )

    # 2. Transcriptions.
    token_ids = [torch.as_tensor(sample["labels"], dtype=torch.long) for sample in samples]

    # The model builds the decoder input on its own: it shifts the labels one position
    # to the right and puts <|startoftranscript|> in front. The tokenizer has already
    # put this token in front too, so it is cut off here. Without that the decoder
    # would get it twice.
    token_ids = [ids[1:] if ids[0] == decoder_start_id else ids for ids in token_ids]

    # Transcriptions differ in length and a tensor has to be a rectangle. Start from
    # a tensor filled with IGNORE_INDEX, as wide as the longest transcription,
    # and write each transcription into the beginning of its row.
    longest = max(len(ids) for ids in token_ids)
    labels = torch.full((len(samples), longest), IGNORE_INDEX, dtype=torch.long)
    for row, ids in enumerate(token_ids):
        labels[row, : len(ids)] = ids

    return {"input_features": input_features, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="experiment file: its settings replace those of configs/base.yaml "
                                         "(without it: base.yaml as it is)")
    parser.add_argument("--output_dir", help="where the checkpoints and the final model go "
                                             "(default: runs/<name of the config file>/ next to this script)")
    args = parser.parse_args()

    # --- Settings ---
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / Path(args.config or BASE_CONFIG).stem
    print(f"Settings of this run ({args.config or BASE_CONFIG}); results go to {output_dir}", flush=True)
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), flush=True)
    model_id = config.pop("model")          # everything that is left goes to Seq2SeqTrainingArguments

    # --- Data ---
    # Synthetic conversations, one clip per turn (data/training/README.md describes the folder).
    dataset = load_dataset("audiofolder", data_dir=str(DATA_DIR))

    # Evaluation during training needs the validation split (data/training/validation/).
    has_validation = "validation" in dataset

    # Made before the long steps below, so that a wrong name in a config stops the run at once.
    training_args = training_arguments(config, output_dir, has_validation)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language="polish", task="transcribe")

    # Spectrograms and token IDs for every clip. The columns the data came with (audio, text, voice, ...)
    # are not needed after this step, so only the two new ones stay.
    dataset = dataset.map(
        prepare_dataset,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )

    # --- Model ---
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Used when the model writes transcriptions during evaluation: the language and the task are known,
    # so Whisper does not have to guess the language of every clip.
    model.generation_config.language = "polish"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # --- Training ---
    # Trainer: joins the model, the data and the functions defined above, and runs the training loop.
    # It calls the two functions with one argument each, so the rest is bound here with partial.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_validation else None,
        data_collator=partial(prepare_batch, decoder_start_id=model.config.decoder_start_token_id),
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        # Feature extractor and tokenizer are saved next to the weights, so a saved directory
        # can be loaded on its own (evaluation/evaluate.py --model <directory>).
        processing_class=WhisperProcessor(feature_extractor, tokenizer),
    )
    trainer.train()

    # The model that is in memory now: the best checkpoint with a validation split, otherwise the last state.
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"Model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
