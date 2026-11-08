#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export for training: every turn -> 16 kHz wav + metadata.jsonl with the "text" label
(Hugging Face audiofolder format, <out>/<split>/).

  python export_dataset.py tts_data/ --out hf_data/ --split train

To drop a bad conversation, delete its directory from tts_data/ before exporting. Only turns longer than --max-s
(Whisper's 30 s window) are skipped. "medical_terms" lists the conversation's terms that occur in the turn's text
(for evaluation/evaluate.py).

Loading: datasets.load_dataset("audiofolder", data_dir="hf_data")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_dataset import conversation_dirs, read_wav, to_16k, write_wav


def cmd_export(args) -> None:
    """Turns -> 16 kHz files + metadata.jsonl in HF audiofolder format (<out>/<split>/)."""
    # audiofolder takes the split name from the directory name: hf_data/train, hf_data/test, ...
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    rows, too_long = [], 0
    for d in conversation_dirs(args.data_dirs):   # every tts_data/<id>/ that has a conversation.json
        meta = json.loads((d / "conversation.json").read_text(encoding="utf-8"))
        # copied from the conversation JSON by generate_dataset.py; empty for conversations generated before that
        terms = meta.get("meta", {}).get("medical_terms", [])
        for t in meta["turns"]:
            # Chatterbox writes 24 kHz, Whisper (its feature extractor) expects 16 kHz
            x16 = to_16k(*read_wav(d / t["file"]))
            # Whisper sees at most 30 s at once; the rest of a longer clip would be cut off, but its label
            # would still contain the whole text
            if len(x16) > args.max_s * 16000:
                too_long += 1
                continue
            name = f"{meta['id']}_t{t['turn']:02d}.wav"   # conversation id in the name keeps names unique
            write_wav(out / name, x16, 16000)
            # "file_name" links the row to its audio file (required by audiofolder), "text" is the label the model
            # learns (the written form, not tts_text); the other columns are for evaluation and analysis
            rows.append({"file_name": name, "text": t["text"], "conversation": meta["id"], "turn": t["turn"],
                         "speaker": t["speaker"], "voice": t["voice"], "duration": round(len(x16) / 16000, 2),
                         "specialization": meta.get("meta", {}).get("specialization"),
                         # only the terms said in this turn; evaluate.py counts how many the model gets right
                         "medical_terms": [m for m in terms if m.lower() in t["text"].lower()]})

    # Remove files from a previous export that are no longer in the metadata (e.g. a deleted conversation),
    # so the directory holds exactly what metadata.jsonl describes
    keep = {r["file_name"] for r in rows}
    for f in out.glob("*.wav"):
        if f.name not in keep:
            f.unlink()
    # One JSON object per line; ensure_ascii=False keeps Polish letters readable in the file
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    minutes = sum(r["duration"] for r in rows) / 60
    print(f"{out}: {len(rows)} clips, {minutes:.1f} min"
          + (f"; skipped {too_long} turns longer than {args.max_s:g} s" if too_long else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dirs", nargs="+", help="output directories of generate_dataset.py, e.g. tts_data/")
    ap.add_argument("--out", required=True, help="e.g. hf_data/ (files go to hf_data/<split>/)")
    ap.add_argument("--split", default="train", help="train / validation / test (the name of the dataset split)")
    ap.add_argument("--max-s", type=float, default=30.0, help="longer turns are skipped (Whisper's window is 30 s)")
    cmd_export(ap.parse_args())


if __name__ == "__main__":
    main()
