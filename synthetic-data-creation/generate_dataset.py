#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates synthetic doctor–patient conversation recordings (Chatterbox Multilingual) for STT fine-tuning.

  python generate_dataset.py conversations/ --voices voices/train/voices.csv --out tts_data/ [--limit 3]

Conversation JSON -> voices cloned from voices.csv -> audio of each turn -> joined conversation.
Next step: export_dataset.py (16 kHz + metadata.jsonl).

Output for each conversation:
  tts_data/<id>/turns/t01.wav ...      each turn separately (24 kHz, as returned by the model)
  tts_data/<id>/conversation.wav       the whole conversation with pauses between turns
  tts_data/<id>/conversation.json      voices, parameters, turn timings, label and the text read by TTS
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Chatterbox parameter ranges, drawn separately for each speaker in each conversation.
PARAMS = {"exaggeration": (0.35, 0.65), "cfg_weight": (0.35, 0.60), "temperature": (0.65, 0.85)}
MAX_CHUNK_CHARS = 220          # a longer utterance goes to TTS sentence by sentence, in chunks of up to 220 chars
TURN_GAP = (0.45, 0.5, 0.15, 2.0)  # pause between turns: median, sigma (log-normal), min, max [s]
SENTENCE_GAP = (0.15, 0.45)    # pause between chunks of one turn [s]
EDGE_SILENCE = 0.5             # silence at the start and end of the conversation [s]
TARGET_DBFS = -21.0            # RMS loudness of each turn
PEAK_DBFS = -1.0


# --- Text ----------------------------------------------------------------------------------------

EVENT = re.compile(r"\[([^\]]+)\]")
MARKUP = re.compile(r"\{([^|}]*)\|([^}]*)\}")   # {written|spoken}


def label_text(text: str) -> str:
    """STT label: [events] removed, {written|spoken} keeps the written form."""
    text = MARKUP.sub(lambda m: m.group(1), EVENT.sub("", text))
    return re.sub(r"\s+", " ", text).strip()


def tts_text(turn: dict) -> str:
    """Text for TTS: text_tts, or if it is missing (a turn with an event), text without [events]
    and with the spoken form from {written|spoken}."""
    if turn.get("text_tts"):
        return turn["text_tts"].strip()
    text = MARKUP.sub(lambda m: m.group(2), EVENT.sub("", turn["text"]))
    return re.sub(r"\s+", " ", text).strip()


def split_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Splits text into sentences and joins them into chunks ≤ max_chars.
    Does not split at an ellipsis ("yyy... bisoprolol")."""
    sentences = [s for s in re.split(r"(?<=[^.][.!?])\s+", text) if s]
    chunks: list[str] = []
    for s in sentences:
        if chunks and len(chunks[-1]) + 1 + len(s) <= max_chars:
            chunks[-1] += " " + s
        else:
            chunks.append(s)
    return chunks


# --- Voices --------------------------------------------------------------------------------------

def age_group(age) -> str | None:
    """Age from the conversation (years or a category) -> age category from voices.csv."""
    if isinstance(age, str):
        return age
    if age is None:
        return None
    return "mlody" if age < 30 else "dorosly" if age < 60 else "senior"


def load_voices(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        voices = list(csv.DictReader(f))
    for v in voices:
        v["path"] = str((path.parent / v["path"]).resolve())
    return voices


def pick_voices(conv: dict, voices: list[dict], rng: np.random.Generator) -> dict[str, dict]:
    """Each speaker gets a different voice of the same gender, from the same age group where possible."""
    chosen: dict[str, dict] = {}
    for role, info in conv["speakers"].items():
        gender = {"F": "K"}.get(info.get("gender"), info.get("gender"))
        free = [v for v in voices if v["id"] not in {c["id"] for c in chosen.values()}]
        same_gender = [v for v in free if v["gender"] == gender] or free
        same_age = [v for v in same_gender if v.get("age") == age_group(info.get("age"))]
        pool = same_age or same_gender
        chosen[role] = pool[rng.integers(len(pool))]
    return chosen


# --- Audio ---------------------------------------------------------------------------------------

def trim_silence(x: np.ndarray, sr: int, top_db: float = 45.0, frame_ms: int = 10) -> np.ndarray:
    """Trims silence at the clip edges (frames more than top_db below the loudest one)."""
    n = max(1, sr * frame_ms // 1000)
    frames = len(x) // n
    if frames == 0:
        return x
    rms = np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms / rms.max()) > -top_db)[0]
    return x[loud[0] * n: (loud[-1] + 1) * n] if len(loud) else x


def normalize(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    y = x * 10 ** (TARGET_DBFS / 20) / rms
    peak = np.abs(y).max()
    limit = 10 ** (PEAK_DBFS / 20)
    return y * (limit / peak) if peak > limit else y


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(np.float32) / 32768, sr


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(16000, sr)
    return x if sr == 16000 else resample_poly(x, 16000 // g, sr // g).astype(np.float32)


# --- TTS -----------------------------------------------------------------------------------------

class ChatterboxTTS:
    """Chatterbox Multilingual. The voice is cloned from a reference file; conds are computed once per voice."""

    def __init__(self, device: str, t3_model: str):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.torch = torch
        # Older releases (e.g. 0.1.x from PyPI) only have V2 and do not accept the t3_model parameter.
        if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=t3_model)
        elif t3_model == "v2":
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        else:
            sys.exit("The installed chatterbox-tts only has the V2 model (no t3_model parameter). "
                     "Install the version from requirements.txt in a separate environment or use --t3-model v2.")
        self.name = f"chatterbox-{t3_model}"
        self.sr = int(self.model.sr)
        self._conds: dict[str, object] = {}

    def synth(self, text: str, voice: dict, params: dict, seed: int) -> np.ndarray:
        if voice["id"] not in self._conds:
            self.model.prepare_conditionals(voice["path"], exaggeration=params["exaggeration"])
            self._conds[voice["id"]] = self.model.conds
        self.model.conds = self._conds[voice["id"]]
        self.torch.manual_seed(seed)
        wav = self.model.generate(text, language_id="pl", **params)
        return wav.squeeze().detach().cpu().numpy().astype(np.float32)


class DummyTTS:
    """A tone whose length depends on the text, for testing the script without a GPU or a model."""
    name = "dummy"
    sr = 24000

    def synth(self, text: str, voice: dict, params: dict, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dur = max(0.4, sum(c.isalpha() for c in text) / 15.0)
        t = np.arange(int(dur * self.sr)) / self.sr
        f0 = 210.0 if voice["gender"] == "K" else 120.0
        x = np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t)) ** 2
        pad = np.zeros(int(0.2 * self.sr))
        return np.concatenate([pad, 0.3 * x + 0.01 * rng.standard_normal(len(t)), pad]).astype(np.float32)


# --- Conversation --------------------------------------------------------------------------------

def stable_seed(*parts) -> int:
    """Seed that depends only on the given values (same conversation -> same voices, pauses and audio)."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


def synth_turn(tts, spoken: str, voice: dict, params: dict, seed_parts: tuple, rng: np.random.Generator) -> np.ndarray:
    """One turn: sentence chunks, each with its own seed, joined with short pauses."""
    audio = []
    for i, chunk in enumerate(split_chunks(spoken)):
        x = trim_silence(tts.synth(chunk, voice, params, stable_seed(*seed_parts, i)), tts.sr)
        if audio:
            audio.append(np.zeros(int(rng.uniform(*SENTENCE_GAP) * tts.sr), dtype=np.float32))
        audio.append(x)
    return normalize(np.concatenate(audio))


def generate_conversation(conv: dict, voices: list[dict], tts, out_dir: Path, seed: int) -> dict:
    conv_id = conv["id"]
    rng = np.random.default_rng(stable_seed(seed, conv_id))
    voice_of = pick_voices(conv, voices, rng)
    params_of = {role: {k: round(float(rng.uniform(*r)), 3) for k, r in PARAMS.items()} for role in voice_of}
    sr = tts.sr

    pieces = [np.zeros(int(EDGE_SILENCE * sr), dtype=np.float32)]
    pos = len(pieces[0])
    turns_meta = []
    t_start = time.time()
    for turn in conv["turns"]:
        role, n = turn["speaker"], turn["turn"]
        spoken = tts_text(turn)
        x = synth_turn(tts, spoken, voice_of[role], params_of[role], (seed, conv_id, n), rng)
        write_wav(out_dir / "turns" / f"t{n:02d}.wav", x, sr)

        if turns_meta:
            median, sigma, lo, hi = TURN_GAP
            g = np.zeros(int(np.clip(rng.lognormal(np.log(median), sigma), lo, hi) * sr), dtype=np.float32)
            pieces.append(g)
            pos += len(g)
        pieces.append(x)
        turns_meta.append({
            "turn": n, "speaker": role, "voice": voice_of[role]["id"],
            "start": round(pos / sr, 3), "end": round((pos + len(x)) / sr, 3),
            "text": label_text(turn["text"]), "tts_text": spoken,
            "events": EVENT.findall(turn["text"]), "file": f"turns/t{n:02d}.wav",
        })
        pos += len(x)
        print(f"  {conv_id} t{n:02d}/{len(conv['turns'])} {role:7s} {len(x) / sr:5.1f} s", flush=True)

    pieces.append(np.zeros(int(EDGE_SILENCE * sr), dtype=np.float32))
    full = np.concatenate(pieces)
    write_wav(out_dir / "conversation.wav", full, sr)
    duration = len(full) / sr
    meta = {
        "id": conv_id, "tts_model": tts.name, "sr": sr, "seed": seed, "duration": round(duration, 2),
        "rtf": round((time.time() - t_start) / duration, 3),
        "speakers": {role: {"voice": v["id"], "gender": v["gender"], "age": v.get("age"),
                            "ref": v["path"], "params": params_of[role]} for role, v in voice_of.items()},
        "meta": {k: conv[k] for k in ("specialization", "visit_type", "description", "medical_terms") if k in conv},
        "turns": turns_meta,
    }
    (out_dir / "conversation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def conversation_dirs(roots: list[str]) -> list[Path]:
    """Conversation directories (containing conversation.json) inside the given output directories."""
    return [p.parent for r in roots for p in sorted(Path(r).glob("*/conversation.json"))]


def cmd_generate(args) -> None:
    src = Path(args.conversations)
    files = sorted(src.glob("*.json")) if src.is_dir() else [src]
    if args.only:
        files = [f for f in files if any(o in f.stem for o in args.only)]
    files = files[: args.limit] if args.limit else files
    voices = load_voices(Path(args.voices))
    print(f"{len(files)} conversations, {len(voices)} voices from {args.voices}")
    tts = DummyTTS() if args.backend == "dummy" else ChatterboxTTS(args.device, args.t3_model)

    for f in files:
        conv = json.loads(f.read_text(encoding="utf-8"))
        out_dir = Path(args.out) / conv["id"]
        if (out_dir / "conversation.json").exists() and not args.overwrite:
            print(f"{conv['id']}: already exists, skipping (--overwrite generates it again)")
            continue
        meta = generate_conversation(conv, voices, tts, out_dir, args.seed)
        voices_txt = ", ".join(f"{r}={s['voice']}" for r, s in meta["speakers"].items())
        print(f"{conv['id']}: {meta['duration']:.0f} s, RTF {meta['rtf']}, {voices_txt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conversations", help="directory with *.json conversations, or a single file")
    ap.add_argument("--voices", required=True, help="voices/<split>/voices.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="only the first N conversations")
    ap.add_argument("--only", nargs="+", help="only conversations whose file name contains this text, e.g. train_014")
    ap.add_argument("--backend", choices=["chatterbox", "dummy"], default="chatterbox")
    ap.add_argument("--t3-model", default="v3", help="v3 (default) or v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    cmd_generate(ap.parse_args())


if __name__ == "__main__":
    main()
