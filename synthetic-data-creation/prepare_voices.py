#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds reference clips for voice cloning in Chatterbox, split into train/dev/test folders with voices.csv files.

Chatterbox uses the first ~6 s of the reference (T3 conditioning) and the first ~10 s (decoder), so a good
clip is 8–12 s of clean Polish speech from one person, starting with speech right away, with no music,
reverb or noise. The script joins several recordings of the same person, trims silence, shortens long
pauses, normalizes loudness and rejects recordings with a low estimated SNR.

Modes:
  1) Own recordings (not implemented yet) – each subdirectory is one person:
       python prepare_voices.py --input real_voices/ --out ../data/voices/ [--speakers-csv speakers.csv]
     speakers.csv: speaker,gender,age   (gender K/M, age dziecko/mlody/dorosly/senior)
     Without the CSV, gender is taken from a K_ / M_ prefix in the directory name (e.g. K_anna, M_jan).

  2) Mozilla Common Voice (Polish, CC0) – the language directory of an extracted release, containing
     validated.tsv, dev.tsv, test.tsv and clips/:
       python prepare_voices.py --common-voice cv-corpus-XX/pl --out ../data/voices/ --max-speakers 200
       python prepare_voices.py --common-voice cv-corpus-XX/pl --out ../data/voices/ --include-cv-splits dev test
     Skips people who reported the age "teens" at least once and people with no gender or age in the
     metadata. By default it also skips speakers from dev.tsv/test.tsv, so those Common Voice splits stay
     clean for regression checks. --include-cv-splits takes their speakers too (dev, test or both) – most
     Common Voice speakers are there, so this gives many more voices, but then do not use those splits
     to evaluate the STT model (use e.g. FLEURS instead). Picks voices evenly between K and M, and within
     each gender takes turns between age groups.

     Output in data/ – one folder per voice split, each with its own voices.csv:
       voices/train/  voices for the training conversations (data/texts/train/)
       voices/dev/    voices for the synthetic validation set checked during training (data/texts/dev/)
       voices/test/   voices for the test conversations (data/eval/texts/)
       voices/common_voice_sources.csv – every voice with its split, client_id, Common Voice split and
                      the recordings used (later Common Voice releases may assign speakers differently)
       voices/run_info.json – how the folder was made: the command, the dev/test fractions, the other
                      parameters and the resulting voice counts per split
     About --dev-frac and --test-frac (default 20% each) of the voices go to dev and test, the rest to
     train. Each person lands in exactly one folder, always the same one (see voice_split), so no voice
     is heard in both training and evaluation. Do not change these fractions once training data has been
     generated – the next run compares them with run_info.json and warns if they differ.
     Note: dev/test here is the split of our synthetic voices – unrelated to the Common Voice
     dev.tsv/test.tsv files mentioned above.

Gender and age values follow the Polish convention used across this project (voices.csv, README):
gender K = female (kobieta), M = male (mężczyzna); age dziecko = child, mlody = young, dorosly = adult,
senior = senior.

How the script judges recordings:
  It does not listen to anything and uses no model – it only looks at loudness. It splits a recording into
  20 ms frames, computes each frame's level in dB (frame_db), and every decision is a simple threshold
  on those numbers:

    frame:   1    2    3  ...  35   36   37  ...  150  151
    dB:    -62  -61  -63  ... -24  -19  -22  ...  -58  -60
           └─── silence ───┘  └── speech ─┘       └ pause / end ┘

  - SNR (snr_db): speech level (95th percentile of the frames) minus background level (10th percentile,
    i.e. the silence at the edges of the recording). In the example: −19 − (−62) ≈ 43 dB. A recording
    below --min-snr (20 dB) is rejected.
  - Clipping: a recording is rejected if more than 0.1% of its samples reach full scale (|x| ≥ 0.99).
  - What counts as speech (speech_mask): a frame at most 35 dB quieter than the loudest frames and at least
    8 dB louder than the background (about −54 dB in the example above). Only energy matters, so any loud
    sound – a knock, a cough – also counts as "speech".
  - Trimming (clean_clip): everything before the first and after the last speech frame is cut (keeping
    a 60 ms margin), and pauses longer than --max-pause (0.35 s) have their middle cut out.
  - Normalization (normalize): the mean level of the speech frames (pauses excluded) is set to −20 dBFS,
    then the whole clip is scaled down proportionally if its peak exceeds −1 dBFS.

  What the script will not detect:
  - reverb (bathroom, empty room) – it barely raises the background level,
  - other voices, music or a TV under the speech – to the script that is "speech" too,
  - a poor microphone: a "telephone-like" or muffled sound, mp3 artifacts,
  - a badly or unnaturally read sentence, a voice that does not match the gender/age in the metadata.
  It can also go the other way: a clean recording with no silence at the edges gets an underestimated SNR
  (the 10th percentile is then quiet speech) and is rejected for no good reason.

So always listen to the output and remove voices that sound bad from voices.csv – Chatterbox also copies
noise and reverb. Do not clone the voices of people whose recordings will end up in the test set (speaker
leakage into evaluation).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from math import gcd
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000            # Chatterbox decoder sample rate; the model itself resamples to 16 kHz for T3
FRAME_S = 0.02
EDGE_PAD_S = 0.06     # margin kept when trimming silence, so quiet sounds at the edges are not cut off
GAP_S = 0.25          # silence between joined recordings
MAX_CLIPPED = 0.001   # a recording with a larger share of clipped samples is rejected
CLIPS_PER_SPEAKER = 12
CV_HELD_OUT_SPLITS = ("dev", "test")   # Common Voice splits whose speakers are skipped unless included
VOICE_SPLITS = ("train", "dev", "test")   # our own voice split = output subfolders
SPLIT_SALT = "eval-split:"                # prefix hashed with client_id in voice_split; changing it reshuffles

# K/M and mlody/dorosly/senior are the Polish labels used in voices.csv (see the module docstring)
CV_GENDER = {"male_masculine": "M", "male": "M", "female_feminine": "K", "female": "K"}
# Common Voice spells the 40s group "fourties" (e.g. release 27.0); "forties" is kept in case it gets fixed
CV_AGE = {"twenties": "mlody", "thirties": "dorosly", "forties": "dorosly", "fourties": "dorosly",
          "fifties": "dorosly", "sixties": "senior", "seventies": "senior", "eighties": "senior",
          "nineties": "senior"}
# "teens" is skipped on purpose: the group may include minors
AGE_ORDER = ("senior", "dorosly", "mlody")   # seniors are the rarest, so they come first when taking turns


# --- audio ---------------------------------------------------------------------------------------

def read_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Reads an audio file as a single channel (mono, float32) at sample rate `sr`.

    Stereo channels are averaged, and a different sample rate (Common Voice is usually 48 kHz) is converted
    with a polyphase filter (resample_poly). Mp3 needs libsndfile ≥ 1.1, which soundfile ≥ 0.12 wheels ship.
    Raises an exception if the file cannot be read.
    """
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Splits a recording into 20 ms frames and returns the loudness (RMS) of each frame in dBFS.

    0 dBFS is full scale; digital silence gives about −120 dBFS (the 1e-12 constant guards against log(0)).
    An incomplete last frame is dropped. All further decisions are based on these numbers.
    """
    n = int(sr * FRAME_S)
    frames = x[: len(x) // n * n].reshape(-1, n)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def speech_mask(db: np.ndarray) -> np.ndarray:
    """Marks (True) the frames treated as speech – based on loudness alone.

    The input is the frame levels from frame_db. A frame is speech when it is:
      - at most 35 dB quieter than the loudest frames (95th percentile) – decides in a clean recording,
      - at least 8 dB louder than the background (10th percentile) – decides in a noisy one, so noise
        is not taken for speech.
    The threshold never exceeds the loudest frame, so the mask is never empty.
    It cannot tell speech from other loud sounds (a knock, a cough, music).
    """
    thr = max(np.percentile(db, 95) - 35.0, np.percentile(db, 10) + 8.0)
    return db >= min(thr, db.max())   # the loudest frame always passes, so the mask is never empty


def snr_db(db: np.ndarray) -> float:
    """Estimates the speech-to-noise ratio (SNR) in dB from the frame levels returned by frame_db.

    SNR = speech level (95th percentile) − background level (10th percentile). It assumes the recording
    has at least some silence – without it the 10th percentile is quiet speech and the result is too low.
    Frames of digital silence (below −90 dBFS) are skipped: they are plain zeros, not background noise,
    and would inflate the result.
    For a recording shorter than 10 frames it returns 0, which in practice means "reject".
    """
    live = db[db > -90.0]
    if len(live) < 10:
        return 0.0
    return float(np.percentile(live, 95) - np.percentile(live, 10))


def clean_clip(x: np.ndarray, sr: int, max_pause: float) -> np.ndarray:
    """Removes unneeded silence from a recording: all of it at the edges, and long pauses inside are shortened.

    Steps:
      1. speech_mask finds the speech frames.
      2. Everything before the first and after the last speech frame is cut. EDGE_PAD_S (60 ms) of margin
         is kept, so quiet sounds at the edges, e.g. "s" or "f", are not cut off.
      3. Every pause longer than `max_pause` has its middle cut out, leaving `max_pause` in total on both sides.
    Why: Chatterbox only takes the first ~6 s of the reference, so every second of silence is a second
    of speech lost.
    Returns an empty array when the recording contains no speech.
    """
    n = int(sr * FRAME_S)
    voiced = speech_mask(frame_db(x, sr))
    if not voiced.any():
        return x[:0]
    first, last = np.flatnonzero(voiced)[[0, -1]]
    pad = int(round(EDGE_PAD_S / FRAME_S))
    keep = np.zeros(len(voiced), bool)
    keep[max(0, first - pad): last + pad + 1] = True
    max_frames = int(round(max_pause / FRAME_S))
    run_start = None
    for i in range(first, last + 1):
        if not voiced[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > max_frames:   # keep half of max_pause next to each neighbouring utterance
                keep[run_start + max_frames // 2: i - (max_frames - max_frames // 2)] = False
            run_start = None
    return x[: len(keep) * n][np.repeat(keep, n)]


def fade(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    """Fades a recording in and out over `ms` milliseconds at each end.

    Without it, a cut in the middle of a waveform causes an audible click, e.g. where recordings are joined.
    Does not modify the input array.
    """
    n = min(len(x) // 2, int(sr * ms / 1000))
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def cut_at_quietest(y: np.ndarray, sr: int, max_s: float) -> np.ndarray:
    """Shortens a recording to at most `max_s` seconds, cutting at the quietest point of the last second.

    Cutting exactly at `max_s` often lands in the middle of a word. Instead, the function looks for the
    quietest 20 ms frame between `max_s` − 1 s and `max_s` – usually a gap between recordings or words.
    """
    lo = int((max_s - 1.0) * sr)
    db = frame_db(y[lo: int(max_s * sr)], sr)
    return y[: lo + int(np.argmin(db)) * int(sr * FRAME_S)]


def normalize(y: np.ndarray, sr: int, speech_dbfs: float = -20.0, peak_dbfs: float = -1.0) -> np.ndarray:
    """Sets the speech loudness to `speech_dbfs` and makes sure the peak does not exceed `peak_dbfs`.

    The mean level is computed from speech frames only (speech_mask), so pause length does not affect it.
    First the whole recording is amplified or attenuated so that speech sits at −20 dBFS. If the highest
    sample then exceeds −1 dBFS, the whole clip is additionally scaled down proportionally. There is no
    compression, so the timbre does not change, but speech may end up slightly quieter than −20 dBFS.
    This gives all reference voices a similar loudness.
    """
    db = frame_db(y, sr)
    level = 10 * np.log10(np.mean(10 ** (db[speech_mask(db)] / 10)))
    y = y * 10 ** ((speech_dbfs - level) / 20)
    limit = 10 ** (peak_dbfs / 20)
    peak = float(np.max(np.abs(y)))
    if peak > limit:
        y = y * (limit / peak)
    return y.astype(np.float32)


def build_reference(files: List[Path], sr: int, target_s: float, max_s: float, min_s: float,
                    min_snr: float, max_pause: float) -> Tuple[Optional[np.ndarray], str, List[Path]]:
    """Builds one reference clip from several recordings of the same person.

    Goes through `files` in order (best rated first) and for each recording:
      - rejects it if it is unreadable, shorter than 0.5 s, has an SNR below `min_snr`, or is clipped,
      - trims its silence and shortens its pauses (clean_clip),
      - adds it to the pool until `target_s` seconds of speech have been collected.
    Then it orders the recordings from the cleanest (T3 only sees the first ~6 s), joins them with a GAP_S
    gap, shortens the result to `max_s` if needed (cut_at_quietest) and evens out the loudness (normalize).

    Returns (clip, description, files used). When there is less than `min_s` of clean speech, the clip is
    None, the file list is empty, and the description says how many recordings were rejected and why.
    """
    pieces: List[Tuple[float, np.ndarray, Path]] = []   # (SNR, cleaned recording, file)
    total = 0.0
    rejected: Counter = Counter()
    for f in files:
        try:
            x = read_mono(f, sr)
        except Exception:  # corrupted file, or libsndfile without mp3 support
            rejected["unreadable"] += 1
            continue
        if len(x) < sr * 0.5:
            rejected["too short"] += 1
            continue
        snr = snr_db(frame_db(x, sr))
        if snr < min_snr:
            rejected["noise"] += 1
            continue
        if np.mean(np.abs(x) >= 0.99) > MAX_CLIPPED:
            rejected["clipping"] += 1
            continue
        y = clean_clip(x, sr, max_pause)
        if len(y) < sr * 0.5:
            rejected["too little speech"] += 1
            continue
        pieces.append((snr, fade(y, sr), f))
        total += len(y) / sr
        if total >= target_s:
            break
    if total < min_s:
        why = ", ".join(f"{k}: {v}" for k, v in rejected.items())
        msg = f"too little clean speech: {total:.1f} s" + (f"; rejected recordings – {why}" if why else "")
        return None, msg, []
    pieces.sort(key=lambda p: -p[0])     # cleanest recording first: T3 only sees the first ~6 s
    gap = np.zeros(int(sr * GAP_S), dtype=np.float32)
    y = np.concatenate([part for _, piece, _ in pieces for part in (piece, gap)][:-1])
    if len(y) > max_s * sr:
        y = cut_at_quietest(y, sr, max_s)
    y = fade(normalize(y, sr), sr)
    snr = float(np.median([p[0] for p in pieces]))
    return y, f"{len(y) / sr:.1f} s from {len(pieces)} recordings, SNR≈{snr:.0f} dB", [p[2] for p in pieces]


# --- Common Voice --------------------------------------------------------------------------------

@dataclass
class CvSpeaker:
    """A Common Voice speaker who passed the filters in load_common_voice.

    `client_id` is the anonymous Common Voice account identifier (a SHA hash), and `clips` holds up to
    CLIPS_PER_SPEAKER recordings of this person, best rated by listeners first. `cv_split` says whether
    the person is in dev.tsv or test.tsv ("dev" / "test"); empty means train.tsv or no split at all.
    """
    client_id: str
    gender: str          # K / M
    age: str             # mlody / dorosly / senior
    clips: List[Path]    # best rated first
    cv_split: str = ""   # dev / test / "" (train or no split)

    @property
    def voice_id(self) -> str:
        """Voice identifier in voices.csv, e.g. cv_k_3f9a1c07b2.

        Derived from client_id, so the same person gets the same identifier on every run.
        """
        return f"cv_{self.gender.lower()}_{self.client_id[:10]}"


def voice_split(client_id: str, dev_frac: float, test_frac: float) -> str:
    """Decides which voice split a person goes to: "train", "dev" or "test".

    The decision depends only on the person's client_id: its hash is turned into a number in [0, 1).
    Below `test_frac` the voice goes to test, at or above 1 − `dev_frac` to dev, and in between to train:

        0 ── test ── test_frac ──────── train ──────── 1 − dev_frac ── dev ── 1

    So a person always lands in the same split – on every run, with any --max-speakers, whatever other
    voices were picked. This matters: if the split could change, a voice already used to generate
    training data could later become a test voice.

    Test and dev sit at opposite ends, so raising one fraction only moves voices out of train – never
    between dev and test, never back into train. Even so, change the fractions only before any training
    data has been generated. The price of stability: the shares are only roughly `dev_frac` / `test_frac`,
    also within each gender, which matters when a gender has few voices.
    """
    h = int(hashlib.sha256(f"{SPLIT_SALT}{client_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
    if h < test_frac:
        return "test"
    if h >= 1.0 - dev_frac:
        return "dev"
    return "train"


def read_tsv(path: Path) -> Iterator[dict]:
    """Reads a Common Voice .tsv file row by row as dicts {column name: value}.

    Quoting is disabled (QUOTE_NONE) because Common Voice sentences contain quotation marks. A regular
    parser would take them for the start of a quoted field and merge several rows into one.
    """
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_common_voice(root: Path, min_clips: int,
                      include_splits: Tuple[str, ...] = ()) -> Tuple[List[CvSpeaker], Counter]:
    """Reads the metadata of a Common Voice release and picks speakers who can serve as reference voices.

    Groups the recordings in validated.tsv by client_id and skips a person who:
      - appears in dev.tsv or test.tsv, unless that split is listed in `include_splits` – by default
        their voices stay out of the training data, so those splits remain clean for evaluation,
      - has fewer than `min_clips` recordings,
      - reported the age "teens" at least once – they may be a minor,
      - reported no gender, or has two different genders under one account – usually an account shared
        by several people,
      - reported no age.
    A person's age is the most frequently reported group (someone may have moved from "twenties" to
    "thirties" in the meantime). A person's recordings are sorted best rated first (up votes minus down
    votes). The function does not open any audio files, it only reads the .tsv files.

    Returns (list of speakers, counter of skipped speakers by reason).
    """
    tsv = root / "validated.tsv"
    if not tsv.exists():
        raise SystemExit(f"Not found: {tsv}")
    cv_split: Dict[str, str] = {}   # client_id -> "dev" / "test"
    for name in CV_HELD_OUT_SPLITS:
        path = root / f"{name}.tsv"
        if path.exists():
            cv_split.update((r["client_id"], name) for r in read_tsv(path))
        elif name not in include_splits:
            print(f"! Missing {path} – speakers from this split will not be excluded", file=sys.stderr)

    rows: Dict[str, List[dict]] = defaultdict(list)
    for r in read_tsv(tsv):
        rows[r["client_id"]].append(r)

    skipped: Counter = Counter()
    speakers = []
    for cid, rs in rows.items():
        ages_raw = {(r.get("age") or "").strip() for r in rs}
        genders = {CV_GENDER[g] for g in ((r.get("gender") or "").strip() for r in rs) if g in CV_GENDER}
        ages = Counter(CV_AGE[a] for a in ((r.get("age") or "").strip() for r in rs) if a in CV_AGE)
        split = cv_split.get(cid, "")
        if split and split not in include_splits:
            skipped[f"speaker in {split}.tsv"] += 1
        elif len(rs) < min_clips:
            skipped[f"fewer than {min_clips} recordings"] += 1
        elif "teens" in ages_raw:
            skipped['age "teens"'] += 1
        elif len(genders) != 1:
            skipped["unknown gender" if not genders else "ambiguous gender"] += 1
        elif not ages:
            skipped["unknown age"] += 1
        else:
            rs = sorted(rs, key=lambda r: (-(int(r.get("up_votes") or 0) - int(r.get("down_votes") or 0)),
                                           r["path"]))
            speakers.append(CvSpeaker(cid, genders.pop(), ages.most_common(1)[0][0],
                                      [root / "clips" / r["path"] for r in rs[:CLIPS_PER_SPEAKER]], split))
    return speakers, skipped


def candidate_queues(speakers: List[CvSpeaker]) -> Dict[str, List[CvSpeaker]]:
    """Sets the order in which candidates are tried – separately for K and M.

    Within a gender, age groups take turns (senior, adult, young, senior, …), so the voices are not
    dominated by the largest group (in Common Voice, usually young people). When a group runs out, the
    queue continues with the remaining ones. Within a group, people with more recordings come first,
    since there is more to choose clean fragments from.

    Returns {"K": [...], "M": [...]}.
    """
    groups: Dict[Tuple[str, str], List[CvSpeaker]] = defaultdict(list)
    for s in speakers:
        groups[(s.gender, s.age)].append(s)
    for lst in groups.values():
        lst.sort(key=lambda s: (-len(s.clips), s.client_id))
    return {g: [s for batch in zip_longest(*(groups[(g, a)] for a in AGE_ORDER)) for s in batch if s is not None]
            for g in ("K", "M")}


def print_pool(speakers: List[CvSpeaker], skipped: Counter) -> None:
    """Prints the number of candidates per gender/age group and how many speakers were skipped, by reason.

    It runs before any audio is processed, so you can see right away whether the pool is enough
    for --max-speakers.
    """
    print(f"Common Voice candidates: {len(speakers)}")
    for g in ("K", "M"):
        by_age = Counter(s.age for s in speakers if s.gender == g)
        print(f"  {g}: " + ", ".join(f"{a} {by_age[a]}" for a in AGE_ORDER))
    for split, n in sorted(Counter(s.cv_split for s in speakers if s.cv_split).items()):
        print(f"  of which from {split}.tsv: {n}")
    for why, n in skipped.most_common():
        print(f"  skipped ({why}): {n}")


def prepare_common_voice(root: Path, out: Path, args: argparse.Namespace) -> Tuple[List[dict], List[dict]]:
    """Builds Common Voice reference clips and saves them as wav files in out/train, out/dev and out/test.

    Splits --max-speakers evenly between K and M (with an odd number, K gets one extra voice).
    For each gender it takes the next candidates from candidate_queues and builds a clip (build_reference)
    until its quota is filled. A candidate rejected by the quality checks does not take up a slot – the next
    one steps in. If one gender runs out of candidates, the other one gets the missing slots.
    Each accepted voice goes to the folder chosen by voice_split.

    Returns (rows for voices.csv, rows for common_voice_sources.csv).
    """
    speakers, skipped = load_common_voice(root, args.min_clips, tuple(args.include_cv_splits))
    print_pool(speakers, skipped)
    queues = candidate_queues(speakers)
    quota = {"K": (args.max_speakers + 1) // 2, "M": args.max_speakers // 2}
    accepted: Dict[str, List[CvSpeaker]] = {"K": [], "M": []}
    pos = {"K": 0, "M": 0}
    rows, sources = [], []

    def fill(g: str, want: int) -> None:
        """Adds voices of gender `g` until there are `want` of them or the candidates run out."""
        while len(accepted[g]) < want and pos[g] < len(queues[g]):
            s = queues[g][pos[g]]
            pos[g] += 1
            y, msg, used = build_reference(s.clips, SR, args.target_s, args.max_s, args.min_s, args.min_snr,
                                           args.max_pause)
            if y is None:
                print(f"- {s.voice_id}: skipped ({msg})")
                continue
            split = voice_split(s.client_id, args.dev_frac, args.test_frac)
            sf.write(str(out / split / f"{s.voice_id}.wav"), y, SR, subtype="PCM_16")
            accepted[g].append(s)
            rows.append({"id": s.voice_id, "path": f"{s.voice_id}.wav", "gender": s.gender, "age": s.age,
                         "split": split})
            sources.append({"id": s.voice_id, "split": split, "client_id": s.client_id,
                            "cv_split": s.cv_split, "clips": ";".join(p.name for p in used)})
            print(f"+ {s.voice_id} ({s.gender}, {s.age}, {split}): {msg}")

    print()
    for g in ("K", "M"):
        fill(g, quota[g])
    for g, other in (("K", "M"), ("M", "K")):   # if one gender ran short, fill up from the other
        missing = quota[other] - len(accepted[other])
        if missing > 0:
            fill(g, len(accepted[g]) + missing)
    return rows, sources


def remove_stale_clips(folder: Path, keep_ids: Set[str]) -> int:
    """Deletes clips left in `folder` by earlier runs that are not in its new voices.csv; returns how many.

    Only the script's own files (cv_*.wav) are touched. Without this, a change of --dev-frac/--test-frac
    could leave an old clip of a person in train/ while that person is now in test/.
    """
    stale = [p for p in folder.glob("cv_*.wav") if p.stem not in keep_ids]
    for p in stale:
        p.unlink()
    return len(stale)


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    """Writes a list of dicts to a CSV file (UTF-8, comma-separated) with a header made of `fields`."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_split_fracs(path: Path) -> Optional[Dict[str, float]]:
    """Returns the dev/test fractions recorded in run_info.json by an earlier run, or None if there are none."""
    try:
        split = json.loads(path.read_text(encoding="utf-8"))["split"]
        return {"dev_frac": float(split["dev_frac"]), "test_frac": float(split["test_frac"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_run_info(path: Path, argv: List[str], args: argparse.Namespace, rows: List[dict]) -> None:
    """Saves how the voices folder was made: the command, the split, the other parameters and the counts.

    `split` holds --dev-frac, --test-frac and how voice_split uses them, so the folder documents itself
    and the next run can warn when the fractions change. `result` counts the voices per split and gender
    as written by this run – later manual removals from voices.csv are not reflected there.
    """
    result = {}
    for split in VOICE_SPLITS:
        n = Counter(r["gender"] for r in rows if r["split"] == split)
        total = n["K"] + n["M"]
        result[split] = {"K": n["K"], "M": n["M"], "total": total,
                         "share": round(total / len(rows), 3) if rows else 0.0}
    params = {k: v for k, v in vars(args).items() if k not in ("dev_frac", "test_frac")}
    params["common_voice"] = str(Path(args.common_voice).resolve())
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": shlex.join(["python", "prepare_voices.py", *argv]),
        "split": {
            "dev_frac": args.dev_frac,
            "test_frac": args.test_frac,
            "method": f"sha256('{SPLIT_SALT}' + client_id) mapped to [0, 1): below test_frac -> test, "
                      "at or above 1 - dev_frac -> dev, otherwise train",
        },
        "parameters": params,
        "result": result,
    }
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: parses arguments, builds the clips, writes out/<split>/voices.csv and the summary files.

    `argv` lets you call the script from code or a test (None = command-line arguments).
    Returns exit code 0. A missing validated.tsv ends the program with a message (SystemExit).
    """
    ap = argparse.ArgumentParser(description="Reference clips for voice cloning + voices.csv")
    ap.add_argument("--common-voice", required=True, help="Common Voice release directory for Polish")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-speakers", type=int, default=200)
    ap.add_argument("--min-clips", type=int, default=3, help="minimum number of recordings per person")
    ap.add_argument("--target-s", type=float, default=10.0, help="how much speech to collect before joining")
    ap.add_argument("--max-s", type=float, default=12.0)
    ap.add_argument("--min-s", type=float, default=6.0)
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument("--max-pause", type=float, default=0.35, help="longer pauses inside recordings are shortened")
    ap.add_argument("--include-cv-splits", nargs="+", choices=CV_HELD_OUT_SPLITS, default=[], metavar="SPLIT",
                    help="also take speakers from these Common Voice splits: dev, test or both "
                         "(do not use them to evaluate the STT model afterwards)")
    ap.add_argument("--dev-frac", type=float, default=0.2,
                    help="share of voices for out/dev (synthetic validation during training)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="share of voices for out/test (evaluation conversations)")
    argv = sys.argv[1:] if argv is None else argv
    args = ap.parse_args(argv)
    if args.dev_frac < 0 or args.test_frac < 0 or args.dev_frac + args.test_frac >= 1:
        ap.error("--dev-frac and --test-frac must be >= 0 and add up to less than 1")

    out = Path(args.out)
    for split in VOICE_SPLITS:
        (out / split).mkdir(parents=True, exist_ok=True)
    previous = read_split_fracs(out / "run_info.json")
    rows, sources = prepare_common_voice(Path(args.common_voice), out, args)
    rows.sort(key=lambda r: r["id"])
    sources.sort(key=lambda r: r["id"])
    removed = 0
    for split in VOICE_SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        write_csv(out / split / "voices.csv", split_rows, ["id", "path", "gender", "age", "split"])
        removed += remove_stale_clips(out / split, {r["id"] for r in split_rows})
    write_csv(out / "common_voice_sources.csv", sources, ["id", "split", "client_id", "cv_split", "clips"])
    write_run_info(out / "run_info.json", argv, args, rows)

    by = Counter((r["gender"], r["age"]) for r in rows)
    print(f"\nVoices: {len(rows)} (" + ", ".join(f"{g}/{a}: {by[(g, a)]}" for g in "KM" for a in AGE_ORDER)
          + ")")
    for split in VOICE_SPLITS:
        n = Counter(r["gender"] for r in rows if r["split"] == split)
        print(f"  {split + ':':6} K {n['K']:3}, M {n['M']:3}  -> {out / split / 'voices.csv'}")
    print(f"Split and parameters saved in {out / 'run_info.json'}")
    if removed:
        print(f"Removed {removed} clips left over from earlier runs.")
    if previous and previous != {"dev_frac": args.dev_frac, "test_frac": args.test_frac}:
        print(f"! Split changed since the previous run: dev {previous['dev_frac']} -> {args.dev_frac}, "
              f"test {previous['test_frac']} -> {args.test_frac}. If training data was already generated, "
              "some of its voices may now be in dev/test.")
    total = Counter(r["gender"] for r in rows)
    for split, frac in (("dev", args.dev_frac), ("test", args.test_frac)):
        for g in "KM":
            if frac > 0 and total[g] and not any(r["split"] == split and r["gender"] == g for r in rows):
                print(f"! No {g} voice in {split}. Raise --{split}-frac, "
                      "but only before generating training data.")
    if (out / "voices.csv").exists() or any(out.glob("cv_*.wav")):
        print(f"! {out} still holds voices.csv / cv_*.wav from the old single-folder layout. Delete them, so "
              "they are not used by mistake.")
    if len(rows) < args.max_speakers:
        print(f"! Not enough candidates: {len(rows)} of {args.max_speakers}. Lower --min-clips or --min-snr.")
    if args.include_cv_splits:
        names = " and ".join(f"{x}.tsv" for x in args.include_cv_splits)
        print(f"! Voices include speakers from Common Voice {names} – do not use these splits to evaluate "
              "the STT model (use e.g. FLEURS instead).")
    print("Listen to the clips and remove the ones that sound bad from the voices.csv files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
