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
     validated.tsv, other.tsv, dev.tsv, test.tsv and clips/:
       python prepare_voices.py --common-voice cv-corpus-XX/pl --out ../data/voices/
       python prepare_voices.py --common-voice cv-corpus-XX/pl --out /tmp/try/ --max-speakers 20
     Takes everyone whose recordings give a good enough clip. A person's recordings come from validated.tsv
     and from other.tsv (recordings nobody has reviewed yet); invalidated.tsv, rejected by listeners, is
     not read. Who is left out:
       - people who reported the age "teens" at least once – they may be minors,
       - accounts with two different genders – usually shared by several people,
       - people with too little clean speech: less than --min-s (6 s), or less than --min-s-few-clips
         (8 s) for a person with fewer than --min-clips (3) recordings.
     A missing gender or age leaves nobody out. The age then stays empty in voices.csv and the generator
     uses the voice for a speaker of any age. A gender is needed to match a voice to a speaker, so when
     the metadata has none, it is taken from the pitch of the finished clip (see below) and the
     gender_source column says so.
     Speakers of the Common Voice dev.tsv / test.tsv are taken too – most Common Voice speakers are there.
     So do not test the STT model on recordings of those splits: their speakers are in the voice pools,
     also in the training one (use e.g. FLEURS instead). --exclude-cv-splits dev test leaves these
     people out.

     Output in data/ – one folder per voice split, each with its own voices.csv:
       voices/train/  voices for the training conversations (data/texts/train/)
       voices/dev/    voices for the synthetic validation set checked during training (data/texts/dev/)
       voices/test/   voices for the test conversations (data/eval/texts/)
       voices/common_voice_sources.csv – every voice with its split, client_id, Common Voice split, the
                      recordings used and how many of them nobody has reviewed (later Common Voice
                      releases may assign speakers differently)
       voices/run_info.json – how the folder was made: the command, the dev/test fractions, the other
                      parameters, how many people each rule left out and the voice counts per split
     voices.csv has the columns id, path, gender (K/M), age (may be empty), split, gender_source
     (metadata / pitch / pitch-uncertain) and pitch_hz (median pitch of the clip).
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

  The gender of a person who gave none is the one thing taken from more than loudness:
  - Pitch (median_pitch): how many times per second the waveform of the voiced sounds repeats, as the
    median over the finished clip (pitch_hz). From 165 Hz up the voice is K, below it is M. Between 150
    and 180 Hz male and female voices overlap, so there gender_source is "pitch-uncertain". On the 152
    voices of the first run, which all had a gender in the metadata, this rule agreed with it for 151.

  What the script will not detect:
  - reverb (bathroom, empty room) – it barely raises the background level,
  - other voices, music or a TV under the speech – to the script that is "speech" too,
  - a poor microphone: a "telephone-like" or muffled sound, mp3 artifacts,
  - a badly or unnaturally read sentence, a voice that does not match the age in the metadata,
  - a recording of other.tsv that is not Polish speech at all: nobody has listened to those.
  It can also go the other way: a clean recording with no silence at the edges gets an underestimated SNR
  (the 10th percentile is then quiet speech) and is rejected for no good reason.

So always listen to the output and remove voices that sound bad from voices.csv – Chatterbox also copies
noise and reverb. At the end the script lists the voices to start with: an uncertain gender, a pitch that
does not fit the gender in the metadata, no pitch at all. Do not clone the voices of people whose
recordings will end up in the test set (speaker leakage into evaluation).
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
CV_SPLITS = ("dev", "test")               # Common Voice splits noted for every voice; --exclude-cv-splits skips them
VOICE_SPLITS = ("train", "dev", "test")   # our own voice split = output subfolders
SPLIT_SALT = "eval-split:"                # prefix hashed with client_id in voice_split; changing it reshuffles
PITCH_RANGE = (60.0, 400.0)               # pitches median_pitch looks for [Hz]
PITCH_K = 165.0                           # without a gender in the metadata: K from this median pitch up, M below
PITCH_UNSURE = (150.0, 180.0)             # male and female voices overlap here

# K/M and mlody/dorosly/senior are the Polish labels used in voices.csv (see the module docstring)
CV_GENDER = {"male_masculine": "M", "male": "M", "female_feminine": "K", "female": "K"}
# Common Voice spells the 40s group "fourties" (e.g. release 27.0); "forties" is kept in case it gets fixed
CV_AGE = {"twenties": "mlody", "thirties": "dorosly", "forties": "dorosly", "fourties": "dorosly",
          "fifties": "dorosly", "sixties": "senior", "seventies": "senior", "eighties": "senior",
          "nineties": "senior"}
# "teens" is skipped on purpose: the group may include minors
AGE_GROUPS = ("senior", "dorosly", "mlody")


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


def median_pitch(y: np.ndarray, sr: int) -> float:
    """Returns the median pitch of the voiced sounds in a recording in Hz, or 0 when it finds none.

    A voiced sound (a vowel, "m", "l") repeats with the period of the vocal folds, so a 40 ms frame looks
    like itself when shifted by that period. For every frame the function measures this likeness
    (autocorrelation, corrected for the window as in Praat) for the shifts that PITCH_RANGE allows and takes
    the shortest shift that is nearly as good as the best one. The best shift alone is sometimes two
    periods long, which would halve the pitch.
    Quiet frames and frames that do not repeat well (noise, "s", "sz") are left out. Fewer than 20 voiced
    frames (0.2 s) count as no pitch.
    """
    n, hop = int(sr * 0.04), int(sr * 0.01)
    if len(y) < n:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop].astype(np.float64)
    frames = frames - frames.mean(axis=1, keepdims=True)
    window = np.hanning(n)
    nfft = 1 << (2 * n - 1).bit_length()

    def autocorr(a: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.abs(np.fft.rfft(a, nfft)) ** 2, nfft)[..., :n]

    r = autocorr(frames * window)
    energy = r[:, 0]
    of_window = autocorr(window)
    r = r / np.maximum(energy[:, None], 1e-12) / np.maximum(of_window / of_window[0], 1e-3)
    lo, hi = int(sr / PITCH_RANGE[1]), int(sr / PITCH_RANGE[0])
    r = r[:, lo: hi + 1]
    best = r.max(axis=1)
    peaks = np.zeros(r.shape, bool)
    peaks[:, 1:-1] = (r[:, 1:-1] >= r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:])
    good = peaks & (r >= 0.9 * best[:, None])
    shift = lo + good.argmax(axis=1)               # the first good peak = the shortest period
    db = 10 * np.log10(energy + 1e-12)
    voiced = good.any(axis=1) & (best >= 0.5) & (db >= np.percentile(db, 95) - 30.0)
    if voiced.sum() < 20:
        return 0.0
    return float(np.median(sr / shift[voiced]))


def gender_from_pitch(pitch: float) -> Tuple[str, str]:
    """Returns (gender, gender_source) for a voice whose owner gave no gender in the metadata.

    K from PITCH_K up, M below. Inside PITCH_UNSURE the source is "pitch-uncertain": listen to such a voice
    and correct voices.csv if needed. Without a pitch (0) both values are empty.
    """
    if pitch <= 0:
        return "", ""
    unsure = PITCH_UNSURE[0] <= pitch < PITCH_UNSURE[1]
    return ("K" if pitch >= PITCH_K else "M"), ("pitch-uncertain" if unsure else "pitch")


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
        msg = f"too little clean speech: {total:.1f} s of the {min_s:g} s needed" \
              + (f"; rejected recordings – {why}" if why else "")
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
    CLIPS_PER_SPEAKER recordings of this person: reviewed ones first, best rated by listeners first.
    `gender` and `age` are what the person reported, empty when they reported nothing. `cv_split` says
    whether the person is in dev.tsv or test.tsv ("dev" / "test"); empty means train.tsv or no split at all.
    """
    client_id: str
    gender: str                # K / M / "" (not in the metadata)
    age: str                   # mlody / dorosly / senior / "" (not in the metadata)
    clips: List[Path]          # the recordings to try, in order
    n_recordings: int          # all recordings of the person, also those beyond CLIPS_PER_SPEAKER
    not_reviewed: Set[str]     # file names in `clips` that come from other.tsv
    cv_split: str = ""         # dev / test / "" (train or no split)

    @property
    def voice_id(self) -> str:
        """Voice identifier in voices.csv, e.g. cv_k_3f9a1c07b2, or cv_x_… without a gender in the metadata.

        Derived from client_id and the reported gender only, so the same person gets the same identifier
        on every run, whatever the pitch says.
        """
        return f"cv_{(self.gender or 'x').lower()}_{self.client_id[:10]}"


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


def load_common_voice(root: Path, exclude_splits: Tuple[str, ...] = ()) -> Tuple[List[CvSpeaker], Counter]:
    """Reads the metadata of a Common Voice release and returns the people who can serve as reference voices.

    Groups the recordings in validated.tsv and other.tsv by client_id and skips a person who:
      - reported the age "teens" at least once – they may be a minor,
      - has two different genders under one account – usually an account shared by several people,
      - appears in a Common Voice split listed in `exclude_splits` (dev.tsv / test.tsv).
    Nobody is skipped for a missing gender or age, or for having few recordings: whether a person gives
    enough speech is decided later, on the audio. A person's age is the most frequently reported group
    (someone may have moved from "twenties" to "thirties" in the meantime). A person's recordings are
    sorted: reviewed ones (validated.tsv) first, then best rated first (up votes minus down votes).
    The function does not open any audio files, it only reads the .tsv files.

    Returns (list of speakers, people with the most recordings first; counter of skipped speakers by reason).
    """
    tsv = root / "validated.tsv"
    if not tsv.exists():
        raise SystemExit(f"Not found: {tsv}")
    cv_split: Dict[str, str] = {}   # client_id -> "dev" / "test"
    for name in CV_SPLITS:
        path = root / f"{name}.tsv"
        if path.exists():
            cv_split.update((r["client_id"], name) for r in read_tsv(path))
        elif name in exclude_splits:
            print(f"! Missing {path} – speakers from this split cannot be excluded", file=sys.stderr)

    rows: Dict[str, List[dict]] = defaultdict(list)
    for r in read_tsv(tsv):
        rows[r["client_id"]].append(r)
    not_reviewed: Set[str] = set()   # file names of other.tsv
    other = root / "other.tsv"
    if other.exists():
        for r in read_tsv(other):
            rows[r["client_id"]].append(r)
            not_reviewed.add(r["path"])
    else:
        print(f"! Missing {other} – only reviewed recordings are used", file=sys.stderr)

    skipped: Counter = Counter()
    speakers = []
    for cid, rs in rows.items():
        ages_raw = {(r.get("age") or "").strip() for r in rs}
        genders = {CV_GENDER[g] for g in ((r.get("gender") or "").strip() for r in rs) if g in CV_GENDER}
        ages = Counter(CV_AGE[a] for a in ((r.get("age") or "").strip() for r in rs) if a in CV_AGE)
        split = cv_split.get(cid, "")
        if split in exclude_splits:
            skipped[f"speaker in {split}.tsv"] += 1
        elif "teens" in ages_raw:
            skipped['age "teens"'] += 1
        elif len(genders) > 1:
            skipped["two genders on one account"] += 1
        else:
            rs = sorted(rs, key=lambda r: (r["path"] in not_reviewed,
                                           -(int(r.get("up_votes") or 0) - int(r.get("down_votes") or 0)),
                                           r["path"]))[:CLIPS_PER_SPEAKER]
            speakers.append(CvSpeaker(cid, genders.pop() if genders else "",
                                      ages.most_common(1)[0][0] if ages else "",
                                      [root / "clips" / r["path"] for r in rs], len(rows[cid]),
                                      {r["path"] for r in rs if r["path"] in not_reviewed}, split))
    speakers.sort(key=lambda s: (-s.n_recordings, s.client_id))
    return speakers, skipped


def print_pool(speakers: List[CvSpeaker], skipped: Counter, min_clips: int, min_s_few_clips: float) -> None:
    """Prints who the candidates are and how many speakers were skipped, by reason.

    It runs before any audio is processed, so you can see right away how large the pool is. Not every
    candidate becomes a voice: the audio checks come next.
    """
    print(f"Common Voice candidates: {len(speakers)}")
    by_gender = Counter(s.gender for s in speakers)
    print(f"  gender in the metadata: K {by_gender['K']}, M {by_gender['M']}; none, to be taken from pitch: "
          f"{by_gender['']}")
    by_age = Counter(s.age for s in speakers)
    print("  age in the metadata: " + ", ".join(f"{a} {by_age[a]}" for a in AGE_GROUPS) + f"; none: {by_age['']}")
    few = sum(s.n_recordings < min_clips for s in speakers)
    print(f"  fewer than {min_clips} recordings: {few} (they need {min_s_few_clips:g} s of clean speech)")
    print(f"  no reviewed recording, only other.tsv: {sum(len(s.not_reviewed) == len(s.clips) for s in speakers)}")
    for split, n in sorted(Counter(s.cv_split for s in speakers if s.cv_split).items()):
        print(f"  of which from {split}.tsv: {n}")
    for why, n in skipped.most_common():
        print(f"  skipped ({why}): {n}")


def prepare_common_voice(root: Path, out: Path, args: argparse.Namespace) -> Tuple[List[dict], List[dict], dict]:
    """Builds Common Voice reference clips and saves them as wav files in out/train, out/dev and out/test.

    Goes through the candidates of load_common_voice, people with the most recordings first, and builds
    a clip for each (build_reference). A person with fewer than --min-clips recordings has to give
    --min-s-few-clips of clean speech instead of --min-s. Everyone who passes becomes a voice, in the folder
    chosen by voice_split; with --max-speakers the script stops after that many voices.
    The gender is the one from the metadata, or else the one gender_from_pitch reads from the clip.

    Returns (rows for voices.csv, rows for common_voice_sources.csv, counts for run_info.json).
    """
    speakers, skipped = load_common_voice(root, tuple(args.exclude_cv_splits))
    print_pool(speakers, skipped, args.min_clips, args.min_s_few_clips)
    rows, sources = [], []
    tried = 0
    print()
    for s in speakers:
        if args.max_speakers and len(rows) >= args.max_speakers:
            break
        tried += 1
        min_s = args.min_s if s.n_recordings >= args.min_clips else args.min_s_few_clips
        y, msg, used = build_reference(s.clips, SR, args.target_s, args.max_s, min_s, args.min_snr, args.max_pause)
        if y is None:
            print(f"- {s.voice_id}: skipped ({msg})")
            continue
        pitch = median_pitch(y, SR)
        gender, source = (s.gender, "metadata") if s.gender else gender_from_pitch(pitch)
        split = voice_split(s.client_id, args.dev_frac, args.test_frac)
        sf.write(str(out / split / f"{s.voice_id}.wav"), y, SR, subtype="PCM_16")
        rows.append({"id": s.voice_id, "path": f"{s.voice_id}.wav", "gender": gender, "age": s.age,
                     "split": split, "gender_source": source, "pitch_hz": round(pitch)})
        sources.append({"id": s.voice_id, "split": split, "client_id": s.client_id, "cv_split": s.cv_split,
                        "clips": ";".join(p.name for p in used),
                        "not_reviewed": sum(p.name in s.not_reviewed for p in used)})
        print(f"+ {s.voice_id} ({gender or '?'}, {s.age or '?'}, {split}): {msg}, pitch {pitch:.0f} Hz")
    pool = {"candidates": len(speakers), "skipped_by_metadata": dict(skipped),
            "too_little_clean_speech": tried - len(rows), "not_tried": len(speakers) - tried}
    return rows, sources, pool


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


def write_run_info(path: Path, argv: List[str], args: argparse.Namespace, rows: List[dict], pool: dict) -> None:
    """Saves how the voices folder was made: the command, the split, the other parameters and the counts.

    `split` holds --dev-frac, --test-frac and how voice_split uses them, so the folder documents itself
    and the next run can warn when the fractions change. `pool` says how many people were candidates and
    how many each rule left out. `result` counts the voices per split and gender as written by this run –
    later manual removals from voices.csv are not reflected there.
    """
    result = {}
    for split in VOICE_SPLITS:
        n = Counter(r["gender"] for r in rows if r["split"] == split)
        total = sum(n.values())
        result[split] = {"K": n["K"], "M": n["M"], "no_gender": n[""], "total": total,
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
        "pool": pool,
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
    ap.add_argument("--max-speakers", type=int,
                    help="stop after this many voices, e.g. for a quick try (default: everyone who passes)")
    ap.add_argument("--min-clips", type=int, default=3,
                    help="a person with fewer recordings needs --min-s-few-clips of clean speech")
    ap.add_argument("--target-s", type=float, default=10.0, help="how much speech to collect before joining")
    ap.add_argument("--max-s", type=float, default=12.0)
    ap.add_argument("--min-s", type=float, default=6.0, help="minimum clean speech of a person [s]")
    ap.add_argument("--min-s-few-clips", type=float, default=8.0,
                    help="minimum clean speech of a person with fewer than --min-clips recordings [s]")
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument("--max-pause", type=float, default=0.35, help="longer pauses inside recordings are shortened")
    ap.add_argument("--exclude-cv-splits", nargs="+", choices=CV_SPLITS, default=[], metavar="SPLIT",
                    help="leave out the speakers of these Common Voice splits: dev, test or both "
                         "(needed only to evaluate the STT model on their recordings)")
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
    rows, sources, pool = prepare_common_voice(Path(args.common_voice), out, args)
    rows.sort(key=lambda r: r["id"])
    sources.sort(key=lambda r: r["id"])
    removed = 0
    for split in VOICE_SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        write_csv(out / split / "voices.csv", split_rows,
                  ["id", "path", "gender", "age", "split", "gender_source", "pitch_hz"])
        removed += remove_stale_clips(out / split, {r["id"] for r in split_rows})
    write_csv(out / "common_voice_sources.csv", sources,
              ["id", "split", "client_id", "cv_split", "clips", "not_reviewed"])
    write_run_info(out / "run_info.json", argv, args, rows, pool)

    by_gender = Counter(r["gender"] for r in rows)
    by_source = Counter(r["gender_source"] for r in rows)
    by_age = Counter(r["age"] for r in rows)
    print(f"\nVoices: {len(rows)} of {pool['candidates']} candidates (K {by_gender['K']}, M {by_gender['M']}"
          + (f", no gender {by_gender['']}" if by_gender[""] else "") + ")")
    print(f"  gender from the metadata: {by_source['metadata']}, from pitch: "
          f"{by_source['pitch'] + by_source['pitch-uncertain']}")
    print("  age: " + ", ".join(f"{a} {by_age[a]}" for a in AGE_GROUPS) + f", not in the metadata {by_age['']}")
    print(f"  not accepted, too little clean speech: {pool['too_little_clean_speech']}"
          + (f"; not tried because of --max-speakers: {pool['not_tried']}" if pool["not_tried"] else ""))
    for split in VOICE_SPLITS:
        n = Counter(r["gender"] for r in rows if r["split"] == split)
        print(f"  {split + ':':6} K {n['K']:4}, M {n['M']:4}  -> {out / split / 'voices.csv'}")
    print(f"Split, parameters and these counts saved in {out / 'run_info.json'}")
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
    in_cv = Counter(r["cv_split"] for r in sources if r["cv_split"])
    if in_cv:
        print("! Voices of people in Common Voice " + ", ".join(f"{split}.tsv: {n}" for split, n in sorted(in_cv.items()))
              + " – do not use these splits to evaluate the STT model (use e.g. FLEURS instead).")

    def listed(ids: List[str], limit: int = 15) -> str:
        return ", ".join(ids[:limit]) + (f" (+{len(ids) - limit} more)" if len(ids) > limit else "")

    unsure = [r["id"] for r in rows if r["gender_source"] == "pitch-uncertain"]
    odd = [f"{r['id']} ({r['gender']}, {r['pitch_hz']} Hz)" for r in rows
           if r["gender_source"] == "metadata" and r["pitch_hz"]
           and (r["pitch_hz"] >= PITCH_UNSURE[1] if r["gender"] == "M" else r["pitch_hz"] < PITCH_UNSURE[0])]
    no_pitch = [r["id"] for r in rows if not r["pitch_hz"]]
    print("Listen to the clips and remove the ones that sound bad from the voices.csv files."
          + (" Start with these:" if unsure or odd or no_pitch else ""))
    if unsure:
        print(f"  gender from a pitch of {PITCH_UNSURE[0]:g}–{PITCH_UNSURE[1]:g} Hz, may be wrong "
              f"(gender_source = pitch-uncertain): {len(unsure)}")
    if odd:
        print(f"  pitch does not fit the gender in the metadata: {listed(odd)}")
    if no_pitch:
        print(f"  no pitch found, maybe not speech: {listed(no_pitch)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
