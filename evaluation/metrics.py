#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counts the word and character errors of a transcript against its reference (score). Imported by
ft-stt-pipeline/training_metrics.py for WER and CER on the validation split; nothing here reads audio or runs
a model.

Both texts are normalized the same way (normalize): lowercase, punctuation becomes a space ("art. 5 § 2 k.c."
-> "art 5 2 k c", "ka-er-es" -> "ka er es"), full hours lose their minutes ("22:00" = "22"). Everything else counts:
"cztery" where the reference has "4" is an error, because the reference is what the fine-tuned model should write.

This file holds only what training uses. evaluate.py, compare.py, check_set.py and evalset.reference_rows call
functions that are not here (key terms, numeric values, one written form, confidence intervals), so they do not
run with it.
"""
from __future__ import annotations

import re

import numpy as np

from textnorm_pl import clean

_TIME = re.compile(r"(?<![\w.,:])(\d{1,2}):(\d\d)(?![\w:]|[.,]\d)")


# --- Text ----------------------------------------------------------------------------------------

def normalize(text: str) -> list[str]:
    """Lowercase words without punctuation; a hyphen also splits words. A time loses the zero before its hour
    and, at a full hour, its minutes: "08:15" = "8:15", "22:00" = "22"."""
    text = _TIME.sub(lambda m: f"{int(m.group(1))}" + ("" if m.group(2) == "00" else f":{m.group(2)}"), clean(text))
    return re.sub(r"[\W_]+", " ", text.lower()).split()


# --- Edit distance -------------------------------------------------------------------------------

def _ids(a, b) -> tuple[np.ndarray, np.ndarray]:
    """The items of two sequences as integers, the same item as the same integer."""
    table: dict = {}
    return (np.array([table.setdefault(x, len(table)) for x in a], dtype=np.int64),
            np.array([table.setdefault(x, len(table)) for x in b], dtype=np.int64))


def _rows(r: np.ndarray, h: np.ndarray, edit: int = 1, match: int = 0):
    """Rows of the Levenshtein table, one per item of the reference. An edit costs `edit`, an equal pair `match`.

    Inside a row every cell also depends on its left neighbour (an insertion), which a loop would compute cell
    by cell. Here the row is first filled from the row above, then all insertions are applied at once:
    cell[j] = j + min(cell[k] - k for k <= j).
    """
    ar = edit * np.arange(len(h) + 1)
    prev = ar.copy()
    for i, x in enumerate(r, 1):
        cur = np.empty_like(prev)
        cur[0] = i * edit
        np.minimum(prev[1:] + edit, prev[:-1] + np.where(h != x, edit, match), out=cur[1:])
        prev = np.minimum.accumulate(cur - ar) + ar
        yield prev


def edit_distance(ref, hyp) -> int:
    """Levenshtein distance between two sequences: lists of words for WER, strings for CER."""
    r, h = _ids(ref, hyp)
    if not len(r) or not len(h):
        return len(r) + len(h)
    for row in _rows(r, h):
        pass
    return int(row[-1])


# --- One recording -------------------------------------------------------------------------------

def score(reference: str, transcript: str) -> dict:
    """Errors of one transcript against its reference; counts to be summed over recordings."""
    ref, hyp = normalize(reference), normalize(transcript)
    return {"words": len(ref), "word_errors": edit_distance(ref, hyp),
            "chars": len(" ".join(ref)), "char_errors": edit_distance(" ".join(ref), " ".join(hyp))}
