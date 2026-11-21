"""
WER and CER on the validation split during training. Used by train.py.

The errors are counted by evaluation/metrics.py, the same code that evaluation/evaluate.py uses, so a WER
shown during training is counted the same way as a WER in an evaluation report.

This file cannot be named metrics.py: `import metrics` below would then load this file itself instead of
evaluation/metrics.py.
"""
import sys
from pathlib import Path

import numpy as np

# evaluation/metrics.py and the textnorm_pl.py it imports. The folder goes to the end of the search path, so that
# no installed package is replaced by a file from it (the folder has its own evaluate.py).
EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.append(str(EVALUATION_DIR))
import metrics  # noqa: E402  (has to come after the search path is extended)

# Fill value for labels. PyTorch cross-entropy skips positions with -100,
# so the model is not scored on the fill, only on real tokens.
IGNORE_INDEX = -100


def compute_metrics(prediction, tokenizer):
    """Compare transcriptions written by the model with the reference ones.

    prediction: what the trainer passes after an evaluation run. The trainer has to generate
    text for that (predict_with_generate=True):
        prediction.predictions - token IDs written by the model, shape (recordings, tokens)
        prediction.label_ids   - token IDs of the reference transcriptions, filled with -100
    tokenizer: turns token IDs back into text. The trainer passes only `prediction`,
    so train.py binds the tokenizer with functools.partial.

    Returns error rates as fractions (0.25 = 25%), lower is better:
        "wer" - word error rate: wrong, missing and extra words / words in the references
        "cer" - like "wer", counted on characters
    """
    def to_texts(token_ids):
        # -100 is not a token, so the tokenizer cannot decode it. The pad token goes in its place:
        # it is a special token and is dropped together with <|startoftranscript|>, <|pl|> etc.
        token_ids = np.where(token_ids == IGNORE_INDEX, tokenizer.pad_token_id, token_ids)
        return tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    transcripts = to_texts(prediction.predictions)
    references = to_texts(prediction.label_ids)

    # Lowercasing, removing punctuation and counting errors happen inside metrics.score.
    scores = [metrics.score(reference, transcript) for reference, transcript in zip(references, transcripts)]

    # Errors and lengths are summed over all recordings and divided at the end. An average
    # of per-recording rates would let a wrong "Tak." weigh as much as a wrong long turn.
    def rate(errors, length):
        return sum(score[errors] for score in scores) / max(sum(score[length] for score in scores), 1)

    return {
        "wer": rate("word_errors", "words"),
        "cer": rate("char_errors", "chars"),
    }
