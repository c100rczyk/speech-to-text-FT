# data/training/

The data that fine-tuning reads: synthetic conversations (read by Chatterbox TTS), one clip per turn, in the
Hugging Face `audiofolder` format.

```
training/
  train/              training split
  validation/         validation split, checked during training
    *.wav             one turn of a conversation: 16 kHz mono, at most 30 s
    metadata.jsonl    one row per wav
```

In `metadata.jsonl`, `file_name` links a row to its wav and `text` is the label the model learns (the written
form). The other columns (`conversation`, `turn`, `speaker`, `voice`, `duration`, `subdomain`, `terms`) are
for analysis and evaluation, not for training.

```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="data/training")   # DatasetDict with train and validation
```

## Do not edit by hand

`synthetic-data-creation/export_dataset.py` makes this folder from `data/tts/train/` and `data/tts/dev/`, and
every export overwrites its split. To change or drop a conversation, do it in `data/tts/` and export again.

Test data is never here; it is in `data/eval/`.
