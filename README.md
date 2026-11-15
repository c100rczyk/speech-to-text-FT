# Fine-tuning Whisper for Polish conversations in specialized domains

A general speech-to-text model often gets wrong exactly the words that matter in a specialized field: terms,
names, abbreviations, numbers with units. This project is a pipeline for fine-tuning and evaluating Whisper for
such a domain. The code does not depend on the domain. The domain is defined by the data: conversations, the
key terms, the transcription conventions and the test sets.

The project does not assume any particular field. The examples in the documentation and in the code come from
**law** (lawyer–client consultations); any other field uses the same scripts with its own data.

- Base model: Whisper large-v3, fine-tuned with LoRA.
- Language: Polish.
- Training data: synthetic for now. An LLM writes conversations and Chatterbox TTS reads them with voices cloned
  from Common Voice. Real recordings will be added later.

## Repository

| Path | Contents |
|---|---|
| `synthetic-data-creation/` | scripts: reference voices, speech synthesis, export to Hugging Face `audiofolder` ([README](synthetic-data-creation/README.md)) |
| `evaluation/` | scripts: building and checking test sets, evaluating models, comparing runs ([README](evaluation/README.md)) |
| `evaluation/prompts/` | LLM prompt for writing conversations |
| `data/` | all data: voices, conversations, audio, the training export and the test sets (layout below) |
| `ft-pipeline/` | fine-tuning, not started yet |
| `Materials.md`, `used_practices.md` | reading list and practices used in the project |

## Data

All data is in `data/`, and none of it is in git: `.gitignore` keeps only the empty folders. Evaluation
results (`evaluation/results/`) are not in git either.

```
data/
  voices/     train/ dev/ test/               reference clips for voice cloning, one pool per split
  texts/      train/ dev/                     training and validation conversations (JSON)
  tts/        train/ dev/ test-chatterbox/    synthesizer output: 24 kHz turns and whole conversations
  training/   train/ validation/              what fine-tuning reads: 16 kHz turns and metadata.jsonl
  eval/       texts/<domain>-v<N>/            test conversations
              <set>/                          test sets; can live outside the project (STT_EVAL_DATA)
```

Every split goes the same way, from its own texts and its own voices to its final form:

| Split | Texts | Voices | Synthesizer output | Final form |
|---|---|---|---|---|
| training | `texts/train/` | `voices/train/` | `tts/train/` | `training/train/` |
| validation | `texts/dev/` | `voices/dev/` | `tts/dev/` | `training/validation/` |
| test | `eval/texts/legal-v1/` | `voices/test/` | `tts/test-chatterbox/` | `eval/legal-chatterbox-v1/` |

- `export_dataset.py` writes the final form of the training and validation splits,
  `evaluation/build_tts_set.py` that of the test sets. Test audio never goes into `training/`.
- The validation split is called `dev` where it is made and `validation` in the export, because `audiofolder`
  takes the name of the split from the name of the folder.
- The conversations and the `voices.csv` files checked by hand are sources that no script makes again, so
  keep a copy of them. `tts/` is an intermediate step. `training/` and `eval/` are what the models read.
- The test conversations are in `eval/`, apart from the other texts. The evaluation scripts freeze them
  there and refuse to take prompt terms from that directory.

## Pipeline

1. An LLM writes conversations as JSON files (format below).
2. `prepare_voices.py` makes reference voices from Common Voice PL, split into disjoint train / dev / test sets.
3. `generate_dataset.py` reads the conversations with cloned voices, `export_dataset.py` writes 16 kHz turns
   with labels.
4. The test texts are read by test voices and built into frozen test sets (`evaluation/build_tts_set.py`),
   next to test sets of general Polish and of recordings of people. `evaluation/evaluate.py` measures the
   base model and fine-tuned checkpoints on all of them, and `compare.py` compares the runs:

   ```bash
   python evaluation/evaluate.py --out evaluation/results/base
   python evaluation/evaluate.py --out evaluation/results/ft_v1 --model <checkpoint directory>
   python evaluation/compare.py evaluation/results/base evaluation/results/ft_v1
   ```

## Adding a domain

1. **Conversations.** Write training, validation and test conversations in the format below. Training
   conversations go to `data/texts/train/`, validation conversations to `data/texts/dev/`, test conversations
   to `data/eval/texts/<domain>-v1/`. The test conversations must not overlap with the others
   (`evaluation/check_set.py --train-texts` reports what they share).
   `evaluation/prompts/generowanie_rozmow.md` is a prompt template for an LLM: fill in the field, its roles,
   subdomains and conventions.
2. **Key terms.** List in each conversation the terms the model must get right (`terms`). The evaluation
   measures how many of them the model recognizes.
3. **Transcription conventions.** Decide how the domain's abbreviations and names are written in the labels and
   how they are pronounced (see below).
4. **Audio and evaluation.** Generate and export the training data, build and freeze the test sets, evaluate
   the base model, fine-tune, then evaluate the checkpoint on the same test sets
   ([evaluation/README.md](evaluation/README.md)).

## Conversation format

One JSON file per conversation, the same for training and test. An example from law:

```json
{"id": "train_005", "subdomain": "prawo spadkowe", "conversation_type": "pierwsza konsultacja",
 "speakers": {"lawyer": {"gender": "F", "age": 47}, "client": {"gender": "M", "age": 34}},
 "terms": ["zachowek", "k.c.", "KRS"], "pronunciations": {"k.c.": "ka-ce", "KRS": "ka-er-es"},
 "turns": [{"turn": 7, "speaker": "lawyer",
            "text": "Zachowek reguluje art. 991 k.c., sprawdzimy też KRS.",
            "text_tts": "Zachowek reguluje artykuł dziewięćset dziewięćdziesiąt jeden ka-ce, sprawdzimy też ka-er-es."}]}
```

| Field | Meaning |
|---|---|
| `id`, `speakers`, `turns[].turn`, `turns[].speaker`, `turns[].text` | **required**; `text` is the label, what the STT model should output |
| `speakers.<role>` | any role names, e.g. `lawyer`, `client`, `witness`; `gender` F/M selects the voice gender, `age` (optional) the age group |
| `turns[].text_tts` | text for the synthesizer; without it, `text` is read with `[events]` removed |
| `terms` | the key terms of the conversation, the words the model must get right. For test texts, `terms.tsv` next to the conversations gives the base form and type of each term |
| `pronunciations` | spelling → pronunciation map, for listening checks |
| `subdomain` | the part of the field, e.g. `prawo rodzinne`; included in the export, `evaluate.py` reports results per subdomain |
| `conversation_type`, `description` | metadata |

## Transcription conventions

Synthetic labels and manual transcripts of human recordings must follow the same rules, otherwise the model
learns two conflicting conventions.

Rules for every domain:

| Phenomenon | `text` (label) | `text_tts` (audio) |
|---|---|---|
| numbers, values with units | digits: "od 3 dni", "500 zł", "7,8%", "o 17:00" | words, inflected: "od trzech dni", "pięćset złotych" |
| ordinals, "pół", "półtora" | words | words |
| "yyy", "eee", "hmm" | left out | written |
| "mhm", "aha" as an answer | written, always spelled this way | same |
| repetitions, self-corrections | verbatim; a corrected value keeps both values | same |
| cut-off word fragments | left out | written: "po tej inter... intercyzie" |
| cough, sigh, laughter | `[kaszel]` in `text`; the generator removes it from the label and stores it in `events` | turn without `text_tts` |

A turn with an event has no `text_tts`, so it must be plain text without digits, abbreviations or foreign names.

Rules set for each domain (legal example):

| Phenomenon | `text` (label) | `text_tts` (audio) |
|---|---|---|
| abbreviations | as in the domain's documents: KRS, k.p.c., ZUS-ie | Polish letter names joined with hyphens, or as the word is said: ka-er-es, ka-pe-ce, zusie |
| foreign names | spelled as in the domain's reference source (in law the text of the act or of the contract): due diligence, know-how | Polish spelling: dju dilidżens, nołhau |

## Keeping the test data clean

- Test conversations are read only by voices from `data/voices/test`, validation conversations only by voices
  from `data/voices/dev`. The generator does not enforce this; it depends on the `voices.csv` passed to
  `--voices`. `evaluation/check_set.py --voices` reports a test set with a training or validation voice.
- Test sets are frozen (`check_set.py --freeze`) and never regenerated between runs; a correction is a new
  version. The validation set used during training to pick a checkpoint is not a test set and stays out of
  `data/eval/`.
- Evaluation data follows the same license rule as training data: commercial use must be allowed.
- Do not clone voices of people whose recordings are in the test set. The reference voices come from all of
  Common Voice, also its `dev.tsv` and `test.tsv`, so do not test the model on Common Voice recordings.
- If real Common Voice recordings go into training, exclude the people from `data/voices/dev` and
  `data/voices/test` (columns `split` and `client_id` in `data/voices/common_voice_sources.csv`).
- The synthetic test is diagnostics only, because it still sounds like TTS. The real evaluation needs human
  recordings with voices that are not in the training data.
- At most 2–3× as much synthetic data as real data, plus an ablation with and without synthetic data.
- The generated audio is clean. Noise, reverb, phone simulation and speed changes are best added on the fly
  during training, so that every epoch hears different conditions.
