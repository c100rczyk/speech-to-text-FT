# Fine-tuning Whisper for Polish conversations in specialized domains

A general speech-to-text model often gets wrong exactly the words that matter in a specialized field: terms,
names, abbreviations, numbers with units. This project is a pipeline for fine-tuning and evaluating Whisper for
such a domain. The code does not depend on the domain. The domain is defined by the data: conversations, the
key terms, the transcription conventions and the test sets.

The first domain is **medicine** (doctor–patient visits), and the examples below come from it. Other domains,
e.g. law, finance or technical support, use the same scripts with their own data.

- Base model: Whisper large-v3, fine-tuned with LoRA.
- Language: Polish.
- Training data: synthetic for now. An LLM writes conversations and Chatterbox TTS reads them with voices cloned
  from Common Voice. Real recordings will be added later.

## Repository

| Path | Contents |
|---|---|
| `synthetic-data-creation/` | reference voices, speech synthesis, export to Hugging Face `audiofolder` ([README](synthetic-data-creation/README.md)) |
| `evaluation/evaluate.py` | WER, CER and recall of the domain's key terms, overall and per subdomain and speaker |
| `evaluation/<domain>-evaluation/` | test conversations of a domain, now `medical-evaluation/` |
| `evaluation/general-polish-evaluation/` | general Polish test set, to check that the model does not get worse outside the domain (empty for now) |
| `evaluation/prompts/` | LLM prompt for writing conversations |
| `ft-pipeline/` | fine-tuning, not started yet |
| `Materials.md`, `used_practices.md` | reading list and practices used in the project |

Data (conversations, voices, audio, exports) is not in git. `.gitignore` keeps only the empty folders.

## Pipeline

1. An LLM writes conversations as JSON files (format below).
2. `prepare_voices.py` makes reference voices from Common Voice PL, split into disjoint train / dev / test sets.
3. `generate_dataset.py` reads the conversations with cloned voices, `export_dataset.py` writes 16 kHz turns
   with labels.
4. `evaluation/evaluate.py` measures the base model and fine-tuned checkpoints on the test split:

   ```bash
   python evaluation/evaluate.py synthetic-data-creation/hf_data/test --model openai/whisper-large-v3 --out evaluation/results/base
   ```

## Adding a domain

1. **Conversations.** Write training and test conversations in the format below. Training conversations go to
   `synthetic-data-creation/conversations/`, test conversations to `evaluation/<domain>-evaluation/`, and the two
   must not overlap. `evaluation/prompts/generowanie_rozmow.md` is written for medical visits and still
   describes the old `.txt` format, so it needs adapting.
2. **Key terms.** List in each conversation the terms the model must get right (`medical_terms`). The evaluation
   measures how many of them the model recognizes.
3. **Transcription conventions.** Decide how the domain's abbreviations and names are written in the labels and
   how they are pronounced (see below).
4. **Audio and evaluation.** Generate and export the data, evaluate the base model, fine-tune, then evaluate the
   checkpoint on the same test split.

## Conversation format

One JSON file per conversation, the same for training and test. An example from the medical domain:

```json
{"id": "train_005", "specialization": "gastroenterolog", "visit_type": "pierwsza wizyta",
 "speakers": {"doctor": {"gender": "M", "age": 47}, "patient": {"gender": "F", "age": 34}},
 "medical_terms": ["celiakia", "tTG", "IgA"], "pronunciations": {"tTG": "te-te-gie", "IgA": "i-gie-a"},
 "turns": [{"turn": 7, "speaker": "doctor",
            "text": "Lekarz rodzinny zlecił przeciwciała tTG w klasie IgA.",
            "text_tts": "Lekarz rodzinny zlecił przeciwciała te-te-gie w klasie i-gie-a."}]}
```

| Field | Meaning |
|---|---|
| `id`, `speakers`, `turns[].turn`, `turns[].speaker`, `turns[].text` | **required**; `text` is the label, what the STT model should output |
| `speakers.<role>` | any role names (`doctor`, `patient`, `caregiver`; in another domain e.g. `advisor`, `client`); `gender` F/M selects the voice gender, `age` (optional) the age group |
| `turns[].text_tts` | text for the synthesizer; without it, `text` is read with `[events]` removed |
| `medical_terms` | the domain's key terms in the conversation; the name comes from the first domain, the field works for any domain |
| `pronunciations` | spelling → pronunciation map, for listening checks |
| `specialization` | subdomain, e.g. the medical specialty; included in the export, `evaluate.py` reports results per subdomain |
| `visit_type`, `description` | metadata |

## Transcription conventions

Synthetic labels and manual transcripts of human recordings must follow the same rules, otherwise the model
learns two conflicting conventions.

Rules for every domain:

| Phenomenon | `text` (label) | `text_tts` (audio) |
|---|---|---|
| numbers, values with units | digits: "od 3 dni", "5 mg", "148 na 92", "7,8%", "o 17:00" | words, inflected: "od trzech dni", "pięć miligramów" |
| ordinals, "pół", "półtora" | words | words |
| "yyy", "eee", "hmm" | left out | written |
| "mhm", "aha" as an answer | written, always spelled this way | same |
| repetitions, self-corrections | verbatim; a corrected value keeps both values | same |
| cut-off word fragments | left out | written: "po tej metfo... metforminie" |
| cough, sigh, laughter | `[kaszel]` in `text`; the generator removes it from the label and stores it in `events` | turn without `text_tts` |

A turn with an event has no `text_tts`, so it must be plain text without digits, abbreviations or foreign names.

Rules set for each domain (medical example):

| Phenomenon | `text` (label) | `text_tts` (audio) |
|---|---|---|
| abbreviations | as in the domain's documents: EKG, eGFR, SOR-ze | Polish letter names joined with hyphens: e-ka-gie, e-gie-ef-er, sorze |
| foreign names | spelled as in the domain's reference source (in medicine the SmPC, ChPL): Xarelto, Hashimoto | Polish spelling: ksarelto, haszimoto |

## Keeping the test data clean

- Test conversations are read only by voices from `voices/test`, validation conversations only by voices from
  `voices/dev`. The scripts do not enforce this; it depends on the `voices.csv` passed to `--voices`.
- Do not clone voices of people whose recordings are in the test set.
- If real Common Voice recordings go into training, exclude the people from `voices/dev` and `voices/test`
  (columns `split` and `client_id` in `voices/common_voice_sources.csv`).
- The synthetic test is diagnostics only, because it still sounds like TTS. The real evaluation needs human
  recordings with voices that are not in the training data.
- At most 2–3× as much synthetic data as real data, plus an ablation with and without synthetic data.
- The generated audio is clean. Noise, reverb, phone simulation and speed changes are best added on the fly
  during training, so that every epoch hears different conditions.
