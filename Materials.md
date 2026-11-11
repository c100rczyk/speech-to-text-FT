# Materials that are useful for this project

### LoRA Without Regret by John Schulman
https://thinkingmachines.ai/blog/lora/
Most important elements from this article:
+ When using LORA/FullFT, we have to see on the batch size during training. Too large
batch size will not be good for the performance of training. Final loss with batch 128 
can be larger than loss with batch 64. Same with 64 and 32 batch. We should search for
smaller batch size but not too small.
+ To which layers apply LORA:
  + Attention-only LORA significantly underperforms MLP-only LoRA



Whisper fine-tuning:
https://huggingface.co/blog/fine-tune-whisper


### DATA:

#### Training data
1. Data used for create various audio conversations (different voice probes):
Mozilla Common Voice (Mozilla Data Collective)
https://mozilladatacollective.com/datasets/cmu5wsxwp00e0nq07m8v0274j
   + This audio will also be used in fine-tuning. In fine tuning we dont want only synthetic data.
   So real audio from this dataset after merging, will also be used.
We split 

Part of conversations will be synthetic (prepared with Chatterbox - Test to speech model),
with vocal timbres based on prepared voice probes from Mozilla Common Voice.

2. VoxPopuli PL https://huggingface.co/datasets/facebook/voxpopuli License: CC0 (dane)
  Contains European Parliament event recordings. (Polish 111 hours).

#### walidation/test data
Data (high quality general polish voices) that will be used in validation na test sets.

+ Multilingual LibriSpeech PL https://www.openslr.org/94/ License: CC BY 4.0 
 Contains audiobooks (Polish 137 hours)
+ CLARIN-PL Studio https://clarin-pl.eu/dspace/items/cd3f6380-97e6-488f-9a91-c228df5c4ac6
  License: CC BY 3.0 , Contains: speech read in the studio
+ CLARIN-PL Mobile https://clarin-pl.eu/dspace/items/92004353-58c3-49f0-97d9-66edf1cf1095
  License: CC BY 3.0 , Contains: speech read by phone
+ ADMEDVOICE https://www.kaggle.com/datasets/karmarci/polish-medical-multi-source-corpus
  License: CC BY 4.0 , Medical terms speech in Polish


### Models:

#### **Speech to text models:**

For text to speech stuff, needed for create synthetic domain data for 
fine tuning whisper we need to create this data.
1. The number one choice is Chatterbox https://huggingface.co/ResembleAI/chatterbox
With this model we create training audio data, based on prepared text conversations.

2. Next model that can be useful for create synthetic audio for text
or evaluation, will be probably use different model tts because fine-tuned
whisper, can be overfitted to chatterbox characteristics of generated voices.
This could be for example: https://huggingface.co/rhasspy/piper-voices

#### **Text to speech models:** 

Main model to fine-tune will be whisper, because of his good general
quality. This is state-of-the-art open source model. https://huggingface.co/openai/whisper-large-v3.


#### **Audio-text embedding model**
This model will be prepared for help with find the proper world from set of few propositions.
After the world has been spoken, and it is potentially the medical term, we get audio of this
word, create emebedding from this, find most similar embeddings of medical terms.
Thanks to this, user, can change the wrong written word, with set of fast to replace most similar
to the real one, propositions.



