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
