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

1. Data used for create various audio conversations (different voice probes):
Mozilla Common Voice (Mozilla Data Collective)
https://mozilladatacollective.com/datasets/cmu5wsxwp00e0nq07m8v0274j

2. Data (high quality general polish voices) can be used as part of data for fine-tuning model.
Despite we want to prepare good medical model, we need to have high quality data 

