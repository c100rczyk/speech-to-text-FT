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

Data used for create various audio conversations (different voice probes):
Mozilla Common Voice (Mozilla Data Collective)
https://mozilladatacollective.com/datasets/cmu5wsxwp00e0nq07m8v0274j


Whisper fine-tuning:
https://huggingface.co/blog/fine-tune-whisper