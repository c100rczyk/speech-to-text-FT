


### Train model with LoRA

After training we have to merge weights form base not trained part of model with fine-tuned weights with LoRA. We can do this after training. Because merging
weights is possible, we can prepare one standalone model.

When we have base model, and trained parts, we can merge, unmerge, deleting,
loading different adapters.

After **load base model**, we can **define parameters that are part of LoRA configuration**. 
This configuration give the control how base model is modified by LoRA.
List of parameters on site: huggingface.co/docs/peft/main/conceptual_guides/lora
This parameters answers on four questions: how big is change, where this change go,
which layers are exceptions, and what else is training.

And after this we have trainable model: base model + adapters LoRA + configuration.
This trainable model is not different in training than original model.

There is a set of configs that causes differences in training. 
+ We can set how much of matrices weights we want to train
+ set which layers get LoRA adapters
+ set scaling factor - how strongly the LoRA update is added to the frozen weights;
  (acts like a learning-rate multiplier)
+ set bias - whether the layer's bias terms are trained too

As we know, in models based on neural-networks, initialization before training is important
because can help in better results of training, or just faster training.
E.g. In case of quantized based model and using LoRA for training, there can be good to use
LoftQ initialization, that can result in better results.


### QLoRA training
In quantized LoRA training, quantized are frozen wages. Matrices that are trained
(LoRA) part of layers, stay in full precision. 
+ Layers that have to be trained, cannot be quantized. This is because, step of gradient
is very small, and if wages have e.g in 4-bits only 16 different values, than this 
value of gradient would be invisible.
+ We quantize only small part of model, so quantization of this small part, would have
no big gain.

Quantization is only because, during training we want to load whole big model into 
memory. Gradient is only slightly different, because we change model slightly. And this
not cause big problems with precision of training.
Each weight have its value with just small differences in value.