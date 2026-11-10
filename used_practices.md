# Practices that was used in this project

1. During creating audio, we put into audio insertions like "yyy", "eee", etc.
But in transcription, there is not this text. We learn model to not capture 
attention to this insertions.

2. In training data we have different voices, mens, womens, different age. So far
there are only synthetic data from chatterbox, but will be also original ones, for
give better performance

3. Fine tuning of whisper needs the 16kHz audio files. So we need to convert audio files to this 
sampling rates. https://huggingface.co/blog/fine-tune-whisper#:~:text=The%20Whisper%20feature%20extractor%20expects%20audio%20inputs%20with%20a%20sampling%20rate%20of%2016kHz%2C%20so%20we%20need%20to%20match%20our%20inputs%20to%20this%20value%2E .
Different rates would be issue for training model.

4. Whisper was not learned in unsupervised way. All audio data had his substitute in a form of
text of the spoken words.
    + Data in pretraining: pairs (audio + transcription)
    + Task: Prediction of next transcription token
    + What is learned in training: Full mapping audio -> text
    + After pretraining: Model can do transcription of audio
    + Role of Fine-Tuning: **Fine tune to domain and Language**


