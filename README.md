# Synthesis

This repository contains synthesis systems from ESPNet adapted to be simpler and work with an intuitive python interface
for training and inference. The goal of this implementation is to allow rapid prototyping and easy tweaks of way too
complex systems.

## First Iteration TODOS

- Prepare a Speaker Verification Dataset ✅
- Train Speaker Embedding Function ✅
- Prepare TransformerTTS Dataset ✅
- Train TransformerTTS ✅
- Implement Neural Vocoder ✅
- Prepare Data for Neural Vocoder ✅
- Train Neural Vocoder
- Build Pitch Extractor
- Build Energy Extractor
- Build Duration Extractor
- Prepare FastSpeech Dataset
- Train FastSpeech
- Build easy Inference Interface

## Second Iteration TODOS

- Redo dataloading for TransformerTTS
- Redo dataloading for SpeakerEmbedding
- Hyperparameter optimization for each model (optimizer settings)
- Train TransformerTTS that matches Vocoder input better
- Train SpeakerEmbedding that matches new spectrogram settings
- Extend TransformerTTS to multi-speaker
- Extend FastSpeech to multi-speaker
- Extend MelGAN to multi-speaker