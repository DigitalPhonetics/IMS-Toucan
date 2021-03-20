# Speech Synthesis

Toolkit to train state-of-the-art Speech Synthesis models. Everything is pure Python and PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

The PyTorch Modules of TransformerTTS and FastSpeech2 are taken from https://github.com/espnet/espnet, the PyTorch Modules of MelGAN are taken from https://github.com/kan-bayashi/ParallelWaveGAN

## First Iteration TODOS

| TODO Item                              | Status |
| ---------------------------------------| :---:|
| Prepare a Speaker Verification Dataset | ✅ |
| Train Speaker Embedding Function       | ✅ |
| Prepare TransformerTTS Dataset         | ✅ |
| Train TransformerTTS                   | ✅ |
| Implement Neural Vocoder               | ✅ |
| Prepare Data for Neural Vocoder        | ✅ |
| Train Neural Vocoder                   | ✅ |
| Build Pitch Extractor                  | ✅ |
| Build Energy Extractor                 | ✅ |
| Build Duration Extractor               | ✅ |
| Prepare FastSpeech Dataset             | ✅ |
| Train FastSpeech                       | |
| Build easy Inference Interface         | ✅ |
| Build easy Visualization               | ✅ |

## Second Iteration TODOS

| TODO Item                              | Status |
| ---------------------------------------| :---:|
| Redo dataloading for TransformerTTS | ✅|
| Redo dataloading for SpeakerEmbedding |✅|
| Hyperparameter optimization for each model | ✅|
| Train TransformerTTS that matches Vocoder input better | ✅|
| Train SpeakerEmbedding that matches new filterbank settings| ✅ |
| Extend TransformerTTS to multi-speaker| ✅|
| Extend FastSpeech to multi-speaker| ✅|
| Extend MelGAN to multi-speaker| ✅|
| Prepare MelGAN multi-speaker dataset|✅|
| Prepare Transformer multi-speaker dataset|✅|
| Prepare FastSpeech multi-speaker dataset|✅|
| Train MelGAN on multi-speaker dataset|✅|
| Train Transformer on multi-speaker dataset|🏗️|
| Train FastSpeech on multi-speaker dataset||

## Third Iteration TODOS

| TODO Item                              | Status |
| ---------------------------------------| :---:|
| The Great Refactoring                  | |

## Dataset Support

Pipelines are available for at least one synthesis system for the following datasets:

| Dataset | Language | Single or Multi | MelGAN | TransformerTTS|FastSpeech2| 
| --------|-----------|--------------------| :---:|:---:|:---:|
| Hokuspokus| German | Single Speaker    | ✅ |✅ |✅ |
| LJSpeech | English | Single Speaker    | ✅ |✅ |✅ |
| LibriTTS | English | Multi Speaker     |  ✅ |✅ |✅ |
