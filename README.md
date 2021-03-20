# Speech Synthesis

Toolkit to train state-of-the-art Speech Synthesis models. Everything is pure Python and PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

The PyTorch Modules of TransformerTTS and FastSpeech2 are taken from https://github.com/espnet/espnet, the PyTorch Modules of MelGAN are taken from https://github.com/kan-bayashi/ParallelWaveGAN

## Dataset Support

Pipelines are available for at least one synthesis system for the following datasets:

| Dataset | Language | Single or Multi | MelGAN | TransformerTTS|FastSpeech2| 
| --------|-----------|--------------------| :---:|:---:|:---:|
| Hokuspokus| German | Single Speaker    | ✅ |✅ |✅ |
| LJSpeech | English | Single Speaker    | ✅ |✅ |✅ |
| LibriTTS | English | Multi Speaker     |  ✅ |✅ |✅ |
