# Speech Synthesis

Toolkit to train state-of-the-art Speech Synthesis models. Everything is pure Python and PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

The PyTorch Modules of TransformerTTS and FastSpeech2 are taken from [ESPNet](https://github.com/espnet/espnet), the PyTorch Modules of MelGAN are taken from the [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN) which is also made by the brillant [Tomoki Hayashi](https://github.com/kan-bayashi).

## Vocoder
I have trained a Full-Band-MelGAN on a combination of 11 large and clean single speaker datasets in 11 different languages (word accent languages, pitch accent languages and tonal languages) with both female and male speakers. It can invert spectrograms of any speech near flawlessly very very quickly, even on CPU. So there is not really any need to train a vocoder model yourself, just ask me for the model. [Spectrogram inversion is a solved problem since 2019](https://arxiv.org/abs/1910.06711), spectrogram synthesis is the remaining challenge in text-to-speech.

## Dataset Support

| Dataset      | Language  | Single or Multi     | MelGAN | TransformerTTS | FastSpeech2 | 
| -------------|-----------|---------------------| :-----:|:--------------:|:-----------:|
| Hokuspokus   | German    | Single Speaker      | ✅     | ✅            | ✅          |
| Thorsten   | German    | Single Speaker      | ✅     | ✅            |           |
| LJSpeech     | English   | Single Speaker      | ✅     | ✅            | ✅          |
| LibriTTS     | English   | Multi Speaker       | ✅     | ✅            | ✅          |
| CSS10DE      | German    | Single Speaker      | ✅     | ✅            | ✅          |
| CSS10GR      | Greek     | Single Speaker      | ✅     |                |             |
| CSS10FR      | French    | Single Speaker      | ✅     |                |             |
| CSS10FI      | Finnish   | Single Speaker      | ✅     |                |             |
| CSS10RU      | Russian   | Single Speaker      | ✅     |                |             |
| CSS10CN      | Chinese   | Single Speaker      | ✅     |                |             |
| CSS10JP      | Japanese  | Single Speaker      | ✅     |                |             |
| CSS10DU      | Dutch     | Single Speaker      | ✅     |                |             |
| CSS10HU      | Hungarian | Single Speaker      | ✅     |                |             |
| CSS10ES      | Spanish   | Single Speaker      | ✅     |                |             |

## Embrace Redundancy

While it is a bad practice to have redundancy in regular code, and as much as possible should be abstracted and
parameterized, my experiences in creating this toolkit and working with other toolkits led me to believe that sometimes
redundancy is not only ok, it is necessary. It enables safety when making changes, because you know that you don't
accidentially change too much and cause legacy problems. And it allows parameterization to focus on the truly important
parameters, which would otherwise drown in tons of case specific parameters.

Parameterized wrappers around redundant code seems to be a simple solution to get the best of both worlds I believe. So
every system that one might want to train gets its own unique training pipeline with lots of duplicate code among them.
And every trained system gets a unique inference interface to load and use it, again with lots of shared code inbetween
them. This ensures every model can be reproduced and loaded.
