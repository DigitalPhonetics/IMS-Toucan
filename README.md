# Speech Synthesis

This is a toolkit to train state-of-the-art Speech Synthesis models. Everything is pure Python and PyTorch based to keep it as
simple and beginner-friendly, yet powerful as possible.

The PyTorch Modules of [TransformerTTS](https://arxiv.org/abs/1809.08895)
and [FastSpeech2](https://arxiv.org/abs/2006.04558) are taken from [ESPNet](https://github.com/espnet/espnet), the
PyTorch Modules of [MelGAN](https://arxiv.org/abs/1910.06711) are taken from
the [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN) which is also made by the
brillant [Tomoki Hayashi](https://github.com/kan-bayashi).

## Embrace Redundancy

While it is a bad practice to have redundancy in regular code, and as much as possible should be abstracted and
parameterized, my experiences in creating this toolkit and working with other toolkits led me to believe that sometimes
redundancy is not only ok, it is actually very convenient. While it does make it more difficult to change things, it also 
makes it also more difficult to break things and cause legacy problems. Also it allows parameterization to focus on the 
truly important parameters, which would otherwise drown in tons of case specific parameters that are never changed.

---

## Working with this Toolkit

The standard way of working with this toolkit is to make your own fork of it, so you can change as much of the code as you like and fully adapt it to your needs. Making pipelines to train models on new datasets, even in new languages requires absolutely minimal new code and you can take the existing code for such models as reference/template.

## Installation
To install this toolkit, clone it onto the machine you want to use it on (should have at least one GPU if you intend to train models on that machine. For inference you can get by without GPU). Navigate to the directory you have cloned and run the command shown below. It is recommended to first create and activate a [pip virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). 

```
pip install -r requirements.txt 
```

If you want to use multi-speaker synthesis, you will need a speaker embedding function. The one assumed in the code is [dvector](https://github.com/yistLin/dvector), because it is incedibly easy to use and freely available. Create a directory "Models" in the top-level of your clone. Then create a directory "Use" in there and in this directory create another directory called "SpeakerEmbedding". In this directory you put the two files "wav2mel.pt" and "dvector-step250000.pt" that you can obtain from the release page of the [dvector](https://github.com/yistLin/dvector) GitHub. This process might become automated in the future.

## Creating a new Pipeline
tbd

## Training a Model
tbd

## Creating a new Inference Interface
tbd

## Using a trained Model for Inference
tbd

---

## Example Pipelines available

| Dataset               | Language  | Single or Multi     | MelGAN | TransformerTTS | FastSpeech2 | 
| ----------------------|-----------|---------------------| :-----:|:--------------:|:-----------:|
| Hokuspokus            | German    | Single Speaker      | ✅     | ✅            | ✅          |
| Thorsten              | German    | Single Speaker      | ✅     | ✅            | ✅          |
| MAILabs Karlsson      | German    | Single Speaker      | ✅     | ✅            | ✅          |
| MAILabs Eva           | German    | Single Speaker      | ✅     | ✅            | ✅          |
| LJSpeech              | English   | Single Speaker      | ✅     | ✅            | ✅          |
| MAILabs Elizabeth     | English   | Single Speaker      | ✅     | ✅            | ✅          |
| LibriTTS              | English   | Multi Speaker       | ✅     | ✅            | ✅          |

