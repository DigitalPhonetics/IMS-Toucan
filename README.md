![image](Utility/toucan.png)

IMS Toucan is a toolkit for teaching, training and using state-of-the-art Speech Synthesis models, developed at the
**Institute for Natural Language Processing (IMS), University of Stuttgart, Germany**. Everything is pure Python and
PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

---

## Demonstrations 🦚

### Pre-Generated Audios

[Multi-lingual and multi-speaker audios](https://multilingualtoucan.github.io/)

[Cloning prosody across speakers](https://toucanprosodycloningdemo.github.io)

[Human-in-the-loop edited poetry for German literary studies](https://poetictts.github.io/)

### Interactive Demos

[Check out our multi-lingual demo on Huggingface🤗](https://huggingface.co/spaces/Flux9665/IMS-Toucan)

[Check out our demo on exact style cloning on Huggingface🤗](https://huggingface.co/spaces/Flux9665/SpeechCloning)

[Check out our human-in-the-loop poetry reading demo on Huggingface🤗](https://huggingface.co/spaces/Flux9665/PoeticTTS)

[You can also design the voice of a speaker who doesn't exist on Huggingface🤗](https://huggingface.co/spaces/Flux9665/ThisSpeakerDoesNotExist)

### Pretrained models are available!

You can just try it yourself! Pretrained checkpoints for our massively multi-lingual model, the self-contained aligner,
the embedding function, the vocoder and the embedding GAN are available in the
[release section](https://github.com/DigitalPhonetics/IMS-Toucan/releases). Run the `run_model_downloader.py` script
to automatically download them from the release page and put them into their appropriate locations with appropriate
names.

---

## New Features 🐣

### 2023

- We now use the same PostFlow
  as [PortaSpeech](https://proceedings.neurips.cc/paper/2021/file/748d6b6ed8e13f857ceaa6cfbdca14b8-Paper.pdf) for
  increased quality at basically no expense.
- We also reduce the sampling-rate of the vocoder from 48kHz to 24kHz. While the theoretical upper bound for quality is
  reduced, in practice, the vocoder produces much fewer artifacts.
- Lots of quality of life changes: Finetuning your own model from the provided pretrained checkpoints is easier than
  ever! Just follow the `finetuning_example.py` pipeline.
- We now have the option of choosing between Avocodo and [BigVGAN](https://arxiv.org/abs/2206.04658), which has an improved  
  generator over HiFiGAN. It is significantly slower on CPU, but the quality is the best I have heard at the time of writing
  this. The speed on GPU is fine, just for CPU inference you might want to stick with Avocodo.
- We compile a bunch of quality enhancements from all our previous works so far into one very stable and nice sounding
  architecture, which we call **ToucanTTS**. We submitted a system based on this architecture to the Blizzard Challenge 2023,
  you can try out our system [speaking French here](https://huggingface.co/spaces/Flux9665/Blizzard2023IMS).

### 2022

- We reworked our input representation to now include tone, lengthening and primary stress. All phonemes in the IPA
  standard are now supported, so you can train on **any** language, as long as you have a way to convert text to IPA. We
  also include word-boundary pseudo-tokens which are only visible to the text encoder.
- By conditioning the TTS on speaker and language embeddings in a specific way, multi-lingual and multi-speaker models
  are possible. You can use any speaker in any language, regardless of the language that the speakers themselves are
  speaking. We presented [a paper on this at AACL 2022](https://arxiv.org/abs/2210.12223)!
- Exactly cloning the prosody of a reference utterance is now also possible, and it works in conjunction with everything
  else! So any utterance in any language spoken by any speaker can be replicated and controlled. We
  presented [a paper on this at SLT 2022](https://arxiv.org/abs/2206.12229).
- We apply this to literary studies on poetry and presented
  [a paper on this at Interspeech 2022!](https://www.isca-speech.org/archive/interspeech_2022/koch22_interspeech.html)
- We added simple and intuitive parameters to scale the variance of pitch and energy in synthesized speech.
- We added a scorer utility to inspect your data and find potentially problematic samples.
- You can now use [weights & biases](https://wandb.ai/site) to keep track of your training runs, if you want.
- We upgraded our vocoder from HiFiGAN to the recently published [Avocodo](https://arxiv.org/abs/2206.13404).
- We now use a self-supervised embedding function based on GST, but with a special training procedure to allow for very
  rich speaker conditioning.
- We trained a GAN to sample from this new embeddingspace. This allows us to speak in voices of speakers that do not
  exist. We also found a way to make the sampling process very controllable using intuitive sliders. Check out our
  newest demo on Huggingface to try it yourself!

### 2021

- We officially introduced IMS Toucan in
  [our contribution to the Blizzard Challenge 2021](http://festvox.org/blizzard/bc2021/BC21_IMS.pdf).
- [As shown in this paper](http://festvox.org/blizzard/bc2021/BC21_DelightfulTTS.pdf) vocoders can be used to perform
  super-resolution and spectrogram inversion simultaneously. We added this to our HiFi-GAN vocoder. It now takes 16kHz
  spectrograms as input, but produces 24kHz waveforms.
- We now use articulatory representations of phonemes as the input for all models. This allows us to easily use
  multilingual data to benefit less resource-rich languages.
- We provide a checkpoint trained with a variant of model agnostic meta learning from which you can fine-tune a model
  with very little data in almost any language. The last two contributions are described in
  [our paper that we presented at the ACL 2022](https://aclanthology.org/2022.acl-long.472/)!
- We now use a small self-contained Aligner that is trained with CTC and an auxiliary spectrogram reconstruction
  objective, inspired by
  [this implementation](https://github.com/as-ideas/DeepForcedAligner) for a variety of applications.

---

## Installation 🦉

These instructions should work for most cases, but I heard of some instances where espeak behaves weird, which are
sometimes resolved after a re-install and sometimes not. Also, M1 and M2 MacBooks require a very different installation
process, with which I am unfortunately not familiar.

#### Basic Requirements

To install this toolkit, clone it onto the machine you want to use it on
(should have at least one GPU if you intend to train models on that machine. For inference, you don't need a GPU).
Navigate to the directory you have cloned. We recommend creating and activating a
[virtual environment](https://docs.python.org/3/library/venv.html)
to install the basic requirements into. The commands below summarize everything you need to do.

```
python -m venv <path_to_where_you_want_your_env_to_be>

source <path_to_where_you_want_your_env_to_be>/bin/activate

pip install --no-cache-dir -r requirements.txt
```

Run the second line everytime you start using the tool again to activate the virtual environment again, if you e.g.
logged out in the meantime. To make use of a GPU, you don't need to do anything else on a Linux machine. On a Windows
machine, have a look at [the official PyTorch website](https://pytorch.org/) for the install-command that enables GPU
support.

#### espeak-ng

And finally you need to have espeak-ng installed on your system, because it is used as backend for the phonemizer. If
you replace the phonemizer, you don't need it. On most Linux environments it will be installed already, and if it is
not, and you have the sufficient rights, you can install it by simply running

```
apt-get install espeak-ng
```

For other systems, e.g. Windows, they provide a convenient .msi installer file
[on their GitHub release page](https://github.com/espeak-ng/espeak-ng/releases). After installation on non-linux
systems, you'll also need to tell the phonemizer library where to find your espeak installation, which is discussed in
[this issue](https://github.com/bootphon/phonemizer/issues/44#issuecomment-1008449718). Since the project is still in
active development, there are frequent updates, which can actually benefit your use significantly.

#### Storage configuration

If you don't want the pretrained and trained models as well as the cache files resulting from preprocessing your
datasets to be stored in the default subfolders, you can set corresponding directories globally by
editing `Utility/storage_config.py` to suit your needs (the path can be relative to the repository root directory or
absolute).

#### Pretrained Models

You don't need to use pretrained models, but it can speed things up tremendously. Run the `run_model_downloader.py`
script to automatically download them from the release page and put them into their appropriate locations with
appropriate names.

---

## Creating a new Pipeline 🦆

### Build an Avocodo/BigVGAN Pipeline

This should not be necessary, because we provide a pretrained model and one of the key benefits of vocoders in general
is how incredibly speaker independent they are. But in case you want to train your own anyway, here are the
instructions: You will need a function to return the list of all the absolute paths to each of
the audio files in your dataset as strings. If you already have a *path_to_transcript_dict* of your data for ToucanTTS training,
you can simply take the keys of the dict and transform them into a list.

Then go to the directory
*TrainingInterfaces/TrainingPipelines*. In there, make a copy of any existing pipeline that has Avocodo or BigVGAN in its name. We
will use this as reference and only make the necessary changes to use the new dataset. Look out for a variable called
*model_save_dir*. This is the default directory that checkpoints will be saved into, unless you specify another one when
calling the training script. Change it to whatever you like. Then pass the list of paths to the instanciation of the
Dataset.

Now you need to add your newly created pipeline to the pipeline dictionary in the file
*run_training_pipeline.py* in the top level of the toolkit. In this file, import the
*run* function from the pipeline you just created and give it a meaningful name. Now in the
*pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense.

### Build a ToucanTTS Pipeline

What we call ToucanTTS is actually mostly FastSpeech 2, but with a couple of changes, such as the normalizing flow
based PostNet that was introduced in PortaSpeech. We found the VAE used in PortaSpeech too unstable for low-resource
cases, so we continue experimenting with those in experimental branches of the toolkit. There are a bunch of other
changes that mostly relate to low-resource scenarios. For more info, have a look at the ToucanTTS docstring.

In the directory called
*Utility* there is a file called
*path_to_transcript_dicts.py*. In this file you should write a function that returns a dictionary that has all the
absolute paths to each of the audio files in your dataset as strings as the keys and the textual transcriptions of the
corresponding audios as the values.

Then go to the directory
*TrainingInterfaces/TrainingPipelines*. In there, make a copy of the `finetuning_example.py` file. We will use this copy
as reference and only make the necessary changes to use the new dataset. Import the function
you have just written as
*build_path_to_transcript_dict*. Since the data will be processed a considerable amount, a cache will be built and saved
as file for quick and easy restarts. So find the variable
*cache_dir* and adapt it to your needs. The same goes for the variable
*save_dir*, which is where the checkpoints will be saved to. This is a default value, you can overwrite it when calling
the pipeline later using a command line argument, in case you want to fine-tune from a checkpoint and thus save into a
different directory.

Since we are using text here, we have to make sure that the text processing is adequate for the language. So check in
*Preprocessing/TextFrontend* whether the TextFrontend already has a language ID (e.g. 'en' and 'de') for the language of
your dataset. If not, you'll have to implement handling for that, but it should be pretty simple by just doing it
analogous to what is there already. Now back in the pipeline, change the
*lang* argument in the creation of the dataset and in the call to the train loop function to the language ID that
matches your data.

The arguments that are given to the train loop are meant for the case of finetuning from a pretrained model. If you want
to train from scratch, have a look at a different pipeline that has ToucanTTS in its name and look at the arguments
used there.

Once this is done, we are almost done, now we just need to make it available to the
*run_training_pipeline.py* file in the top level. In said file, import the
*run* function from the pipeline you just created and give it a meaningful name. Now in the
*pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense.

---

## Training a Model 🦜

Once you have a training pipeline built, training is super easy:

```
python run_training_pipeline.py <shorthand of the pipeline>
```

You can supply any of the following arguments, but don't have to (although for training you should definitely specify at
least a GPU ID).

```
--gpu_id <ID of the GPU you wish to use, as displayed with nvidia-smi, default is cpu> 

--resume_checkpoint <path to a checkpoint to load>

--resume (if this is present, the furthest checkpoint available will be loaded automatically)

--finetune (if this is present, the provided checkpoint will be fine-tuned on the data from this pipeline)

--model_save_dir <path to a directory where the checkpoints should be saved>

--wandb (if this is present, the logs will be synchronized to your weights&biases account, if you are logged in on the command line)

--wandb_resume_id <the id of the run you want to resume, if you are using weights&biases (you can find the id in the URL of the run)>
```

After every epoch, some logs will be written to the console. If you get cuda out of memory errors, you need to decrease
the batchsize in the arguments of the call to the training_loop in the pipeline you are running. Try decreasing the
batchsize in small steps until you get no more out of cuda memory errors.

In the directory you specified for saving, checkpoint files and spectrogram visualization
data will appear. Since the checkpoints are quite big, only the five most recent ones will be kept. The amount of
training steps highly depends on the data you are using and whether you're finetuning from a pretrained checkpoint or
training from scratch. The fewer data you have, the fewer steps you should take to prevent overfitting issues. If
you want to stop earlier, just kill the process, since everything is daemonic all the child-processes should die with
it. In case there are some ghost-processes left behind, you can use the following command to find them and kill them
manually.

```
fuser -v /dev/nvidia*
```

After training is complete, it is recommended to run
*run_weight_averaging.py*. If you made no changes to the architectures and stuck to the default directory layout, it
will automatically load any models you produced with one pipeline, average their parameters to get a slightly more
robust model and save the result as
*best.pt* in the same directory where all the corresponding checkpoints lie. This also compresses the file size
significantly, so you should do this and then use the
*best.pt* model for inference.

---

## Using a trained Model for Inference 🦢

You can load your trained models using an inference interface. Simply instanciate it with the proper directory handle
identifying the model you want to use, the rest should work out in the background. You might want to set a language
embedding or a speaker embedding. The methods for that should be self-explanatory.

An *InferenceInterface* contains two methods to create audio from text. They are
*read_to_file* and
*read_aloud*.

- *read_to_file* takes as input a list of strings and a filename. It will synthesize the sentences in the list and
  concatenate them with a short pause inbetween and write them to the filepath you supply as the other argument.

- *read_aloud* takes just a string, which it will then convert to speech and immediately play using the system's
  speakers. If you set the optional argument
  *view* to
  *True* when calling it, it will also show a plot of the phonemes it produced, the spectrogram it came up with, and the
  wave it created from that spectrogram. So all the representations can be seen, text to phoneme, phoneme to spectrogram
  and finally spectrogram to wave.

Their use is demonstrated in
*run_interactive_demo.py* and
*run_text_to_file_reader.py*.

There are simple scaling parameters to control the duration, the variance of the pitch curve and the variance of the
energy curve. You can either change them in the code when using the interactive demo or the reader, or you can simply
pass them to the interface when you use it in your own code.

---

## FAQ 🐓

Here are a few points that were brought up by users:

- My error message shows GPU0, even though I specified a different GPU - The way GPU selection works is that the
  specified GPU is set as the only visible device, in order to avoid backend stuff running accidentally on different
  GPUs. So internally the program will name the device GPU0, because it is the only GPU it can see. It is actually
  running on the GPU you specified.
- read_to_file produces strange outputs - Check if you're passing a list to the method or a string. Since strings can be
  iterated over, it might not throw an error, but a list of strings is expected.
- `UserWarning: Detected call of lr_scheduler.step() before optimizer.step().` - We use a custom scheduler, and torch
  incorrectly thinks that we call the scheduler and the optimizer in the wrong order. Just ignore this warning, it is
  completely meaningless.
- Loss turns to `NaN` - The default learning rates work on clean data. If your data is less clean, try using the scorer
  to find problematic samples, or reduce the learning rate. The most common problem is there being pauses in the speech,
  but nothing that hints at them in the text. That's why ASR corpora, which leave out punctuation, are usually difficult
  to use for TTS.

---

The basic PyTorch Modules of [FastSpeech 2](https://arxiv.org/abs/2006.04558) are taken from
[ESPnet](https://github.com/espnet/espnet), the PyTorch Modules of
[HiFiGAN](https://arxiv.org/abs/2010.05646) are taken from
the [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN)
which are also authored by the brilliant [Tomoki Hayashi](https://github.com/kan-bayashi). Some Modules related to the
Normalizing Flow based PostNet as outlined in
[PortaSpeech](https://proceedings.neurips.cc/paper/2021/file/748d6b6ed8e13f857ceaa6cfbdca14b8-Paper.pdf) are taken
from the [official PortaSpeech codebase](https://github.com/NATSpeech/NATSpeech).

For a version of the toolkit that includes [TransformerTTS](https://arxiv.org/abs/1809.08895),
[Tacotron 2](https://arxiv.org/abs/1712.05884) or [MelGAN](https://arxiv.org/abs/1910.06711), check out the other
branches. They are separated to keep the code clean, simple and minimal as the development progresses.

## Citation 🐧

### Introduction of the Toolkit [[associated code and models]](https://github.com/DigitalPhonetics/IMS-Toucan/releases/tag/v1.0)

```
@inproceedings{lux2021toucan,
  year         = 2021,
  title        = {{The IMS Toucan system for the Blizzard Challenge 2021}},
  author       = {Florian Lux and Julia Koch and Antje Schweitzer and Ngoc Thang Vu},
  booktitle    = {{Proc. Blizzard Challenge Workshop}},
  publisher    = {Speech Synthesis SIG},
  volume       = 2021
}
```

### Adding Articulatory Features and Meta-Learning Pretraining [[associated code and models]](https://github.com/DigitalPhonetics/IMS-Toucan/releases/tag/v1.1)

```
@inproceedings{lux2022laml,
  year         = 2022,
  title        = {{Language-Agnostic Meta-Learning for Low-Resource Text-to-Speech with Articulatory Features}},
  author       = {Florian Lux and Ngoc Thang Vu},
  booktitle    = {{Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics}},
  pages        = {6858--6868}
}
```

### Adding Exact Prosody-Cloning Capabilities [[associated code and models]](https://github.com/DigitalPhonetics/IMS-Toucan/releases/tag/v2.2)

```
@inproceedings{lux2022cloning,
  year         = 2022,
  title        = {{Exact Prosody Cloning in Zero-Shot Multispeaker Text-to-Speech}},
  author       = {Lux, Florian and Koch, Julia and Vu, Ngoc Thang},
  booktitle    = {{Proc. IEEE SLT}}
}
```

### Adding Language Embeddings and Word Boundaries [[associated code and models]](https://github.com/DigitalPhonetics/IMS-Toucan/releases/tag/v2.2)

```
@inproceedings{lux2022lrms,
  year         = 2022,
  title        = {{Low-Resource Multilingual and Zero-Shot Multispeaker TTS}},
  author       = {Florian Lux and Julia Koch and Ngoc Thang Vu},
  booktitle    = {{Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics}},
}
```
