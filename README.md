![image](Utility/toucan.png)

IMS Toucan is a toolkit for teaching, training and using state-of-the-art Speech Synthesis models, developed at the
**Institute for Natural Language Processing (IMS), University of Stuttgart, Germany**. Everything is pure Python and
PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

The basic PyTorch Modules of [FastSpeech 2](https://arxiv.org/abs/2006.04558) are taken from
[ESPnet](https://github.com/espnet/espnet), the PyTorch Modules of 
[HiFiGAN](https://arxiv.org/abs/2010.05646) are taken
from the [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN)
which are also authored by the brilliant [Tomoki Hayashi](https://github.com/kan-bayashi).

For a version of the toolkit that includes TransformerTTS, Tacotron 2 or MelGAN, check out
the other branches. They are separated to keep the code clean, simple and minimal as the development progresses.

---

## Demonstration 🦚

### Pre-Generated Audios
[Multi-lingual and multi-speaker audios](https://multilingualtoucan.github.io/)

[Cloning prosody across speakers](https://toucanprosodycloningdemo.github.io)

[Human-in-the-loop edited poetry for German literary studies](https://poetictts.github.io/).


### Interactive Demos
[Check out our multi-lingual demo on Huggingface🤗](https://huggingface.co/spaces/Flux9665/IMS-Toucan)

[Check out our demo on exact style cloning on Huggingface🤗](https://huggingface.co/spaces/Flux9665/SpeechCloning)

[Check out our human-in-the-loop poetry reading demo on Huggingface🤗](https://huggingface.co/spaces/Flux9665/PoeticTTS)


---

## New Features 🐣

- We officially introduced IMS Toucan in
  [our contribution to the Blizzard Challenge 2021](http://festvox.org/blizzard/bc2021/BC21_IMS.pdf).
- [As shown in this paper](http://festvox.org/blizzard/bc2021/BC21_DelightfulTTS.pdf) vocoders can be used to perform
  super-resolution and spectrogram inversion simultaneously. We added this to our HiFi-GAN vocoder. It now takes 16kHz
  spectrograms as input, but produces 48kHz waveforms.
- We now use articulatory representations of phonemes as the input for all models. This allows us to easily use 
  multilingual data to benefit less resource-rich languages. 
- We provide a checkpoint trained with a variant of model agnostic meta learning from which you should be able to 
  fine-tune a model with very little data in almost any language. The last two contributions are described in 
  [our paper that we will present at the ACL 2022](https://arxiv.org/abs/2203.03191)! 
- We now use a small self-contained Aligner that is trained with CTC and an auxiliary spectrogram reconstruction objective, inspired by
  [this implementation](https://github.com/as-ideas/DeepForcedAligner). 
- By conditioning the TTS on an ensemble of speaker embeddings as well an an embedding lookup table for language embeddings, 
  multi-lingual and multi-speaker models are possible. 
- We experimented with encoder designs and found one that allows speakers and languages to be very 
  disentangled, so you can use any speaker in any language, regardless of the language that the speakers themselves are speaking.
- Vocoders can also be used to do some slight speech-enhancement by corrupting a small percentage of their input spectrograms, 
  which we also added and experimented with.
- Exactly cloning the speaking style of a reference utterance is also possible and it works in conjunction with everything else! 
  So any utterance in any language spoken by any speaker can be replicated and controlled to allow for maximum customizability. 
  We apply this to literary studies.

**A pretrained checkpoint for our massively multi-lingual model and the self contained aligner is available in the release section.**

---

## Installation 🦉

#### Basic Requirements

To install this toolkit, clone it onto the machine you want to use it on
(should have at least one GPU if you intend to train models on that machine. For inference, you can get by without GPU).
Navigate to the directory you have cloned. We are going to create and activate a
[conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
to install the basic requirements into. After creating the environment, the command you need to use to activate the
virtual environment is displayed. The commands below show everything you need to do.

```
conda create --prefix ./toucan_conda_venv --no-default-packages python=3.8

pip install --no-cache-dir -r requirements.txt
```

The models are implemented and tested with PyTorch 1.9.0. Other versions (especially more recent ones) will most likely also work, I'm not aware of anything relevant that breaks backwards compatibility. To install torch 1.9.0, use the command appropriate for your system from [this link to the official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v190). 

#### Speaker Embedding

We use an ensemble of [Speechbrain's ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
and [Speechbrain's x-Vector](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) as the speaker conditioning.

In the current version of the toolkit no further action should be required. When you are using multispeaker for the
first time, it requires an internet connection to (automatically) download the pretrained models though.

#### espeak-ng

And finally you need to have espeak-ng installed on your system, because it is used as backend for the phonemizer. If
you replace the phonemizer, you don't need it. On most Linux environments it will be installed already, and if it is
not, and you have the sufficient rights, you can install it by simply running

```
apt-get install espeak-ng
```

#### Pretrained Models

You don't need to use pretrained models, but it can speed things up tremendously. Go into the release section and 
download the aligner model, the HiFiGAN model and the multi-lingual-multi-speaker FastSpeech2 model. Place them in 
*Models/Aligner/aligner.pt*, *Models/HiFiGAN_combined/best.pt* and *Models/FastSpeech2_Meta/best.pt*.

---

## Creating a new Pipeline 🦆

To create a new pipeline to train a HiFiGAN vocoder, you only need a set of audio files. To create a new pipeline for a
FastSpeech 2, you need audio files, corresponding text labels, and an already trained Aligner model to estimate the
duration information that FastSpeech 2 needs as input.

### Build a HiFi-GAN Pipeline

In the directory called
*Utility* there is a file called
*file_lists.py*. In this file you should write a function that returns a list of all the absolute paths to each of the
audio files in your dataset as strings.

Then go to the directory
*TrainingInterfaces/TrainingPipelines*. In there, make a copy of any existing pipeline that has HiFiGAN in its name. We
will use this as reference and only make the necessary changes to use the new dataset. Import the function you have just
written as
*get_file_list*. Now look out for a variable called
*model_save_dir*. This is the default directory that checkpoints will be saved into, unless you specify another one when
calling the training script. Change it to whatever you like.

Now you need to add your newly created pipeline to the pipeline dictionary in the file
*run_training_pipeline.py* in the top level of the toolkit. In this file, import the
*run* function from the pipeline you just created and give it a speaking name. Now in the
*pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense. And just like that
you're done.

### Build a FastSpeech 2 Pipeline

In the directory called
*Utility* there is a file called
*path_to_transcript_dicts.py*. In this file you should write a function that returns a dictionary that has all the
absolute paths to each of the audio files in your dataset as strings as the keys and the textual transcriptions of the
corresponding audios as the values.

Then go to the directory
*TrainingInterfaces/TrainingPipelines*. In there, make a copy of any existing pipeline that has FastSpeech 2 in its
name. We will use this copy as reference and only make the necessary changes to use the new dataset. Import the function
you have just written as
*build_path_to_transcript_dict*. Since the data will be processed a considerable amount, a cache will be built and saved
as file for quick and easy restarts. So find the variable
*cache_dir* and adapt it to your needs. The same goes for the variable
*save_dir*, which is where the checkpoints will be saved to. This is a default value, you can overwrite it when calling
the pipeline later using a command line argument, in case you want to fine-tune from a checkpoint and thus save into a
different directory.

In your new pipeline file, look out for the line in which the
*acoustic_model* is loaded. Change the path to the checkpoint of an Aligner model. It can either be the one that is
supplied with the toolkit on the release page, or one that you trained yourself. In the example pipelines, the one
that we provide is finetuned to the dataset it is applied to before it is used to extract durations.

Since we are using text here, we have to make sure that the text processing is adequate for the language. So check in
*Preprocessing/TextFrontend* whether the TextFrontend already has a language ID (e.g. 'en' and 'de') for the language of
your dataset. If not, you'll have to implement handling for that, but it should be pretty simple by just doing it
analogous to what is there already. Now back in the pipeline, change the
*lang* argument in the creation of the dataset and in the call to the train loop function to the language ID that
matches your data.

Now navigate to the implementation of the
*train_loop* that is called in the pipeline. In this file, find the function called
*plot_progress_spec*. This function will produce spectrogram plots during training, which is the most important way to
monitor the progress of the training. In there, you may need to add an example sentence for the language of the data you
are using. It should all be pretty clear from looking at it.

Once this is done, we are almost done, now we just need to make it available to the
*run_training_pipeline.py* file in the top level. In said file, import the
*run* function from the pipeline you just created and give it a speaking name. Now in the
*pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense. And that's it.

---

## Training a Model 🦜

Once you have a pipeline built, training is super easy. Just activate your virtual environment and run the command
below. You might want to use something like nohup to keep it running after you log out from the server (then you should
also add -u as option to python) and add an & to start it in the background. Also, you might want to direct the std:out
and std:err into a specific file using > but all of that is just standard shell use and has nothing to do with the toolkit.

```
python run_training_pipeline.py <shorthand of the pipeline>
```

You can supply any of the following arguments, but don't have to (although for training you should definitely specify at
least a GPU ID). It is recommended to download the pretrained checkpoint from the releases and use it as basis for 
fine-tuning for any new model that you train to significantly reduce training time.

```
--gpu_id <ID of the GPU you wish to use, as displayed with nvidia-smi, default is cpu> 

--resume_checkpoint <path to a checkpoint to load>

--resume (if this is present, the furthest checkpoint available will be loaded automatically)

--finetune (if this is present, the provided checkpoint will be fine-tuned on the data from this pipeline)

--model_save_dir <path to a directory where the checkpoints should be saved>
```

After every epoch, some logs will be written to the console. If the loss becomes NaN, you'll need to use a smaller
learning rate or more warmup steps in the arguments of the call to the training_loop in the pipeline you are running.

If you get cuda out of memory errors, you need to decrease the batchsize in the arguments of the call to the
training_loop in the pipeline you are running. Try decreasing the batchsize in small steps until you get no more out of
cuda memory errors. Decreasing the batchsize may also require you to use a smaller learning rate. The use of GroupNorm
should make it so that the training remains mostly stable.

Speaking of plots: in the directory you specified for saving model's checkpoint files and self-explanatory visualization
data will appear. Since the checkpoints are quite big, only the five most recent ones will be kept. Training will stop
after 500,000 for FastSpeech 2, and after 2,500,000 steps for HiFiGAN. Depending on the machine and configuration you
are using this will take multiple days, so verify that everything works on small tests before running the big thing. If
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

You can load your trained models using an inference interace. Simply instanciate it with the proper directory handle identifying the model you want to use, the rest should work out in the background. You might want to set a language embedding or a speaker embedding. The methods for that should be self-explanatory.

An *InferenceInterface* contains two useful methods. They are
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

---

This toolkit has been written by Florian Lux (except for the pytorch modules taken
from [ESPnet](https://github.com/espnet/espnet) and
[ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), as mentioned above), so if you come across problems
or questions, feel free to [write a mail](mailto:florian.lux@ims.uni-stuttgart.de). Also let me know if you do something
cool with it. Thank you for reading.

## Citation 🐧

### Introduction of the Toolkit
```
@inproceedings{lux2021toucan,
  title={{The IMS Toucan system for the Blizzard Challenge 2021}},
  author={Florian Lux and Julia Koch and Antje Schweitzer and Ngoc Thang Vu},
  year={2021},
  booktitle={Proc. Blizzard Challenge Workshop},
  volume={2021},
  publisher={{Speech Synthesis SIG}}
}
```

### Adding Articulatory Features and Meta-Learning Pretraining
```
@article{lux2022laml,
  title={{Language-Agnostic Meta-Learning for Low-Resource Text-to-Speech with Articulatory Features}},
  author={Florian Lux and Ngoc Thang Vu},
  year={2022},
  journal={arXiv preprint arXiv:2203.03191},
}
```
