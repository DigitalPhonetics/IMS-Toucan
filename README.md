![image](Utility/toucan.png)

IMS Toucan is a toolkit for teaching, training and using state-of-the-art Speech Synthesis models, developed at the
**Institute for Natural Language Processing (IMS), University of Stuttgart, Germany**. Everything is pure Python and
PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

The PyTorch Modules of [Tacotron 2](https://arxiv.org/abs/1712.05884)
and [FastSpeech 2](https://arxiv.org/abs/2006.04558) are taken from
[ESPnet](https://github.com/espnet/espnet), the PyTorch Modules of [HiFiGAN](https://arxiv.org/abs/2010.05646) are taken
from the [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN)
which are also authored by the brilliant [Tomoki Hayashi](https://github.com/kan-bayashi).

For a version of the toolkit that includes TransformerTTS instead of Tacotron 2 and MelGAN instead of HiFiGAN, check out
the TransformerTTS and MelGAN branch. They are separated to keep the code clean, simple and minimal.

## Demonstration

[Here are two sentences](https://drive.google.com/file/d/1ltAyR2EwAbmDo2hgkx1mvUny4FuxYmru/view?usp=sharing)
produced by Tacotron 2 combined with HiFi-GAN, trained on
[Nancy Krebs](https://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/) using this toolkit.

[Here is some speech](https://drive.google.com/file/d/1mZ1LvTlY6pJ5ZQ4UXZ9jbzB651mufBrB/view?usp=sharing)
produced by FastSpeech 2 and MelGAN trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
using this toolkit.

And [here is a sentence](https://drive.google.com/file/d/1FT49Jf0yyibwMDbsEJEO9mjwHkHRIGXc/view?usp=sharing)
produced by TransformerTTS and MelGAN trained on [Thorsten](https://github.com/thorstenMueller/deep-learning-german-tts)
using this toolkit.

[Here is some speech](https://drive.google.com/file/d/14nPo2o1VKtWLPGF7e_0TxL8XGI3n7tAs/view?usp=sharing)
produced by a multi-speaker FastSpeech 2 with MelGAN trained on
[LibriTTS](https://research.google/tools/datasets/libri-tts/) using this toolkit. Fans of the videogame Portal may
recognize who was used as the reference speaker for this utterance.

[Interactive Demo of our entry to the Blizzard Challenge 2021.](https://colab.research.google.com/drive/1bRaySf8U55MRPaxqBr8huWrzCOzlxVqw)
This is based on an older version of the toolkit though. It uses FastSpeech2 and MelGAN as vocoder and is trained on 5
hours of Spanish.

---

## Installation

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

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Speaker Embedding

As [NVIDIA has shown](https://arxiv.org/pdf/2110.05798.pdf), you get better results by fine-tuning a pretrained model on
a new speaker, rather than training a multispeaker model. We have thus dropped support for zero-shot multispeaker models
using speaker embeddings. However we still
use [Speechbrain's ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) for a cycle consistency loss to
make adapting to new speakers a bit faster.

In the current version of the toolkit no further action should be required. When you are using multispeaker for the
first time, it requires an internet connection to download the pretrained models though.

#### espeak-ng

And finally you need to have espeak-ng installed on your system, because it is used as backend for the phonemizer. If
you replace the phonemizer, you don't need it. On most Linux environments it will be installed already, and if it is
not, and you have the sufficient rights, you can install it by simply running

```
apt-get install espeak-ng
```

## Creating a new Pipeline

To create a new pipeline to train a HiFiGAN vocoder, you only need a set of audio files. To create a new pipeline for a
Tacotron 2 you need audio files and corresponding text labels. To create a new pipeline for a FastSpeech 2, you need
audio files, corresponding text labels, and an already trained Tacotron 2 model to estimate the duration information
that FastSpeech 2 needs as input. Let's go through them in order of increasing complexity.

#### Build a HiFiGAN Pipeline

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

#### Build a Tacotron 2 Pipeline

In the directory called
*Utility* there is a file called
*path_to_transcript_dicts.py*. In this file you should write a function that returns a dictionary that has all the
absolute paths to each of the audio files in your dataset as strings as the keys and the textual transcriptions of the
corresponding audios as the values.

Then go to the directory
*TrainingInterfaces/TrainingPipelines*. In there, make a copy of any existing pipeline that has Tacotron 2 in its name.
We will use this copy as reference and only make the necessary changes to use the new dataset. Import the function you
have just written as
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

Now navigate to the implementation of the
*train_loop* that is called in the pipeline. In this file, find the function called
*plot_attention*. This function will produce attention plots during training, which is the most important way to monitor
the progress of the training. In there, you may need to add an example sentence for the language of the data you are
using. It should all be pretty clear from looking at it.

Once this is done, we are almost done, now we just need to make it available to the
*run_training_pipeline.py* file in the top level. In said file, import the
*run* function from the pipeline you just created and give it a speaking name. Now in the
*pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense. And that's it.

#### Build a FastSpeech 2 Pipeline

Most of this is exactly analogous to building a Tacotron 2 pipeline. So to keep this brief, this section will only
mention the additional things you have to do.

In your new pipeline file, look out for the line in which the
*acoustic_model* is loaded. Change the path to the checkpoint of a Tacotron 2 model that you trained on the same dataset
previously. This is used to estimate phoneme-durations based on knowledge-distillation.

Everything else is exactly like creating a Tacotron 2 pipeline, except that in the training_loop, instead of attentions
plots, spectrograms are plotted to visualize training progress. So there you may need to add a sentence if you are using
a new language in the function called
*plot_progress_spec*.

## Training a Model

Once you have a pipeline built, training is super easy. Just activate your virtual environment and run the command
below. You might want to use something like nohup to keep it running after you log out from the server (then you should
also add -u as option to python) and add an & to start it in the background. Also, you might want to direct the std:out
and std:err into a file using > but all of that is just standard shell use and has nothing to do with the toolkit.

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
```

After every epoch, some logs will be written to the console. If the loss becomes NaN, you'll need to use a smaller
learning rate or more warmup steps in the arguments of the call to the training_loop in the pipeline you are running.

If you get cuda out of memory errors, you need to decrease the batchsize in the arguments of the call to the
training_loop in the pipeline you are running. Try decreasing the batchsize in small steps until you get no more out of
cuda memory errors. Decreasing the batchsize may also require you to use a smaller learning rate. The use of GroupNorm
should make it so that the training remains mostly stable.

Speaking of plots: in the directory you specified for saving model's checkpoint files and self-explanatory visualization
data will appear. Since the checkpoints are quite big, only the five most recent ones will be kept. Training will stop
after 100,000 update steps have been made by default for Tacotron 2, 300,000 for FastSpeech 2, and after 500,000 steps
for HiFiGAN. Depending on the machine and configuration you are using this will take between 2 and 4 days, so verify
that everything works on small tests before running the big thing. If you want to stop earlier, just kill the process,
since everything is daemonic all the child-processes should die with it.

After training is complete, it is recommended to run
*run_weight_averaging.py*. If you made no changes to the architectures and stuck to the default directory layout, it
will automatically load any models you produced with one pipeline, average their parameters to get a slightly more
robust model and save the result as
*best.pt* in the same directory where all the corresponding checkpoints lie. This also compresses the file size
slightly, so you should do this and then use the
*best.pt* model for inference.

## Creating a new InferenceInterface

To build a new
*InferenceInterface*, which you can then use for super simple inference, we're going to use an existing one as template
again. Make a copy of the
*InferenceInterface*. Change the name of the class in the copy and change the paths to the models to use the trained
models of your choice. Instantiate the model with the same hyperparameters that you used when you created it in the
corresponding training pipeline. The last thing to check is the language that you supply to the text frontend. Make sure
it matches what you used during training.

With your newly created
*InferenceInterface*, you can use your trained models pretty much anywhere, e.g. in other projects. All you need is the
*Utility* directory, the
*Layers*
directory, the
*Preprocessing* directory and the
*InferenceInterfaces* directory (and of course your model checkpoint). That's all the code you need, it works
standalone.

## Using a trained Model for Inference

An
*InferenceInterface* contains 2 useful methods. They are
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

- Additionally, Tacotron 2
  *InferenceInterfaces* offer a method called
  *plot_attention*. This will take a string, synthesize it and show a plot of the attention matrix, which can be useful
  to gain insights.

Those methods are used in demo code in the toolkit. In
*run_interactive_demo.py* and
*run_text_to_file_reader.py*, you can import
*InferenceInterfaces* that you created and add them to the dictionary in each of the files with a shorthand that makes
sense. In the interactive demo, you can just call the python script, then type in the shorthand when prompted and
immediately listen to your synthesis saying whatever you put in next (be wary of out of memory errors for too long
inputs). In the text reader demo script you have to call the function that wraps around the
*InferenceInterface* and supply the shorthand of your choice. It should be pretty clear from looking at it.

## FAQ

Here are a few points that were brought up by users:

- My error message shows GPU0, even though I specified a different GPU - The way GPU selection works is that the
  specified GPU is set as the only visible device, in order to avoid backend stuff running accidentally on different
  GPUs. So internally the program will name the device GPU0, because it is the only GPU it can see. It is actually
  running on the GPU you specified.

---

## Example Pipelines available

| Dataset                                                                              | Language | Single or Multi | TransformerTTS | Tacotron 2 | FastSpeech 2 | 
| -------------------------------------------------------------------------------------|----------|-----------------|:--------------:|:---------:|:-----------:|
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)                                  | English  | Single Speaker | ✅              | ✅        |✅           |
| [Nancy Krebs](https://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/) | English  | Single Speaker | ✅              | ✅        |✅           |

---

This toolkit has been written by Florian Lux (except for the pytorch modules taken
from [ESPnet](https://github.com/espnet/espnet) and
[ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), as mentioned above), so if you come across problems
or questions, feel free to [write a mail](mailto:florian.lux@ims.uni-stuttgart.de). Also let me know if you do something
cool with it. Thank you for reading.
