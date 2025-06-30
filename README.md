<p align="right">
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/DigitalPhonetics/IMS-Toucan">
<img alt="GitHub Repo Downloads" src="https://img.shields.io/github/downloads/DigitalPhonetics/IMS-Toucan/total">
<img alt="GitHub Release" src="https://img.shields.io/github/v/release/DigitalPhonetics/IMS-Toucan">
<a href=https://huggingface.co/spaces/Flux9665/MassivelyMultilingualTTS><img alt="Demo Link" src="https://img.shields.io/badge/DEMO-<COLOR>.svg"></a>
</p>

---

This branch contains the associated code for our paper **Investigating Stochastic Methods for Prosody Modeling in Speech Synthesis**. It will be published in **Interspeech 2025**. It is a collaboration between scientists from **AppTek** and the **University of Stuttgart**.

Author List: **Paul Mayer, Florian Lux, Alejandro Pérez-González-de-Martos, Angelina Elizarova, Lindsey Vanderlyn, Dirk Väth, Ngoc Thang Vu**

--- 
<br>

## Installation 🦉

#### Basic Requirements

Python 3.10 is the recommended version.

To install this toolkit, clone it onto the machine you want to use it on
(should have at least one cuda enabled GPU if you intend to train models on that machine. For inference, you don't need
a GPU).

If you're using Linux, you should have the following packages installed, or install them with apt-get if you haven't (on
most distributions they come pre-installed):

```
libsndfile1
espeak-ng
ffmpeg
libasound-dev
libportaudio2
libsqlite3-dev
```

Navigate to the directory you have cloned. We recommend creating and activating a
[virtual environment](https://docs.python.org/3/library/venv.html)
to install the basic requirements into. The commands below summarize everything you need to do under Linux. If you are
running Windows, the second line needs to be changed, please have a look at
the [venv documentation](https://docs.python.org/3/library/venv.html).

```
python -m venv <path_to_where_you_want_your_env_to_be>

source <path_to_where_you_want_your_env_to_be>/bin/activate

pip install --no-cache-dir -r requirements.txt
```

Run the second line everytime you start using the tool again to activate the virtual environment again, if you e.g.
logged out in the meantime. To make use of a GPU, you don't need to do anything else on a Linux machine. On a Windows
machine, have a look at [the official PyTorch website](https://pytorch.org/) for the install-command that enables GPU
support.

#### \[optional] eSpeak-NG

eSpeak-NG is an optional requirement, that handles lots of special cases in many languages, so it's good to have.

On most **Linux** environments it will be installed already, and if it is not, and you have the sufficient rights, you
can install it by simply running

```
apt-get install espeak-ng
```

For **Windows**, they provide a convenient .msi installer file
[on their GitHub release page](https://github.com/espeak-ng/espeak-ng/releases). After installation on non-linux
systems, you'll also need to tell the phonemizer library where to find your espeak installation by setting the
`PHONEMIZER_ESPEAK_LIBRARY` environment variable, which is discussed in
[this issue](https://github.com/bootphon/phonemizer/issues/44#issuecomment-1008449718).

For **Mac** you can use homebrew

```
brew install espeak-ng
```

As stated in the Windows install instructions, the espeak-ng installation will need to be set as a variable for the
phonemizer library. The environment variable is `PHONEMIZER_ESPEAK_LIBRARY` as given in the
[GitHub thread](https://github.com/bootphon/phonemizer/issues/44#issuecomment-1008449718) linked above.
However, the espeak-ng installation file you need to set this variable to is a .dylib file rather than a .dll file on
Mac. Locate the espeak-ng library file; it is named `libespeak-ng.dylib`.

--- 
<br>


## Training a Model 🦜

```
python run_training_pipeline.py <shorthand of the pipeline>
```

You can supply any of the following arguments, but don't have to (although for training you should definitely specify at
least a GPU ID).

```
--gpu_id <ID of the GPU you wish to use, as displayed with nvidia-smi, default is cpu. If multiple GPUs are provided (comma separated), then distributed training will be used, but the script has to be started with torchrun.> 

--resume_checkpoint <path to a checkpoint to load>

--resume (if this is present, the furthest checkpoint available will be loaded automatically)

--finetune (if this is present, the provided checkpoint will be fine-tuned on the data from this pipeline)

--model_save_dir <path to a directory where the checkpoints should be saved>

--wandb (if this is present, the logs will be synchronized to your weights&biases account, if you are logged in on the command line)

--wandb_resume_id <the id of the run you want to resume, if you are using weights&biases (you can find the id in the URL of the run)>
```

After every epoch (or alternatively after certain step counts), some logs will be written to the console and to the
Weights and Biases website, if you are logged in and set the flag. If you get cuda out of memory errors, you need to
decrease
the batchsize in the arguments of the call to the training_loop in the pipeline you are running. Try decreasing the
batchsize in small steps until you get no more out of cuda memory errors.

In the directory you specified for saving, checkpoint files and spectrogram visualization
data will appear. Since the checkpoints are quite big, only the five most recent ones will be kept. The amount of
training steps highly depends on the data you are using and whether you're finetuning from a pretrained checkpoint or
training from scratch. The fewer data you have, the fewer steps you should take to prevent a possible collapse. If
you want to stop earlier, just kill the process, since everything is daemonic all the child-processes should die with
it. In case there are some ghost-processes left behind, you can use the following command to find them and kill them
manually.

```
fuser -v /dev/nvidia*
```

Whenever a checkpoint is saved, a compressed version that can be used for inference is also created, which is named
_best.py_

### Configuring the Prosody Modeling

You can customize key parameters for the probabilistic prosody model directly in the `ToucanTTS_Prosody.py` file:

```
order = "ped"  # Order of prosodic features: pitch, energy, duration. Options: "ped", "epd", or "all"
prosody_channels = 8  # Number of channels for the prosody predictor
predictor_layers = 3  # Number of layers in the prosody predictor
predictor_kernel_size = 5  # Kernel size for convolutional layers
predictor_dropout_rate = 0.2  # Dropout rate within the predictor
architecture = "CFM"  # Architecture type: "CFM", "NF", "RF" or "DET"
start_reflow = 91000  # Step count to start reflow; if set higher than current step count, reflow is skipped
dropout = False  # Apply inference-dropout on the encoder outputs in the model
log = False  # Enable logging within this component
```

You can modify these values to experiment with different architectures.

---
<br> 

## Evaluation 🐤

To run evaluation on a trained model, use the following command:

```
python run_evaluation.py
```
You can supply any of the following arguments to customize the evaluation process:

```
--model_dir <Path to the parent directory of all models to evaluate. This should contain the checkpoints and metadata.>

--version <Identifier for the run, useful for distinguishing between different evaluation sets or experiments.>

--gpu_id <Which GPU(s) to use, e.g. 0 or 0,1. If not specified or set to "cpu", evaluation will run on CPU (which is generally only suitable for quick tests or debugging).>

--wandb (if this is present, the results will be tracked on your Weights & Biases account. Make sure you're logged in with `wandb login` beforehand.)

--multi_speaker (if this is present, evaluation will account for multiple speaker identities — useful if your model was trained with multiple voices.)
```

Typical usage looks like this:

```
python run_eval.py --model_dir ./checkpoints/my_model --version eval_v1 --gpu_id 0 --wandb
```

If --wandb is used, the results will be logged to your Weights & Biases dashboard.


--- 
<br>


## Inference 🦢

You can load your trained models, or the pretrained provided one, using the `InferenceInterfaces/ToucanTTSInterface.py`.
Simply create an object from it with the proper directory handle
identifying the model you want to use. The rest should work out in the background. You might want to set a language
embedding or a speaker embedding using the *set_language* and *set_speaker_embedding* functions. Most things should be
self-explanatory.

An *InferenceInterface* contains two methods to create audio from text. They are
*read_to_file* and
*read_aloud*.

- *read_to_file* takes as input a list of strings and a filename. It will synthesize the sentences in the list and
  concatenate them with a short pause inbetween and write them to the filepath you supply as the other argument.

- *read_aloud* takes just a string, which it will then convert to speech and immediately play using the system's
  speakers. If you set the optional argument
  *view* to
  *True*, a visualization will pop up, that you need to close for the program to continue.


There are simple scaling parameters to control the duration, the variance of the pitch curve and the variance of the
energy curve. You can either change them in the code when using the interactive demo or the reader, or you can simply
pass them to the interface when you use it in your own code.

To change the language of the model and see which languages are available in our pretrained model,
[have a look at the list linked here](https://github.com/DigitalPhonetics/IMS-Toucan/blob/feb573ca630823974e6ced22591ab41cdfb93674/Utility/language_list.md)

--- 
<br>

## Citation 🐧

```
@inproceedings{mayer2025stochastic,
  year         = 2025,
  title        = {{Investigating Stochastic Methods for Prosody Modeling in Speech Synthesis}},
  author       = {Paul Mayer and Florian Lux and Alejandro P\'erez-Gonz\'alez-de-Martos and Angelina Elizarova and Lindsey Vanderlyn and Dirk V\"ath and Ngoc Thang Vu},
  booktitle    = {Interspeech}
  publisher    = {ISCA}
}
```
