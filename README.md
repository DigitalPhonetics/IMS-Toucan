# Speech Synthesis 🐣

This is a toolkit to train state-of-the-art Speech Synthesis models. Everything is pure Python and PyTorch based to keep
it as simple and beginner-friendly, yet powerful as possible.

The PyTorch Modules of [TransformerTTS](https://arxiv.org/abs/1809.08895)
and [FastSpeech2](https://arxiv.org/abs/2006.04558) are taken from [ESPnet](https://github.com/espnet/espnet), the
PyTorch Modules of [MelGAN](https://arxiv.org/abs/1910.06711) are taken from
the [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN) which is also made by the
brillant [Tomoki Hayashi](https://github.com/kan-bayashi).

## Demonstration

[Here is some speech](https://drive.google.com/file/d/1mZ1LvTlY6pJ5ZQ4UXZ9jbzB651mufBrB/view?usp=sharing) produced by
FastSpeech 2 and MelGAN trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) using this toolkit.

And [here is a sentence](https://drive.google.com/file/d/1FT49Jf0yyibwMDbsEJEO9mjwHkHRIGXc/view?usp=sharing) produced by
TransformerTTS and MelGAN trained on [Thorsten](https://github.com/thorstenMueller/deep-learning-german-tts) using this
toolkit.

[Here is some speech](https://drive.google.com/file/d/14nPo2o1VKtWLPGF7e_0TxL8XGI3n7tAs/view?usp=sharing) produced by a
multi-speaker FastSpeech 2 with MelGAN trained on [LibriTTS](https://research.google/tools/datasets/libri-tts/) using
this toolkit. Fans of the videogame Portal may recognize who was used as the reference speaker for this utterance.

## Embrace Redundancy

While it is a bad practice to have redundancy in regular code, and as much as possible should be abstracted and
parameterized, my experiences in creating this toolkit and working with other toolkits led me to believe that sometimes
redundancy is not only ok, it is actually very convenient. While it does make it more difficult to change things, it
also makes it also more difficult to break things and cause legacy problems.

---

## Working with this Toolkit

The standard way of working with this toolkit is to make your own fork of it, so you can change as much of the code as
you like and fully adapt it to your needs. Making pipelines to train models on new datasets, even in new languages,
requires absolutely minimal new code and you can take the existing code for such models as reference/template.

## Installation
To install this toolkit, clone it onto the machine you want to use it on (should have at least one GPU if you intend to train models on that machine. For inference you can get by without GPU). Navigate to the directory you have cloned and run the command shown below. It is recommended to first create and activate a [pip virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). 

```
pip install -r requirements.txt 
```

If you want to use multi-speaker synthesis, you will need a speaker embedding function. The one assumed in the code is [dvector](https://github.com/yistLin/dvector), because it is incredibly easy to use and freely available. Create a directory *Models* in the top-level of your clone. Then create a directory *Use* in there and in this directory create another directory called *SpeakerEmbedding*. In this directory you put the two files *wav2mel.pt* and *dvector-step250000.pt* that you can obtain from the release page of the [dvector](https://github.com/yistLin/dvector) GitHub. This process might become automated in the future.

And finally you need to have espeak installed on your system, because it is used as backend for the phonemizer. If you replace the phonemizer, you don't need it. On most Linux environments it will be installed already, and if it is not and you have the sufficient rights you can install it by simply running 

```
apt-get install espeak
```

## Creating a new Pipeline
To create a new pipeline to train a MelGAN vocoder, you only need a set of audio files. To create a new pipeline for a TransformerTTS you need audio files and corresponding text labels. To create a new pipeline for a FastSpeech 2, you need audio files, corresponding text labels, and an already trained TransformerTTS to estimate the duration information that FastSpeech 2 needs as input. Let's go through them in order of increasing complexity.

#### Build a MelGAN Pipeline
In the directory called *Utility* there is a file called *file_lists.py*. In this file you should write a function that returns a list of all of the absolute paths to each of the audio files in your dataset as strings. 

Then go to the directory *Pipelines* from the top level of the toolkit. In there, make a copy of any existing pipeline that has MelGAN in its name. We will use this as reference and only make the necessary changes to use the new dataset. Import the function you have just written as *get_file_list*. Now look out for a variable called *model_save_dir*. This is the default directory that checkpoints will be saved into, unless you specify another one when calling the training script. Change it to whatever you like. 

Now you need to add your newly created pipeline to the the pipeline dictionary in the file *run_training_pipeline.py* in the top level of the toolkit. In this file, import the *run* function from the pipeline you just created and give it a speaking name. Now in the *pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense. And just like that you're done.

#### Build a TransformerTTS Pipeline
In the directory called *Utility* there is a file called *path_to_transcript_dicts.py*. In this file you should write a function that returns a dictionary that has all of the absolute paths to each of the audio files in your dataset as strings as the keys and the textual transcriptions of the corresponding audios as the values. 

Then go to the directory *Pipelines* from the top level of the toolkit. In there, make a copy of any existing pipeline that has TransformerTTS in its name. If your dataset is single-speaker, choose any that is not LibriTTS. If your dataset is multi-speaker, choose the one for LibriTTS as your template. We will use this copy as reference and only make the necessary changes to use the new dataset. Import the function you have just written as *build_path_to_transcript_dict*. Since the data will be processed a considerable amount, a cache will be built and saved as file for quick and easy restarts. So find the variable *cache_dir* and adapt it to your needs. The same goes for the variable *save_dir*, which is where the checkpoints will be saved to. This is a default value, you can overwrite it when calling the pipeline later using a command line argument, in case you want to fine-tune from a checkpoint and thus save into a different directory. 

Since we are using text here, we have to make sure that the text processing is adequate for the language. So check in *PreprocessingForTTS/ProcessText* whether the TextFrontend already has a language ID (e.g. 'en' and 'de') for the language of your dataset. If not, you'll have to implement handling for that, but it should be pretty simple by just doing it analogous to what is there already. Now back in the pipeline, change the *lang* argument in the creation of the dataset and in the call to the train loop function to the language ID that matches your data. 

Now navigate to the implementation of the *train_loop* that is called in the pipeline. In this file, find the function called *get_atts*. This function will produce attention plots during training, which is the most important way to monitor the progress of the training. In there, you may need to add an example sentence for the language of the data you are using. It should all be pretty clear from looking at it. 

Once this is done, we are almost done, now we just need to make it available to the *run_training_pipeline.py* file in the top level. In said file, import the *run* function from the pipeline you just created and give it a speaking name. Now in the *pipeline_dict*, add your imported function as value and use as key a shorthand that makes sense. And that's it.

#### Build a FastSpeech 2 Pipeline
Most of this is exactly analogous to building a TransformerTTS pipeline. So to keep this brief, this section will only mention the additional things you have to do. 

In your new pipeline file, look out for the line in which the *acoustic_model* is loaded. Change the path to the checkpoint of a TransformerTTS model that you trained on the same dataset previously. Then look out for the creation of the *train_set*. In there, there is an argument called *diagonal_attention_head_id*. 

It is recommended to use an InferenceInterface of the aforementioned TransformerTTS model to determine which of the attention heads looks the most like a duration graph. For this, add the InferenceInterface to the dictionary in the *view_attention_heads* function in the *run_visualizations.py* file in the top level of the toolkit. Then call it to see a plot of all of the attention heads visualized with their ID displayed above them. This ID is what you want to supply to the *diagonal_attention_head_id* argument in the pipeline as an integer. If you use the default argument (None) it will try to select the most diagonal head for each sample automatically, but this fails for some samples, so it is safer to do it manually. 

Everything else is exactly like creating a TransformerTTS pipeline, except that in the training_loop, instead of attentions plots, spectrograms are plotted to visualize training progress. So there you may need to add a sentence if you are using a new language in the function called *plot_progress_spec*.

## Training a Model
Once you have a pipeline built, training is super easy. Just activate your virtual environment and run the command below. You might want to use something like nohup to keep it running after you log out from the server (then you should also add -u as option to python) and add an & to start it in the background. Also you might want to direct the std:out and std:err into a file using > but all of that is just standard shell use and has nothing to to with the toolkit.

```
python run_training_pipeline.py <shorthand of the pipeline>
```

You can supply any of the following arguments, but don't have to (although for training you should definitely specify at least a GPU ID).

```
--gpu_id <ID of the GPU you wish to use, as displayed with nvidia-smi, default is cpu> 

--resume_checkpoint <path to a checkpoint to load>

--finetune (if this is present, the provided checkpoint will be fine-tuned on the data from this pipeline)

--model_save_dir <path to a directory where the checkpoints should be saved>
```

After every epoch, some logs will be written to the console. If the loss becomes NaN, you'll need to use a smaller learning rate or more warmup steps in the arguments of the call to the training_loop in the pipeline you are running. 

If you get cuda out of memory errors, you need to decrease the batchsize in the arguments of the call to the training_loop in the pipeline you are running. You can compensate for the smaller batchsize by increasing the gradient_accumulation argument. This will cause an update to the weigths to only happen after the specified amount of batches. This is important for training stability, but makes training a lot slower. So try decreasing the batchsize until you get no more out of cuda memory errors, and only increase the gradient_accumulation if training fails then due to the smaller batch size (i.e. the plots don't show improvments after a few hours).

Speaking of plots, in the directory you specified for saving models checkpoints and self explanatory visualization data will appear. Since the checkpoints are quite big, only the five most recent ones will be kept. Training will stop after 400,000 update steps have been made by default for TransformerTTS and FastSpeech2, and after 2,000,000 steps for MelGAN. Depending on the machine and configuration you are using this will take between 2 and 4 days, so verify that everything works on small tests before running the big thing. If you want to stop earlier, just kill the process, since everything is daemonic all of the child-processes should die with it.

After training is complete, it is recommended to run *run_weight_averaging.py*. If you made no changes to the architectures and stuck to the default directory layout, it will automatically load any models you produced with one pipeline, average their parameters to get a slightly more robust model and save the result as *best.pt* in the same directory where all of the corresponding checkpoints lie. This also compresses the file size slightly, so you should do this and then use the *best.pt* model for inference.

## Creating a new InferenceInterface
To build a new *InferenceInterface*, which you can then use for super simple inference, we're going to use an existing one as template again. If you use multi-speaker, take the LibriTTS ones as template, otherwise take any other one. Make a copy of the InferenceInterface. Change the name of the class in the copy and change the paths to the models to use the trained models of your choice. Instantiate the model with the same hyperparameters that you used when you created it in the corresponding training pipeline. The last thing to check is the language that you supply to the text frontend. Make sure it matches what you used during training. 

With your newly created InferenceInterface, you can use your trained models pretty much anywhere, e.g. in other projects. All you need is the *Utility* directory, the *Layers* directory, the *PreprocessingForTTS* directory and the *InferenceInterfaces* directory (and of course your model checkpoint). That's all the code you need, it works standalone.

## Using a trained Model for Inference
A *InferenceInterface* of TransformerTTS contains 3 useful methods for inference, a FastSpeech 2 one contains 2 useful methods. The ones they share are *read_to_file* and *read_aloud*. 

- The first one takes as input a list of strings and a filename. It will synthesize the sentences in the list and concatenate them with a short pause inbetween and write them to the filepath you supply as the other argument. 
- The second one takes just a string, which it will then convert to speech and immediately play using the system's speakers. If you set the optional argument *view* to *True* when calling it, it will also show a plot of the phonemes it produced, the spectrogram it came up with and the wave it created from that spectrogram. So all of the representations can be seen, text to phoneme, phoneme to spectrogram and finally spectrogram to wave. 
- The one that is exclusive to TransformerTTS is *plot_attentions*. This will take a string, synthesize it and show a plot of the attention matrices of all of the attention heads in all of the attention layers. By default there are 24. The index of each head is displayed above (0-23 by default). This is useful to figure out which one of the heads contains the best temporal relation between text and audio, which is the ID that should be given to the FastSpeech 2 feature-extraction to herustically get duration information using a technique called *knowledge-distillation*.

Those three methods are used in demo code in the toolkit. In *run_interactive_demo.py*, *run_text_to_file_reader.py* and *run_visualizations.py* you can import *InferenceInterfaces* that you created and add them to the dictionary in each of the files with a shorthand that makes sense. In the interactive demo, you can just call the python script, then type in the shorthand when prompted and immediately listen to your synthesis saying whatever you put in next (be wary of out of memory errors for too long inputs). In the other two demo scripts you have to call the function that wraps around the *InferenceInterface* and supply the shorthand of your choice. It should be pretty clear from looking at it.

---

## Example Pipelines available

| Dataset               | Language  | Single or Multi     | MelGAN | TransformerTTS | FastSpeech2 | 
| ----------------------|-----------|---------------------| :-----:|:--------------:|:-----------:|
| Hokuspokus            | German    | Single Speaker      | ✅     | ✅            | ✅          |
| Thorsten              | German    | Single Speaker      | ✅     | ✅            | ✅          |
| MAILabs Karlsson      | German    | Single Speaker      | ✅     | ✅            | ✅          |
| MAILabs Eva           | German    | Single Speaker      | ✅     | ✅            | ✅          |
| LJSpeech              | English   | Single Speaker      | ✅     | ✅            | ✅          |
| Nancy Krebs Lessac    | English   | Single Speaker      | ✅     | ✅            | ✅          |
| MAILabs Elizabeth     | English   | Single Speaker      | ✅     | ✅            | ✅          |
| LibriTTS              | English   | Multi Speaker       | ✅     | ✅            | ✅          |

---

This toolkit has been written by Florian Lux (except for the pytorch modules taken from [ESPnet](https://github.com/espnet/espnet) and [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), as mentioned above), so if you come across problems or questions, feel free to write a mail to [florian.lux@ims.uni-stuttgart.de](mailto:florian.lux@ims.uni-stuttgart.de). Also let me know if you do something cool with it. Thank you for reading.
