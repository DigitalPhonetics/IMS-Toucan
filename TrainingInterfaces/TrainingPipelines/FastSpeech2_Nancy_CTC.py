import os
import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop_ctc import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_nancy as build_path_to_transcript_dict


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")
    cache_dir = os.path.join("Corpora", "Nancy")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Nancy_CTC")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "aligner"), exist_ok=True)

    path_to_transcript_dict = build_path_to_transcript_dict()
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    """
    if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")):
        print("Training aligner")
        train_aligner(train_dataset=AlignerDataset(path_to_transcript_dict,
                                                   cache_dir=cache_dir,
                                                   lang="en"),
                      device=device,
                      save_directory=os.path.join(save_dir, "aligner"),
                      steps=10000,
                      batch_size=32,
                      path_to_checkpoint="Models/Aligner/aligner.pt",
                      fine_tune=True,
                      debug_img_path=save_dir_aligner,
                      resume=resume)

    acoustic_checkpoint_path = os.path.join(save_dir, "aligner", "aligner.pt")

    print("Preparing Dataset")
    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  cache_dir=cache_dir,
                                  acoustic_checkpoint_path=acoustic_checkpoint_path,
                                  lang="en",
                                  device=device)
    """

    model = FastSpeech2()

    train_sents = """Believe me.
Call me back.
As soon as possible
Do me a favor
Give me a hand
I do not understand
I do not mean it
I decline!
I’m on a diet
I just made it
I’m sorry
Absolutely not.
I have no idea.
I agree.
I’m at home
It’s on the tip of my tongue
It’s ok
It really takes time
It’s fort he best
No, I don’t want
See you
See you next time
So I do
So so
Allow me
Any day will do
Be calm
Be careful!
Be quiet!
Cheer up!
Come on
Don’t be ridicolus
Don’t be so childish
Don’t move!
Don’t worry
Enjoy yourself
Follow me
Forgive me
Forget it
God bless you
It’s very thoughtful of you
It’s up to you
It’s none of you business
Shut up
Slow down
Stop making such a noice
You are going too fast
Come with me
Good afternoon
Good morning
Good night
Have a good trip
Have a good weekend
I admire you
I apologize
I can’t wait
I don’t have time
I got it.
I hate you!
I hope so.
I knew it.
I love you.
I would love to.
I am busy.
I am tired.
I don’t agree.
You are wasting my time.
I feel much better.
They like each other.
I’m sorry.
I’m good.
It doesn’t matter.
Join me.
Let’s catch up!
Let’s do it.
Nice to meet you.
Not yet.
Talk to you tomorrow.
Thank you very much.
You turn.
A lovely day, isn’t it?
Do i have to?
Can I help you?
How are things going?
Any thing else?
Are you kidding?
Are you sure?
Do you understand me?
Are you done?
Can I ask you something?
Can you please repeat that?
Did you get it?
Do you need anything?
How are you?
How do you feel?
How much is it?
How old are you?
How was your weekend?
Is all good?
Is everything OK?
What are you doing?
What are you talking about?
What are you up to?
What are your hobbies?
What did you say?
What do you need?
What do you think?
What do you want to do?
What do you want?
What’s the weather like?
What’s your e-mail address?
What is your job?
What’s your name?
What’s your phone number?
What is going on?
When is the train leaving?
How can I go to the town centre?
Where are you from?
Where are you going?
Where are you?
Where did you get it?
Where do you live?
Are you coming with me?
How long will you stay?""".split("\n")

    print("Training model")
    train_loop(net=model,
               train_sentences=train_sents,
               device=device,
               save_directory=save_dir,
               aligner_checkpoint=os.path.join("Models", "Aligner", "aligner.pt"),
               steps=500000,
               batch_size=32,
               lang="en",
               lr=0.0001,
               warmup_steps=14000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume)
