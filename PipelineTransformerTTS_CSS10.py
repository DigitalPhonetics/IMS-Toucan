"""
Train an autoregressive Transformer TTS model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time

import soundfile as sf
import torch
import torchviz
from torch.nn.utils.rnn import pad_sequence

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset

torch.manual_seed(17)
random.seed(17)


class CSS10SingleSpeakerFeaturizer():
    def __init__(self):
        self.tf = TextFrontend(language="en",
                               use_panphon_vectors=False,
                               use_shallow_pos=False,
                               use_sentence_type=False,
                               use_positional_information=False,
                               use_word_boundaries=False,
                               use_chinksandchunks_ipb=True,
                               use_explicit_eos=True)
        self.ap = None
        self.file_to_trans = dict()
        self.file_to_spec = dict()

    def featurize_corpus(self):
        with open("Corpora/CSS10/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            print(line)
            if line.strip() != "":
                self.file_to_trans[line.split("|")[0]] = self.tf.string_to_tensor(line.split("|")[2]).numpy().tolist()
        for file in self.file_to_trans.keys():
            print(file)
            wave, sr = sf.read(os.path.join("Corpora/CSS10/", file))
            if self.ap is None:
                self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80)
            self.file_to_spec[file] = self.ap.audio_to_mel_spec_tensor(wave).numpy().tolist()
        if not os.path.exists("Corpora/TransformerTTS/SingleSpeaker/CSS10/"):
            os.makedirs("Corpora/TransformerTTS/SingleSpeaker/CSS10/")
        with open(os.path.join("Corpora/TransformerTTS/SingleSpeaker/CSS10/features.json"), 'w') as fp:
            json.dump(self.collect_features(), fp)

    def collect_features(self):
        features = list()
        for file in self.file_to_trans:
            text_tensor = self.file_to_trans[file]
            text_length = len(self.file_to_trans[file])
            speech_tensor = self.file_to_spec[file]
            speech_length = len(self.file_to_spec[file][0])
            if speech_length > 100:
                features.append([text_tensor, text_length, speech_tensor, speech_length])
        return features


def train_loop(net, train_dataset, eval_dataset, device, save_directory, config, epochs=150, batchsize=64):
    start_time = time.time()
    loss_plot = [[], []]
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(config)
    val_loss_highscore = 100.0
    batch_counter = 0
    net = net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        index_list = random.sample(range(len(train_dataset)), len(train_dataset))
        train_losses = list()
        # train one epoch
        texts = list()
        text_lens = list()
        speeches = list()
        speech_lens = list()
        for index in index_list:
            # accumulate batch
            train_datapoint = train_dataset[index]
            texts.append(train_datapoint[0])
            text_lens.append(train_datapoint[1])
            speeches.append(train_datapoint[2])
            speech_lens.append(train_datapoint[3])
            if (index + 1) % batchsize == 0:
                # 0-pad elements in batch
                torch.cuda.empty_cache()
                text_batch_padded = pad_sequence(texts, batch_first=True).to(device)
                speech_batch_padded = pad_sequence(speeches, batch_first=True).to(device)
                # push batch through network
                train_loss = net(text_batch_padded,
                                 torch.cat(text_lens, 0).to(device),
                                 speech_batch_padded,
                                 torch.cat(speech_lens, 0).to(device)
                                 )[0]
                batch_counter += 1
                print("Step:         {}".format(batch_counter * batchsize))
                train_loss.backward()
                optimizer.step()
                # reset for next batch
                optimizer.zero_grad()
                texts = list()
                text_lens = list()
                speeches = list()
                speech_lens = list()
        # evaluate after epoch
        with torch.no_grad():
            net.eval()
            val_losses = list()
            for validation_datapoint_index in range(len(eval_dataset)):
                eval_datapoint = eval_dataset[validation_datapoint_index]
                val_losses.append(net(eval_datapoint[0].unsqueeze(0),
                                      eval_datapoint[1],
                                      eval_datapoint[2].unsqueeze(0),
                                      eval_datapoint[3]
                                      )[0])
            val_loss = sum(val_losses) / len(val_losses)
            if val_loss_highscore > val_loss:
                val_loss_highscore = val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(save_directory, "checkpoint_{}_{}.pt".format(round(float(val_loss), 4),
                                                                                     batch_counter * batchsize)))
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(sum(train_losses) / len(train_losses)))
            print("Valid Loss:   {}".format(val_loss))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            loss_plot[0].append(float(sum(train_losses) / len(train_losses)))
            loss_plot[1].append(float(val_loss))
            with open(os.path.join(save_directory, "train_val_loss.json"), 'w') as fp:
                json.dump(loss_plot, fp)
            net.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def plot_model():
    trans = Transformer(idim=132, odim=80, spk_embed_dim=128)
    out = trans(text=torch.randint(high=120, size=(1, 23)),
                text_lengths=torch.tensor([23]),
                speech=torch.rand((1, 1234, 80)),
                speech_lengths=torch.tensor([1234]),
                spembs=torch.rand(128).unsqueeze(0))
    torchviz.make_dot(out[0].mean(), dict(trans.named_parameters())).render("transformertts_graph", format="png")


if __name__ == '__main__':
    print("Extracting features")
    # fe = CSS10SingleSpeakerFeaturizer()
    # fe.featurize_corpus()
    print("Loading data")
    device = torch.device("cuda")
    distributed = True
    with open("Corpora/TransformerTTS/SingleSpeaker/CSS10/features.json", 'r') as fp:
        feature_list = json.load(fp)
    print("Building datasets")
    css10_train = TransformerTTSDataset(feature_list, type="train")
    css10_valid = TransformerTTSDataset(feature_list, type="valid")
    model = Transformer(idim=132, odim=80, spk_embed_dim=None)
    if not os.path.exists("Models/TransformerTTS/SingleSpeaker/CSS10"):
        os.makedirs("Models/TransformerTTS/SingleSpeaker/CSS10")
    print("Training model")
    if not distributed:
        train_loop(net=model,
                   train_dataset=css10_train,
                   eval_dataset=css10_valid,
                   device=device,
                   config=model.get_conf(),
                   save_directory="Models/TransformerTTS/SingleSpeaker/CSS10",
                   batchsize=5)
    else:
        train_loop(net=torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]),
                   train_dataset=css10_train,
                   eval_dataset=css10_valid,
                   device=device,
                   config=model.get_conf(),
                   save_directory="Models/TransformerTTS/SingleSpeaker/CSS10",
                   batchsize=42)
