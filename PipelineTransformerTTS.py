import json
import os

import soundfile as sf
import torch
import torchviz

import SpeakerEmbedding.SiameseSpeakerEmbedding as SiamSpemb
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerTTS import Transformer


class MultiSpeakerFeaturizer():
    def __init__(self):
        self.tf = TextFrontend(language="en",
                               use_panphon_vectors=False,
                               use_shallow_pos=False,
                               use_sentence_type=False,
                               use_positional_information=False,
                               use_word_boundaries=False,
                               use_chinksandchunks_ipb=True,
                               use_explicit_eos=True)
        self.ap = AudioPreprocessor(input_sr=999999, output_sr=16000, melspec_buckets=80)
        self.spemb_ext = SiamSpemb.build_spk_emb_model()

    def featurize_corpus(self, path_to_corpus):
        # load pair of text and speech
        # apply collect_features()
        # store features in Dataset dict
        # repeat for all pairs
        # Dump dict to file
        pass

    def collect_features(self, text, wave):
        text_tensor = self.tf.string_to_tensor(text).numpy().tolist()
        text_length = len(text_tensor)
        speech_tensor = self.ap.audio_to_mel_spec_tensor(wave).numpy().tolist()
        speech_length = len(speech_tensor[0])
        speaker_embedding = self.spemb_ext.inference(self.spemb_melspec.audio_to_mel_spec_tensor(wave)).numpy().tolist()
        return text_tensor, text_length, speech_tensor, speech_length, speaker_embedding


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
        with open("Corpora/CSS10/transcript.txt") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
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


def train_loop(net, train_dataset, eval_dataset, epochs, batchsize):
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def plot_model():
    trans = Transformer(idim=131, odim=80, spk_embed_dim=128)
    out = trans(text=torch.randint(high=120, size=(1, 23)),
                text_lengths=torch.tensor([23]),
                speech=torch.rand((1, 1234, 80)),
                speech_lengths=torch.tensor([1234]),
                spembs=torch.rand(128).unsqueeze(0))
    torchviz.make_dot(out[0].mean(), dict(trans.named_parameters())).render("transformertts_graph", format="png")


if __name__ == '__main__':
    fe = CSS10SingleSpeakerFeaturizer()
    fe.featurize_corpus()
