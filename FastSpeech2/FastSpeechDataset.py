import os

import soundfile as sf
import torch
from torch.utils.data import Dataset

from FastSpeech2.DurationCalculator import DurationCalculator
from FastSpeech2.EnergyCalculator import EnergyCalculator
from FastSpeech2.PitchCalculator import Dio
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerTTS import build_reference_transformer_tts_model


class FastSpeechDataset(Dataset):

    def __init__(self, path_to_transcript_dict, acoustic_model_name, device=torch.device("cpu"), spemb=False,
                 train=True):
        self.path_to_transcript_dict = path_to_transcript_dict
        if train:
            key_list = list(self.path_to_transcript_dict.keys())[:70]
        else:
            key_list = list(self.path_to_transcript_dict.keys())[-10:]
        self.spemb = spemb
        self.device = device
        tf = TextFrontend(language="de",
                          use_panphon_vectors=False,
                          use_shallow_pos=False,
                          use_sentence_type=False,
                          use_positional_information=False,
                          use_word_boundaries=False,
                          use_chinksandchunks_ipb=False,
                          use_explicit_eos=True)
        ap = None
        acoustic_model = build_reference_transformer_tts_model(model_name=acoustic_model_name)
        dc = DurationCalculator()
        dio = Dio()
        energy_calc = EnergyCalculator()
        # build cache
        print("... building dataset cache ...")
        self.cached_text = list()
        self.cached_text_lens = list()
        self.cached_speech = list()
        self.cached_speech_lens = list()
        self.cached_energy = list()
        self.cached_pitch = list()
        self.cached_durations = list()
        for path in key_list:
            transcript = self.path_to_transcript_dict[path]
            wave, sr = sf.read(os.path.join("Corpora/CSS10/", path))
            if 50000 < len(wave) < 230000:
                print("processing {}".format(path))
                if ap is None:
                    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
                norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
                norm_wave_length = torch.LongTensor([len(norm_wave)])
                melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
                melspec_length = torch.LongTensor([len(melspec)])
                text = tf.string_to_tensor(transcript).long()
                if self.spemb:
                    print("not implemented yet")
                    raise NotImplementedError
                else:
                    self.cached_durations.append(dc(acoustic_model.inference(text=text,
                                                                             speech=melspec,
                                                                             use_teacher_forcing=True,
                                                                             spembs=None)[2])[0])
                duration_length = torch.LongTensor([len(self.cached_durations[-1])])
                self.cached_text.append(text)
                self.cached_text_lens.append(torch.LongTensor([len(self.cached_text[-1])]))
                self.cached_speech.append(melspec)
                self.cached_speech_lens.append(torch.LongTensor([len(self.cached_speech[-1])]))
                self.cached_energy.append(energy_calc(input=norm_wave.unsqueeze(0),
                                                      input_lengths=norm_wave_length,
                                                      feats_lengths=melspec_length,
                                                      durations=self.cached_durations[-1].unsqueeze(0),
                                                      durations_lengths=duration_length)[0].squeeze(0))
                self.cached_pitch.append(dio(input=norm_wave.unsqueeze(0),
                                             input_lengths=norm_wave_length,
                                             feats_lengths=melspec_length,
                                             durations=self.cached_durations[-1].unsqueeze(0),
                                             durations_lengths=duration_length)[0].squeeze(0))

    def __getitem__(self, index):
        return self.cached_text[index], \
               self.cached_text_lens[index], \
               self.cached_speech[index], \
               self.cached_speech_lens[index], \
               self.cached_durations[index], \
               self.cached_energy[index], \
               self.cached_pitch[index]

    def __len__(self):
        return len(self.cached_text_lens)
