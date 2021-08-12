import gc
import json
import os

import soundfile as sf
import torch
import torchaudio
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.ArticulatoryTextFrontend import ArticulatoryTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio


class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_model,
                 cache_dir,
                 speaker_embedding=False,
                 loading_processes=5,
                 lang="en",
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 path_blacklist=None  # because for some datasets, some of the alignments
                 # simply fail because attention heads do weird things. Those need to be
                 # found in the duration_vis folder and manually added to a list of samples
                 # to be excluded from the dataset.
                 ):
        self.speaker_embedding = speaker_embedding
        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.json")) or rebuild_cache:
            if not os.path.isdir(os.path.join(cache_dir, "durations_visualization")):
                os.makedirs(os.path.join(cache_dir, "durations_visualization"))
            resource_manager = Manager()
            self.path_to_transcript_dict = path_to_transcript_dict
            key_list = list(self.path_to_transcript_dict.keys())
            # build cache
            print("... building dataset cache ...")
            self.datapoints = resource_manager.list()
            # make processes
            key_splits = list()
            process_list = list()
            for i in range(loading_processes):
                key_splits.append(key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])
            gc.disable()
            for key_split in key_splits:
                process_list.append(Process(target=self.cache_builder_process, args=(key_split,
                                                                                     acoustic_model,
                                                                                     speaker_embedding,
                                                                                     lang,
                                                                                     min_len_in_seconds,
                                                                                     max_len_in_seconds,
                                                                                     reduction_factor,
                                                                                     device,
                                                                                     cache_dir,
                                                                                     cut_silence), daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            gc.enable()
            self.datapoints = list(self.datapoints)
            # save to json so we can rebuild cache quickly
            with open(os.path.join(cache_dir, "fast_train_cache.json"), 'w') as fp:
                json.dump(self.datapoints, fp)
        else:
            # just load the datapoints
            with open(os.path.join(cache_dir, "fast_train_cache.json"), 'r') as fp:
                self.datapoints = json.load(fp)

        if path_blacklist is not None:
            bl = set(path_blacklist)
            datapoints_with_durations_that_make_sense = list()
            for el in self.datapoints:
                if el[-1] not in bl:
                    datapoints_with_durations_that_make_sense.append(el)
            self.datapoints = datapoints_with_durations_that_make_sense
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def cache_builder_process(self,
                              path_list,
                              acoustic_model,
                              speaker_embedding,
                              lang,
                              min_len,
                              max_len,
                              reduction_factor,
                              device,
                              cache_dir,
                              cut_silence):
        process_internal_dataset_chunk = list()
        tf = ArticulatoryTextFrontend(language=lang)
        _, sr = sf.read(path_list[0])
        if speaker_embedding:
            wav2mel = torch.jit.load("Models/SpeakerEmbedding/wav2mel.pt")
            dvector = torch.jit.load("Models/SpeakerEmbedding/dvector-step250000.pt").eval()
        ap = AudioPreprocessor(input_sr=sr,
                               output_sr=16000,
                               melspec_buckets=80,
                               hop_length=256,
                               n_fft=1024,
                               cut_silence=cut_silence)
        acoustic_model = acoustic_model.to(device)
        dc = DurationCalculator(reduction_factor=reduction_factor)
        dio = Dio(reduction_factor=reduction_factor)
        energy_calc = EnergyCalculator(reduction_factor=reduction_factor)
        for path in tqdm(path_list):
            transcript = self.path_to_transcript_dict[path]
            with open(path, "rb") as audio_file:
                wave, sr = sf.read(audio_file)
            if min_len <= len(wave) / sr <= max_len:
                norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
                norm_wave_length = torch.LongTensor([len(norm_wave)])
                melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
                melspec_length = torch.LongTensor([len(melspec)])
                text = tf.string_to_tensor(transcript).long()
                cached_text = tf.string_to_tensor(transcript).squeeze(0).numpy().tolist()
                cached_text_len = len(cached_text)
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).numpy().tolist()
                cached_speech_len = len(cached_speech)
                if not speaker_embedding:
                    os.path.join(cache_dir, "durations_visualization")
                    cached_duration = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                                  speech=melspec.to(device),
                                                                  use_teacher_forcing=True,
                                                                  speaker_embeddings=None)[2],
                                         vis=os.path.join(cache_dir, "durations_visualization", path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
                else:
                    wav_tensor, sample_rate = torchaudio.load(path)
                    mel_tensor = wav2mel(wav_tensor, sample_rate)
                    cached_speaker_embedding = dvector.embed_utterance(mel_tensor)
                    cached_duration = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                                  speech=melspec.to(device),
                                                                  use_teacher_forcing=True,
                                                                  speaker_embeddings=cached_speaker_embedding.to(device))[2],
                                         vis=os.path.join(cache_dir, "durations_visualization", path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
                cached_energy = energy_calc(input=norm_wave.unsqueeze(0),
                                            input_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            durations=cached_duration.unsqueeze(0),
                                            durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0)
                cached_pitch = dio(input=norm_wave.unsqueeze(0),
                                   input_lengths=norm_wave_length,
                                   feats_lengths=melspec_length,
                                   durations=cached_duration.unsqueeze(0),
                                   durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0)
                if not self.speaker_embedding:
                    process_internal_dataset_chunk.append([cached_text,
                                                           cached_text_len,
                                                           cached_speech,
                                                           cached_speech_len,
                                                           cached_duration.numpy().tolist(),
                                                           cached_energy.numpy().tolist(),
                                                           cached_pitch.numpy().tolist(),
                                                           path])
                else:
                    process_internal_dataset_chunk.append([cached_text,
                                                           cached_text_len,
                                                           cached_speech,
                                                           cached_speech_len,
                                                           cached_duration.numpy().tolist(),
                                                           cached_energy.numpy().tolist(),
                                                           cached_pitch.numpy().tolist(),
                                                           cached_speaker_embedding.detach().numpy().tolist(),
                                                           path])
        self.datapoints += process_internal_dataset_chunk

    def __getitem__(self, index):
        if not self.speaker_embedding:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3], \
                   self.datapoints[index][4], \
                   self.datapoints[index][5], \
                   self.datapoints[index][6]
        else:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3], \
                   self.datapoints[index][4], \
                   self.datapoints[index][5], \
                   self.datapoints[index][6], \
                   self.datapoints[index][7]

    def __len__(self):
        return len(self.datapoints)
