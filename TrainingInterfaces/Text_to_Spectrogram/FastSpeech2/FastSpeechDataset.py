import os

import soundfile as sf
import torch
import torchaudio
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio


class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_model,
                 cache_dir,
                 lang,
                 speaker_embedding=False,
                 loading_processes=5,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False):
        self.speaker_embedding = speaker_embedding
        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")) or rebuild_cache:
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
            self.datapoints = list(self.datapoints)
            tensored_datapoints = list()
            # we had to turn all of the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.
            print("Converting into convenient format...")
            if self.speaker_embedding:
                for datapoint in tqdm(self.datapoints):
                    tensored_datapoints.append([torch.Tensor(datapoint[0]),
                                                torch.LongTensor(datapoint[1]),
                                                torch.Tensor(datapoint[2]),
                                                torch.LongTensor(datapoint[3]),
                                                torch.Tensor(datapoint[4]),
                                                torch.Tensor(datapoint[5]),
                                                torch.Tensor(datapoint[6]),
                                                torch.Tensor(datapoint[7])])
            else:
                for datapoint in tqdm(self.datapoints):
                    tensored_datapoints.append([torch.Tensor(datapoint[0]),
                                                torch.LongTensor(datapoint[1]),
                                                torch.Tensor(datapoint[2]),
                                                torch.LongTensor(datapoint[3]),
                                                torch.Tensor(datapoint[4]),
                                                torch.Tensor(datapoint[5]),
                                                torch.Tensor(datapoint[6])])
            self.datapoints = tensored_datapoints
            # save to cache
            torch.save(self.datapoints, os.path.join(cache_dir, "fast_train_cache.pt"))
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "fast_train_cache.pt"), map_location='cpu')

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
        tf = ArticulatoryCombinedTextFrontend(language=lang)
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
            wave, sr = sf.read(path)
            if min_len <= len(wave) / sr <= max_len:
                norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
                norm_wave_length = torch.LongTensor([len(norm_wave)])
                melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
                melspec_length = torch.LongTensor([len(melspec)])
                text = tf.string_to_tensor(transcript)
                cached_text = tf.string_to_tensor(transcript).squeeze(0).cpu()
                cached_text_len = torch.LongTensor([len(cached_text)])
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).cpu()
                cached_speech_len = torch.LongTensor([len(cached_speech)])
                if not speaker_embedding:
                    os.path.join(cache_dir, "durations_visualization")
                    attention_map = acoustic_model.inference(text_tensor=text.squeeze(0).to(device),
                                                             speech_tensor=melspec.to(device),
                                                             use_teacher_forcing=True,
                                                             speaker_embeddings=None)[2]
                    focus_rate = self._calculate_focus_rate(attention_map.unsqueeze(1))
                    print(focus_rate)
                    cached_duration = dc(attention_map,
                                         vis=os.path.join(cache_dir, "durations_visualization",
                                                          str(int(focus_rate * 1000)) + "_" + path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
                else:
                    wav_tensor, sample_rate = torchaudio.load(path)
                    mel_tensor = wav2mel(wav_tensor, sample_rate)
                    cached_speaker_embedding = dvector.embed_utterance(mel_tensor)
                    attention_map = acoustic_model.inference(text_tensor=text.squeeze(0).to(device),
                                                             speech_tensor=melspec.to(device),
                                                             use_teacher_forcing=True,
                                                             speaker_embeddings=cached_speaker_embedding.to(device))[2]
                    focus_rate = self._calculate_focus_rate(attention_map.unsqueeze(1))
                    cached_duration = dc(attention_map,
                                         vis=os.path.join(cache_dir, "durations_visualization",
                                                          str(int(focus_rate * 1000)) + "_" + path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
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
                    process_internal_dataset_chunk.append([cached_text.numpy(),
                                                           cached_text_len.numpy(),
                                                           cached_speech.numpy(),
                                                           cached_speech_len.numpy(),
                                                           cached_duration.cpu().numpy(),
                                                           cached_energy.cpu().numpy(),
                                                           cached_pitch.cpu().numpy()])
                else:
                    process_internal_dataset_chunk.append([cached_text.numpy(),
                                                           cached_text_len.numpy(),
                                                           cached_speech.numpy(),
                                                           cached_speech_len.numpy(),
                                                           cached_duration.cpu().numpy(),
                                                           cached_energy.cpu().numpy(),
                                                           cached_pitch.cpu().numpy(),
                                                           cached_speaker_embedding.detach().cpu().numpy()])
        self.datapoints += process_internal_dataset_chunk

    @staticmethod
    def _calculate_focus_rate(att_ws):
        # transformer case -> (#layers, #heads, L, T)
        return att_ws.max(dim=-1)[0].mean(dim=-1).max()

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
