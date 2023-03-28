import os
import random
import warnings

import soundfile as sf
import torch
from numpy import trim_zeros
from speechbrain.pretrained import EncoderClassifier
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Utility.storage_config import MODELS_DIR


class AlignerDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 cache_dir,
                 lang,
                 loading_processes=os.cpu_count() if os.cpu_count() is not None else 30,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silences=True,
                 rebuild_cache=False,
                 verbose=False,
                 device="cpu",
                 phone_input=False,
                 allow_unknown_symbols=False):
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")) or rebuild_cache:
            if cut_silences:
                torch.set_num_threads(1)
                torch.hub.load(repo_or_dir='snakers4/silero-vad',
                               model='silero_vad',
                               force_reload=False,
                               onnx=False,
                               verbose=False)  # download and cache for it to be loaded and used later
                torch.set_grad_enabled(True)
            resource_manager = Manager()
            self.path_to_transcript_dict = resource_manager.dict(path_to_transcript_dict)
            key_list = list(self.path_to_transcript_dict.keys())
            with open(os.path.join(cache_dir, "files_used.txt"), encoding='utf8', mode="w") as files_used_note:
                files_used_note.write(str(key_list))
            random.shuffle(key_list)
            # build cache
            print("... building dataset cache ...")
            self.datapoints = resource_manager.list()
            # make processes
            key_splits = list()
            process_list = list()
            for i in range(loading_processes):
                key_splits.append(
                    key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])
            for key_split in key_splits:
                process_list.append(
                    Process(target=self.cache_builder_process,
                            args=(key_split,
                                  lang,
                                  min_len_in_seconds,
                                  max_len_in_seconds,
                                  cut_silences,
                                  verbose,
                                  "cpu",
                                  phone_input,
                                  allow_unknown_symbols),
                            daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            tensored_datapoints = list()
            # we had to turn all of the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.
            print("Converting into convenient format...")
            norm_waves = list()
            filepaths = list()
            for datapoint in tqdm(self.datapoints):
                tensored_datapoints.append([torch.Tensor(datapoint[0]),
                                            torch.LongTensor(datapoint[1]),
                                            torch.Tensor(datapoint[2]),
                                            torch.LongTensor(datapoint[3])])
                norm_waves.append(torch.Tensor(datapoint[-2]))
                filepaths.append(datapoint[-1])

            self.datapoints = tensored_datapoints

            # add speaker embeddings
            self.speaker_embeddings = list()
            speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                          run_opts={"device": str(device)},
                                                                          savedir=os.path.join(
                                                                              MODELS_DIR,
                                                                              "Embedding",
                                                                              "speechbrain_speaker_embedding_ecapa"))
            with torch.no_grad():
                for wave in tqdm(norm_waves):
                    self.speaker_embeddings.append(
                        speaker_embedding_func_ecapa.encode_batch(wavs=wave.to(device).unsqueeze(0)).squeeze().cpu())

            # save to cache
            if len(self.datapoints) == 0:
                raise RuntimeError
            torch.save((self.datapoints, norm_waves, self.speaker_embeddings, filepaths),
                       os.path.join(cache_dir, "aligner_train_cache.pt"))
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')
            self.speaker_embeddings = self.datapoints[2]
            self.datapoints = self.datapoints[0]

        self.tf = ArticulatoryCombinedTextFrontend(language=lang)
        print(f"Prepared an Aligner dataset with {len(self.datapoints)} datapoints in {cache_dir}.")

    def cache_builder_process(self,
                              path_list,
                              lang,
                              min_len,
                              max_len,
                              cut_silences,
                              verbose,
                              device,
                              phone_input,
                              allow_unknown_symbols):
        process_internal_dataset_chunk = list()
        tf = ArticulatoryCombinedTextFrontend(language=lang)
        _, sr = sf.read(path_list[0])
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024,
                               cut_silence=cut_silences, device=device)

        for path in tqdm(path_list):
            if self.path_to_transcript_dict[path].strip() == "":
                continue

            try:
                wave, sr = sf.read(path)
            except:
                print(f"Problem with an audio file: {path}")
                continue

            dur_in_seconds = len(wave) / sr
            if not (min_len <= dur_in_seconds <= max_len):
                if verbose:
                    print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # otherwise we get tons of warnings about an RNN not being in contiguous chunks
                    norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
            except ValueError:
                continue
            dur_in_seconds = len(norm_wave) / 16000
            if not (min_len <= dur_in_seconds <= max_len):
                if verbose:
                    print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                continue
            norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
            # raw audio preprocessing is done
            transcript = self.path_to_transcript_dict[path]

            try:
                try:
                    cached_text = tf.string_to_tensor(transcript, handle_missing=False, input_phonemes=phone_input).squeeze(0).cpu().numpy()
                except KeyError:
                    cached_text = tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=phone_input).squeeze(0).cpu().numpy()
                    if not allow_unknown_symbols:
                        continue  # we skip sentences with unknown symbols
            except ValueError:
                # this can happen for Mandarin Chinese, when the syllabification of pinyin doesn't work. In that case, we just skip the sample.
                continue
            except KeyError:
                # this can happen for Mandarin Chinese, when the syllabification of pinyin doesn't work. In that case, we just skip the sample.
                continue

            cached_text_len = torch.LongTensor([len(cached_text)]).numpy()
            cached_speech = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False,
                                                        explicit_sampling_rate=16000).transpose(0, 1).cpu().numpy()
            cached_speech_len = torch.LongTensor([len(cached_speech)]).numpy()
            process_internal_dataset_chunk.append([cached_text,
                                                   cached_text_len,
                                                   cached_speech,
                                                   cached_speech_len,
                                                   norm_wave.cpu().detach().numpy(),
                                                   path])
        self.datapoints += process_internal_dataset_chunk

    def __getitem__(self, index):
        text_vector = self.datapoints[index][0]
        tokens = self.tf.text_vectors_to_id_sequence(text_vector=text_vector)
        tokens = torch.LongTensor(tokens)
        return tokens, \
               torch.LongTensor([len(tokens)]), \
               self.datapoints[index][2], \
               self.datapoints[index][3], \
               self.speaker_embeddings[index]

    def __len__(self):
        return len(self.datapoints)
