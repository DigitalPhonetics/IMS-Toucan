import os
import random
import warnings

import soundfile as sf
import torch
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
                 loading_processes,
                 cut_silences,
                 do_loudnorm,
                 device,
                 min_len_in_seconds=1,
                 max_len_in_seconds=15,
                 rebuild_cache=False,
                 verbose=False,
                 phone_input=False,
                 allow_unknown_symbols=False):
        self.tf = ArticulatoryCombinedTextFrontend(language=lang)
        if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")) or rebuild_cache:
            os.makedirs(cache_dir, exist_ok=True)
            _ = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               run_opts={"device": str(device)},
                                               savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
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
            fisher_yates_shuffle(key_list)
            # build cache
            print("... building dataset cache ...")
            self.datapoints = resource_manager.list()
            self.speaker_embeddings = resource_manager.list()

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
                                  min_len_in_seconds,
                                  max_len_in_seconds,
                                  cut_silences,
                                  do_loudnorm,
                                  verbose,
                                  device,
                                  phone_input,
                                  allow_unknown_symbols),
                            daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            # we had to turn all the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.

            # save to cache
            if len(self.datapoints) == 0:
                raise RuntimeError

            self.datapoint_feature_dump_list = list()
            os.makedirs(os.path.join(cache_dir, f"aligner_datapoints/"), exist_ok=True)
            for index, (datapoint, speaker_embedding) in tqdm(enumerate(zip(self.datapoints, self.speaker_embeddings))):
                torch.save(([datapoint[0],
                             datapoint[1],
                             datapoint[2],
                             datapoint[3]],
                            speaker_embedding,
                            datapoint[-1]),
                           os.path.join(cache_dir, f"aligner_datapoints/aligner_datapoint_{index}.pt"))
                self.datapoint_feature_dump_list.append(os.path.join(cache_dir, f"aligner_datapoints/aligner_datapoint_{index}.pt"))

            torch.save(self.datapoint_feature_dump_list,
                       os.path.join(cache_dir, "aligner_train_cache.pt"))
        else:
            # just load the datapoints from cache
            self.datapoint_feature_dump_list = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')

        print(f"Prepared an Aligner dataset with {len(self.datapoint_feature_dump_list)} datapoints in {cache_dir}.")

    def cache_builder_process(self,
                              path_list,
                              min_len,
                              max_len,
                              cut_silences,
                              do_loudnorm,
                              verbose,
                              device,
                              phone_input,
                              allow_unknown_symbols):
        speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                run_opts={"device": str(device)},
                                                                savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
        process_internal_dataset_chunk = list()
        _, sr = sf.read(path_list[0])
        assumed_sr = sr
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=cut_silences, do_loudnorm=do_loudnorm, device=device)

        for path in tqdm(path_list):
            if self.path_to_transcript_dict[path].strip() == "":
                continue
            try:
                wave, sr = sf.read(path)
                if sr != assumed_sr:
                    print(f"{path} has an unexpected samplingrate: {sr} vs. {assumed_sr} --> skipping")
                    continue
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
                    norm_wave = ap.normalize_audio(audio=wave)
            except ValueError:
                continue
            dur_in_seconds = len(norm_wave) / 16000
            if not (min_len <= dur_in_seconds <= max_len):  # duration may have changed because of the VAD
                if verbose:
                    print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                continue
            # raw audio preprocessing is done
            transcript = self.path_to_transcript_dict[path]
            try:
                try:
                    cached_text = self.tf.string_to_tensor(transcript, handle_missing=False, input_phonemes=phone_input).squeeze(0).cpu().numpy()
                except KeyError:
                    cached_text = self.tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=phone_input).squeeze(0).cpu().numpy()
                    if not allow_unknown_symbols:
                        continue  # we skip sentences with unknown symbols
            except ValueError:
                # this can happen for Mandarin Chinese, when the syllabification of pinyin doesn't work. In that case, we just skip the sample.
                continue
            except KeyError:
                # this can happen for Mandarin Chinese, when the syllabification of pinyin doesn't work. In that case, we just skip the sample.
                continue
            cached_text_len = torch.LongTensor([len(cached_text)])
            cached_speech = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False,
                                                        explicit_sampling_rate=16000).transpose(0, 1).cpu().numpy()
            with torch.inference_mode():
                self.speaker_embeddings.append(speaker_embedding_func.encode_batch(wavs=norm_wave.unsqueeze(0)).squeeze().cpu().numpy())
            cached_speech_len = torch.LongTensor([len(cached_speech)])
            process_internal_dataset_chunk.append([cached_text,
                                                   cached_text_len,
                                                   cached_speech,
                                                   cached_speech_len,
                                                   None,
                                                   path])
        self.datapoints += process_internal_dataset_chunk

    def __getitem__(self, index):
        path_to_datapoint_file = self.datapoint_feature_dump_list[index]
        datapoint, speaker_embedding, filepath = torch.load(path_to_datapoint_file, map_location='cpu')

        text_vector = torch.Tensor(datapoint[0])
        tokens = self.tf.text_vectors_to_id_sequence(text_vector=text_vector)
        tokens = torch.LongTensor(tokens)
        return tokens, \
            torch.LongTensor([len(tokens)]), \
            torch.Tensor(datapoint[2]), \
            torch.LongTensor(datapoint[3]), \
            torch.Tensor(speaker_embedding)

    def __len__(self):
        return len(self.datapoint_feature_dump_list)


def fisher_yates_shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
