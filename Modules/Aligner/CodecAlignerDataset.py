import os
import random

import librosa
import soundfile as sf
import torch
from speechbrain.pretrained import EncoderClassifier
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from tqdm import tqdm

from Preprocessing.EnCodecAudioPreprocessor import CodecAudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Utility.storage_config import MODELS_DIR


class CodecAlignerDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 cache_dir,
                 lang,
                 loading_processes,
                 device,
                 min_len_in_seconds=1,
                 max_len_in_seconds=15,
                 rebuild_cache=False,
                 verbose=False,
                 phone_input=False,
                 allow_unknown_symbols=False,
                 gpu_count=1,
                 rank=0):

        self.gpu_count = gpu_count
        self.rank = rank
        if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")) or rebuild_cache:
            self._build_dataset_cache(path_to_transcript_dict=path_to_transcript_dict,
                                      cache_dir=cache_dir,
                                      lang=lang,
                                      loading_processes=loading_processes,
                                      device=device,
                                      min_len_in_seconds=min_len_in_seconds,
                                      max_len_in_seconds=max_len_in_seconds,
                                      verbose=verbose,
                                      phone_input=phone_input,
                                      allow_unknown_symbols=allow_unknown_symbols,
                                      gpu_count=gpu_count,
                                      rank=rank)
        self.lang = lang
        self.device = device
        self.cache_dir = cache_dir
        self.tf = ArticulatoryCombinedTextFrontend(language=self.lang, device=device)
        cache = torch.load(os.path.join(self.cache_dir, "aligner_train_cache.pt"), map_location='cpu')
        self.speaker_embeddings = cache[2]
        self.filepaths = cache[3]
        self.datapoints = cache[0]
        if self.gpu_count > 1:
            # we only keep a chunk of the dataset in memory to avoid redundancy. Which chunk, we figure out using the rank.
            while len(self.datapoints) % self.gpu_count != 0:
                self.datapoints.pop(-1)  # a bit unfortunate, but if you're using multiple GPUs, you probably have a ton of datapoints anyway.
            chunksize = int(len(self.datapoints) / self.gpu_count)
            self.datapoints = self.datapoints[chunksize * self.rank:chunksize * (self.rank + 1)]
            self.speaker_embeddings = self.speaker_embeddings[chunksize * self.rank:chunksize * (self.rank + 1)]
        print(f"Loaded an Aligner dataset with {len(self.datapoints)} datapoints from {cache_dir}.")

    def _build_dataset_cache(self,
                             path_to_transcript_dict,
                             cache_dir,
                             lang,
                             loading_processes,
                             device,
                             min_len_in_seconds=1,
                             max_len_in_seconds=15,
                             verbose=False,
                             phone_input=False,
                             allow_unknown_symbols=False,
                             gpu_count=1,
                             rank=0
                             ):
        if gpu_count != 1:
            import sys
            print("Please run the feature extraction using only a single GPU. Multi-GPU is only supported for training.")
            sys.exit()
        os.makedirs(cache_dir, exist_ok=True)
        if type(path_to_transcript_dict) != dict:
            path_to_transcript_dict = path_to_transcript_dict()  # in this case we passed a function instead of the dict, so that the function isn't executed if not necessary.
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
        resource_manager = Manager()
        self.path_to_transcript_dict = resource_manager.dict(path_to_transcript_dict)
        key_list = list(self.path_to_transcript_dict.keys())
        with open(os.path.join(cache_dir, "files_used.txt"), encoding='utf8', mode="w") as files_used_note:
            files_used_note.write(str(key_list))
        fisher_yates_shuffle(key_list)
        # build cache
        print("... building dataset cache ...")
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
        # careful: assumes 16kHz or 8kHz audio
        _, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',  # make sure it gets downloaded during single-processing first, if it's not already downloaded
                              model='silero_vad',
                              force_reload=False,
                              onnx=False,
                              verbose=False)
        self.result_pool = resource_manager.list()
        # make processes
        key_splits = list()
        process_list = list()
        for i in range(loading_processes):
            key_splits.append(
                key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])
        for key_split in key_splits:
            process_list.append(
                Process(target=self._cache_builder_process,
                        args=(key_split,
                              lang,
                              min_len_in_seconds,
                              max_len_in_seconds,
                              verbose,
                              device,
                              phone_input,
                              allow_unknown_symbols),
                        daemon=True))
            process_list[-1].start()
        for process in process_list:
            process.join()
        print("pooling results...")
        pooled_datapoints = list()
        for chunk in self.result_pool:
            for datapoint in chunk:
                pooled_datapoints.append(datapoint)  # unpack into a joint list
        self.result_pool = pooled_datapoints
        del pooled_datapoints
        print("converting text to tensors...")
        text_tensors = [torch.ShortTensor(x[0]) for x in self.result_pool]  # turn everything back to tensors (had to turn it to np arrays to avoid multiprocessing issues)
        print("converting speech to tensors...")
        speech_tensors = [torch.ShortTensor(x[1]) for x in self.result_pool]
        print("converting waves to tensors...")
        norm_waves = [torch.Tensor(x[2]) for x in self.result_pool]
        print("unpacking file list...")
        filepaths = [x[3] for x in self.result_pool]
        del self.result_pool
        self.datapoints = list(zip(text_tensors, speech_tensors))
        del text_tensors
        del speech_tensors
        print("done!")

        # add speaker embeddings
        self.speaker_embeddings = list()
        speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                      run_opts={"device": str(device)},
                                                                      savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
        with torch.inference_mode():
            for wave in tqdm(norm_waves):
                self.speaker_embeddings.append(speaker_embedding_func_ecapa.encode_batch(wavs=wave.to(device).unsqueeze(0)).squeeze().cpu())

        # save to cache
        if len(self.datapoints) == 0:
            raise RuntimeError  # something went wrong and there are no datapoints
        torch.save((self.datapoints, None, self.speaker_embeddings, filepaths),
                   os.path.join(cache_dir, "aligner_train_cache.pt"))

    def _cache_builder_process(self,
                               path_list,
                               lang,
                               min_len,
                               max_len,
                               verbose,
                               device,
                               phone_input,
                               allow_unknown_symbols):
        torch.use_deterministic_algorithms(False)
        process_internal_dataset_chunk = list()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
        # careful: assumes 16kHz or 8kHz audio
        silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False,
                                             onnx=False,
                                             verbose=False)
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
        # this to false globally during model loading rather than using inference mode or no_grad
        
        
        silero_model = silero_model.to(device)
        silence = torch.zeros([16000 // 8]).to(device)
        tf = ArticulatoryCombinedTextFrontend(language=lang, device=device)
        _, sr = sf.read(path_list[0])
        assumed_sr = sr
        ap = CodecAudioPreprocessor(input_sr=assumed_sr, device=device)
        resample = Resample(orig_freq=assumed_sr, new_freq=16000).to(device)

        for path in tqdm(path_list):
            if self.path_to_transcript_dict[path].strip() == "":
                continue
            try:
                wave, sr = sf.read(path)
            except:
                print(f"Problem with an audio file: {path}")
                continue

            if len(wave.shape) > 1:  # oh no, we found a stereo audio!
                if len(wave[0]) == 2:  # let's figure out whether we need to switch the axes
                    wave = wave.transpose()  # if yes, we switch the axes.
            wave = librosa.to_mono(wave)
            if sr != assumed_sr:
                assumed_sr = sr
                ap = CodecAudioPreprocessor(input_sr=assumed_sr, device=device)
                resample = Resample(orig_freq=assumed_sr, new_freq=16000).to(device)
                print(f"{path} has a different sampling rate --> adapting the codec processor")
            try:
                norm_wave = resample(torch.tensor(wave).float().to(device))
            except ValueError:
                continue
            dur_in_seconds = len(norm_wave) / 16000
            if not (min_len <= dur_in_seconds <= max_len):
                if verbose:
                    print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                continue
            with torch.inference_mode():
                
                speech_timestamps = get_speech_timestamps(norm_wave, silero_model, sampling_rate=16000)
            try:
                silence_timestamps = invert_segments(speech_timestamps, len(norm_wave))
                for silence_timestamp in silence_timestamps:
                    begin = silence_timestamp['start']
                    end = silence_timestamp['end']
                    norm_wave = torch.cat([norm_wave[:begin], torch.zeros([end - begin], device=device), norm_wave[end:]])
                result = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]
            except IndexError:
                print("Audio might be too short to cut silences from front and back.")
                continue
            norm_wave = torch.cat([silence, result, silence])

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
            
            cached_speech = ap.audio_to_codebook_indexes(audio=norm_wave, current_sampling_rate=16000).transpose(0, 1).cpu().numpy()
            process_internal_dataset_chunk.append([cached_text,
                                                   cached_speech,
                                                   norm_wave.cpu().detach().numpy(),
                                                   path])
        self.result_pool.append(process_internal_dataset_chunk)

    def __getitem__(self, index):
        text_vector = self.datapoints[index][0]
        tokens = self.tf.text_vectors_to_id_sequence(text_vector=text_vector)
        tokens = torch.LongTensor(tokens)
        token_len = torch.LongTensor([len(tokens)])

        codes = self.datapoints[index][1]
        if codes.size()[0] != 24:  # no clue why this is sometimes the case
            codes = codes.transpose(0, 1)

        return tokens, \
            token_len, \
            codes, \
            None, \
            self.speaker_embeddings[index]

    def __len__(self):
        return len(self.datapoints)

    def remove_samples(self, list_of_samples_to_remove):
        for remove_id in sorted(list_of_samples_to_remove, reverse=True):
            self.datapoints.pop(remove_id)
            self.speaker_embeddings.pop(remove_id)
            self.filepaths.pop(remove_id)
        torch.save((self.datapoints, None, self.speaker_embeddings, self.filepaths),
                   os.path.join(self.cache_dir, "aligner_train_cache.pt"))
        print("Dataset updated!")


def fisher_yates_shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]


def invert_segments(segments, total_duration):
    if not segments:
        return [{'start': 0, 'end': total_duration}]

    inverted_segments = []
    previous_end = 0

    for segment in segments:
        start = segment['start']
        if previous_end < start:
            inverted_segments.append({'start': previous_end, 'end': start})
        previous_end = segment['end']

    if previous_end < total_duration:
        inverted_segments.append({'start': previous_end, 'end': total_duration})

    return inverted_segments