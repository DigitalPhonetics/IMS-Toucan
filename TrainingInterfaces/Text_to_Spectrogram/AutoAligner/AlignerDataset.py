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
            self.path_to_transcript_dict = dict(path_to_transcript_dict)
            key_list = list(self.path_to_transcript_dict.keys())
            fisher_yates_shuffle(key_list)
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
                transcript_list = list()
                for path_key in key_split:
                    transcript_list.append(self.path_to_transcript_dict[path_key])
                process_list.append(
                    Process(target=self.cache_builder_process,
                            args=(key_split,
                                  transcript_list,
                                  min_len_in_seconds,
                                  max_len_in_seconds,
                                  cut_silences,
                                  do_loudnorm,
                                  verbose,
                                  device,
                                  phone_input,
                                  allow_unknown_symbols,
                                  lang),
                            daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            print("pooling results...")
            pooled_datapoints = list()
            for chunk in self.datapoints:
                for datapoint in chunk:
                    pooled_datapoints.append(datapoint)  # unpack into a joint list
            self.datapoints = pooled_datapoints
            del pooled_datapoints
            print("converting text to tensors...")
            text_tensors = [torch.Tensor(x[0]) for x in self.datapoints]  # turn everything back to tensors (had to turn it to np arrays to avoid multiprocessing issues)
            print("converting specs to tensors...")
            speech_tensors = [torch.Tensor(x[1]) for x in self.datapoints]
            print("converting waves to tensors...")
            norm_waves = [torch.Tensor(x[2]) for x in self.datapoints]
            print("unpacking file list...")
            filepaths = [x[3] for x in self.datapoints]
            del self.datapoints
            print("done!")

            # now we add speaker embeddings
            self.speaker_embeddings = list()
            speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                    run_opts={"device": str(device)},
                                                                    savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
            with torch.inference_mode():
                for norm_wave in tqdm(norm_waves):
                    self.speaker_embeddings.append(speaker_embedding_func.encode_batch(wavs=norm_wave.unsqueeze(0).to(device)).squeeze().cpu())

            # save to cache
            if len(self.speaker_embeddings) == 0:
                raise RuntimeError

            self.datapoint_feature_dump_list = list()
            os.makedirs(os.path.join(cache_dir, f"aligner_datapoints/"), exist_ok=True)
            for index, (filepath, speech_tensor, text_tensor, speaker_embedding) in tqdm(enumerate(zip(filepaths, speech_tensors, text_tensors, self.speaker_embeddings)), total=len(filepaths)):
                torch.save(([text_tensor,
                             torch.LongTensor([len(text_tensor)]),
                             speech_tensor,
                             torch.LongTensor([len(speech_tensor)])],
                            speaker_embedding,
                            filepath),
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
                              transcript_list,
                              min_len,
                              max_len,
                              cut_silences,
                              do_loudnorm,
                              verbose,
                              device,
                              phone_input,
                              allow_unknown_symbols,
                              lang):

        process_internal_dataset_chunk = list()
        _, sr = sf.read(path_list[0])
        assumed_sr = sr
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=cut_silences, do_loudnorm=do_loudnorm, device=device)
        tf = ArticulatoryCombinedTextFrontend(language=lang)
        warnings.simplefilter("ignore")  # otherwise we get tons of warnings about an RNN not being in contiguous chunks

        for path, transcript in tqdm(zip(path_list, transcript_list), total=len(path_list)):
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
                    norm_wave = ap.normalize_audio(audio=wave)  # now we can be sure that the wave is 16kHz
            except ValueError:
                continue

            # raw audio preprocessing is done
            try:
                try:
                    cached_text = tf.string_to_tensor(transcript, handle_missing=False, input_phonemes=phone_input).squeeze(0)
                except KeyError:
                    if not allow_unknown_symbols:
                        continue  # we skip sentences with unknown symbols
                    cached_text = tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=phone_input).squeeze(0)
            except ValueError:
                # this can happen for Mandarin Chinese, when the syllabification of pinyin doesn't work. In that case, we just skip the sample.
                continue
            except KeyError:
                # this can happen for Mandarin Chinese, when the syllabification of pinyin doesn't work. In that case, we just skip the sample.
                continue
            cached_speech = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1).cpu()
            process_internal_dataset_chunk.append([cached_text.numpy(),
                                                   cached_speech.numpy(),
                                                   norm_wave.cpu().numpy(),
                                                   path])
        self.datapoints.append(process_internal_dataset_chunk)

    def __getitem__(self, index):
        path_to_datapoint_file = self.datapoint_feature_dump_list[index]
        datapoint, speaker_embedding, filepath = torch.load(path_to_datapoint_file, map_location='cpu')

        text_vector = datapoint[0]
        tokens = self.tf.text_vectors_to_id_sequence(text_vector=text_vector)
        tokens = torch.LongTensor(tokens)
        return tokens, \
            torch.LongTensor([len(tokens)]), \
            datapoint[2], \
            datapoint[3], \
            speaker_embedding

    def __len__(self):
        return len(self.datapoint_feature_dump_list)


def fisher_yates_shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
