from multiprocessing import Manager
from multiprocessing import Process

import librosa
import numpy
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor


class HiFiGANDataset(Dataset):

    def __init__(self,
                 list_of_original_paths,
                 list_of_synthetic_paths,
                 desired_samplingrate=24000,
                 samples_per_segment=12288 * 2,  # = (8192 * 3) 2 , as I used 8192 for 16kHz previously
                 loading_processes=1):
        self.samples_per_segment = samples_per_segment
        self.desired_samplingrate = desired_samplingrate
        self.melspec_ap = AudioPreprocessor(input_sr=self.desired_samplingrate,
                                            output_sr=16000,
                                            cut_silence=False)
        # hop length of spec loss should be same as the product of the upscale factors
        # samples per segment must be a multiple of hop length of spec loss
        list_of_paths = list(zip(list_of_original_paths, list_of_synthetic_paths))
        if loading_processes == 1:
            self.waves = list()
            self.cache_builder_process(list_of_paths)
        else:
            resource_manager = Manager()
            self.waves = resource_manager.list()
            # make processes
            path_splits = list()
            process_list = list()
            for i in range(loading_processes):
                path_splits.append(list_of_paths[i * len(list_of_paths) // loading_processes:(i + 1) * len(
                    list_of_paths) // loading_processes])
            for path_split in path_splits:
                process_list.append(Process(target=self.cache_builder_process, args=(path_split,), daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
        print("{} eligible audios found".format(len(self.waves)))

    def cache_builder_process(self, path_split):
        for path in tqdm(path_split):
            try:
                path1, path2 = path

                wave, sr = sf.read(path1)
                if len(wave.shape) == 2:
                    wave = librosa.to_mono(numpy.transpose(wave))
                if sr != self.desired_samplingrate:
                    wave = librosa.resample(y=wave, orig_sr=sr, target_sr=self.desired_samplingrate)

                if len(wave) > self.samples_per_segment + 2000:
                    spec = torch.load(path2, map_location="cpu")
                    self.waves.append((wave, spec))
                else:
                    print("excluding short sample")

            except RuntimeError:
                print(f"Problem with the following path: {path}")

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of high-res audio and corresponding low-res spectrogram as if it was predicted by the TTS
        """
        wave = self.waves[index][0]
        wave = torch.Tensor(wave)

        spec = self.waves[index][1]

        spec_win, wave_win = get_matching_windows(waveform=wave, spectrogram=spec)
        return wave_win.detach(), spec_win.detach()

    def __len__(self):
        return len(self.waves)


def get_matching_windows(spectrogram, waveform, window_size_wave=24576, hop_length_spec=256, sample_rate_wave=24000, sample_rate_spec=16000):
    """
    Cut random matching windows from a spectrogram and waveform with perfectly aligned time axes.

    Parameters:
    - spectrogram: 2D numpy array (frames x freq_bins) of the spectrogram.
    - waveform: 1D numpy array of the ground truth waveform.
    - window_size_wave: Size of the window in waveform samples (default: 24576).
    - hop_length_spec: Hop length used for spectrogram extraction (default: 200 samples for 16 kHz).
    - sample_rate_wave: Sample rate of the waveform (default: 24000 Hz).
    - sample_rate_spec: Sample rate used to create the spectrogram (default: 16000 Hz).

    Returns:
    - spec_window: A window cut from the spectrogram.
    - wave_window: A window cut from the waveform.
    """
    spectrogram = spectrogram.transpose(0, 1)

    # Calculate the number of samples per spectrogram frame in waveform's time
    spec_frame_duration = hop_length_spec / sample_rate_spec
    wave_sample_duration = 1 / sample_rate_wave
    spec_to_wave_conversion_factor = wave_sample_duration / spec_frame_duration

    num_frames = int(window_size_wave * spec_to_wave_conversion_factor)

    # Ensure we can extract a full window from the spectrogram
    max_start_frame = spectrogram.shape[0] - num_frames
    if max_start_frame <= 0:
        print(f"desired num frames: {num_frames}")
        print(f"spec_to_wave_conversion_factor: {spec_to_wave_conversion_factor}")
        print(f"spec_len: {spectrogram.shape[0]}")
        raise ValueError("Spectrogram is too short to extract the desired window size.")

    # Randomly choose a start frame from the spectrogram
    start_frame = np.random.randint(0, max_start_frame)

    # Calculate the start sample for the waveform based on the chosen start frame
    start_sample = int(start_frame // spec_to_wave_conversion_factor)
    end_sample = start_sample + window_size_wave

    # Ensure the waveform can be fully sliced
    if end_sample > len(waveform):
        print(f"start_sample: {start_sample}")
        print(f"end_sample: {end_sample}")
        print(f"start_frame: {start_frame}")
        print(f"spec_to_wave_conversion_factor: {spec_to_wave_conversion_factor}")
        raise ValueError("Waveform is too short to extract the desired window size.")

    # Extract matching windows
    spec_window = spectrogram[start_frame:start_frame + num_frames, :].transpose(0, 1)
    wave_window = waveform[start_sample:end_sample]

    return spec_window, wave_window
