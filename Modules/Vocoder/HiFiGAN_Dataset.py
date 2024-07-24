import os
import random
from multiprocessing import Manager
from multiprocessing import Process

import librosa
import numpy
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor


def random_pitch_shifter(x):
    n_steps = random.choice([-12, -9, -6, 3, 12])  # when using 12 steps per octave, these are the only ones that are pretty fast. I benchmarked it and the variance is many orders of magnitude.
    return torchaudio.transforms.PitchShift(sample_rate=24000, n_steps=n_steps)(x)


def polarity_inverter(x):
    return x * -1


class HiFiGANDataset(Dataset):

    def __init__(self,
                 list_of_paths,
                 desired_samplingrate=24000,
                 samples_per_segment=12288,  # = (8192 * 3) 2 , as I used 8192 for 16kHz previously
                 loading_processes=max(os.cpu_count() - 2, 1),
                 use_random_corruption=False):
        self.use_random_corruption = use_random_corruption
        self.samples_per_segment = samples_per_segment
        self.desired_samplingrate = desired_samplingrate
        self.melspec_ap = AudioPreprocessor(input_sr=self.desired_samplingrate,
                                            output_sr=16000,
                                            cut_silence=False)
        # hop length of spec loss should be same as the product of the upscale factors
        # samples per segment must be a multiple of hop length of spec loss
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
        # self.masker = torchaudio.transforms.FrequencyMasking(freq_mask_param=16, iid_masks=True)  # up to 16 consecutive bands can be masked, each element in the batch gets a different mask. Taken out because it seems too extreme.
        self.wave_augs = [random_pitch_shifter, polarity_inverter, lambda x: x, lambda x: x, lambda x: x, lambda x: x]  # just some data augmentation
        self.wave_distortions = [CodecSimulator(), lambda x: x, lambda x: x, lambda x: x, lambda x: x]  # simulating the fact, that we train the TTS on codec-compressed waves
        print("{} eligible audios found".format(len(self.waves)))

    def cache_builder_process(self, path_split):
        for path in tqdm(path_split):
            try:
                wave, sr = sf.read(path)
                if len(wave.shape) == 2:
                    wave = librosa.to_mono(numpy.transpose(wave))
                if sr != self.desired_samplingrate:
                    wave = librosa.resample(y=wave, orig_sr=sr, target_sr=self.desired_samplingrate)

                self.waves.append(wave)
            except RuntimeError:
                print(f"Problem with the following path: {path}")

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of high-res audio and corresponding low-res spectrogram as if it was predicted by the TTS
        """
        try:
            wave = self.waves[index]
            while len(wave) < self.samples_per_segment + 50:  # + 50 is just to be extra sure
                # catch files that are too short to apply meaningful signal processing and make them longer
                wave = numpy.concatenate([wave, numpy.zeros(shape=1000), wave])
                # add some true silence in the mix, so the vocoder is exposed to that as well during training
            wave = torch.Tensor(wave)

            if self.use_random_corruption:
                # augmentations for the wave
                wave = random.choice(self.wave_augs)(wave.unsqueeze(0)).squeeze(0)  # it is intentional that this affects the target as well. This is not a distortion, but an augmentation.

            max_audio_start = len(wave) - self.samples_per_segment
            audio_start = random.randint(0, max_audio_start)
            segment = wave[audio_start: audio_start + self.samples_per_segment]

            resampled_segment = self.melspec_ap.resample(segment).float()  # 16kHz spectrogram as input, 24kHz wave as output, see Blizzard 2021 DelightfulTTS
            if self.use_random_corruption:
                # augmentations for the wave
                resampled_segment = random.choice(self.wave_distortions)(resampled_segment.unsqueeze(0)).squeeze(0)
            melspec = self.melspec_ap.audio_to_mel_spec_tensor(resampled_segment,
                                                               explicit_sampling_rate=16000,
                                                               normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
            return segment.detach(), melspec.detach()
        except RuntimeError:
            print("encountered a runtime error, using fallback strategy")
            if index == 0:
                index = len(self.waves) - 1
            return self.__getitem__(index - 1)

    def __len__(self):
        return len(self.waves)


class CodecSimulator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=64)
        self.decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=64)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wav, sr = sf.read("../../audios/speaker_references/female_high_voice.wav")
    resampled_wave = torch.Tensor(librosa.resample(y=wav, orig_sr=sr, target_sr=24000))
    audio = torch.tensor(resampled_wave)
    melspec_ap = AudioPreprocessor(input_sr=24000,
                                   output_sr=16000,
                                   cut_silence=False)

    spec = melspec_ap.audio_to_mel_spec_tensor(melspec_ap.resample(resampled_wave).float(),
                                               explicit_sampling_rate=16000,
                                               normalize=False).transpose(0, 1)[:-1].transpose(0, 1)

    cs = CodecSimulator()
    masker = torchaudio.transforms.FrequencyMasking(freq_mask_param=16, iid_masks=True)  # up to 8 consecutive bands can be masked

    # testing codec simulator
    out = cs(resampled_wave.unsqueeze(0)).squeeze(0)

    plt.plot(resampled_wave, alpha=0.5)
    plt.plot(out, alpha=0.5)
    plt.title("Codec Simulator")
    plt.show()

    # testing spectrogram masking
    for _ in range(5):
        masked_spec = masker(spec.unsqueeze(0)).squeeze(0)
        print(masked_spec)
        plt.imshow(masked_spec.cpu().numpy(), origin="lower", cmap='GnBu')
        plt.title("Masked Spec")
        plt.show()

    # testing pitch shift
    for _ in range(5):
        shifted_wave = random_pitch_shifter(resampled_wave.unsqueeze(0)).squeeze(0)
        shifted_spec = melspec_ap.audio_to_mel_spec_tensor(melspec_ap.resample(shifted_wave).float(),
                                                           explicit_sampling_rate=16000,
                                                           normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
        plt.imshow(shifted_spec.detach().cpu().numpy(), origin="lower", cmap='GnBu')
        plt.title("Pitch Shifted Spec")
        plt.show()
