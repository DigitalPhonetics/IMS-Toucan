import numpy
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from pedalboard import HighpassFilter
from pedalboard import LowpassFilter
from pedalboard import Pedalboard

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def ad_submission(device="cpu", verbose=False):
    tts = ToucanTTSInterface(device=device, tts_model_path="AD_finetuned_final",
                             embedding_model_path="Models/ToucanTTS_AD_finetuned_final/embedding_function.pt",
                             faster_vocoder=False)
    tts.set_language("fr")
    tts.set_utterance_embedding("audios/blizzard_references/AD_REFERENCE.wav")
    effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=200),
                                  LowpassFilter(cutoff_frequency_hz=12000)])
    print("generating AD_test")
    with open("2023-SH1_submission_directory/AD_test/AD_test.csv", encoding="utf8", mode="r") as f:
        prompts = f.read()
    filename_to_prompt = dict()
    for prompt in prompts.split("\n"):
        if prompt.strip() != "":
            filename_to_prompt[prompt.split("|")[0]] = prompt.split("|")[1]

    for filename in filename_to_prompt:
        prompt = filename_to_prompt[filename][1:].replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        if verbose:
            print(f"Now synthesizing: {prompt}")
        silence = torch.zeros([6000])
        wav = tts(prompt,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.0,
                  energy_variance_scale=1.0).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -33.0)
        wav = float2pcm(wav)
        sf.write(file=f"2023-SH1_submission_directory/AD_test/wav/{filename}.wav",
                 data=wav,
                 samplerate=sr)


def neb_submission(device="cpu", verbose=False):
    tts = ToucanTTSInterface(device=device, tts_model_path="NEB_finetuned_final",
                             embedding_model_path="Models/ToucanTTS_NEB_finetuned_final/embedding_function.pt",
                             faster_vocoder=False)
    tts.set_language("fr")
    tts.set_utterance_embedding("audios/blizzard_references/NEB_REFERENCE.wav")
    effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=60),
                                  LowpassFilter(cutoff_frequency_hz=12000)])

    print("generating NEB_test")
    with open("2023-FH1_submission_directory/NEB_test/NEB_test.csv", encoding="utf8", mode="r") as f:
        prompts = f.read()
    filename_to_prompt = dict()
    for prompt in prompts.split("\n"):
        if prompt.strip() != "":
            filename_to_prompt[prompt.split("|")[0]] = prompt.split("|")[1]

    for filename in filename_to_prompt:
        prompt = filename_to_prompt[filename][1:].replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        if verbose:
            print(f"Now synthesizing: {prompt}")
        silence = torch.zeros([6000])
        wav = tts(prompt,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.1,
                  energy_variance_scale=1.1).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -29.0)
        wav = float2pcm(wav)
        sf.write(file=f"2023-FH1_submission_directory/NEB_test/wav/{filename}.wav",
                 data=wav,
                 samplerate=sr)

    print("generating NEB_test_sus")
    with open("2023-FH1_submission_directory/NEB_test_sus/NEB_test_sus.csv", encoding="utf8", mode="r") as f:
        prompts = f.read()
    filename_to_prompt = dict()
    for prompt in prompts.split("\n"):
        if prompt.strip() != "":
            filename_to_prompt[prompt.split("|")[0]] = prompt.split("|")[1]

    for filename in filename_to_prompt:
        prompt = filename_to_prompt[filename][1:].replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        if verbose:
            print(f"Now synthesizing: {prompt}")
        silence = torch.zeros([6000])
        wav = tts(prompt,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.1,
                  pitch_variance_scale=1.1,
                  energy_variance_scale=1.1).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -29.0)
        wav = float2pcm(wav)
        sf.write(file=f"2023-FH1_submission_directory/NEB_test_sus/wav/{filename}.wav",
                 data=wav,
                 samplerate=sr)

    print("generating NEB_test_par")
    with open("2023-FH1_submission_directory/NEB_test_par/NEB_test_par.csv", encoding="utf8", mode="r") as f:
        prompts = f.read()
    filename_to_prompt = dict()
    for prompt in prompts.split("\n"):
        if prompt.strip() != "":
            filename_to_prompt[prompt.split("|")[0]] = prompt.split("|")[1]

    for filename in filename_to_prompt:
        prompt = filename_to_prompt[filename][1:].replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        if verbose:
            print(f"Now synthesizing: {prompt}")
        silence = torch.zeros([6000])
        wav = tts(prompt,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.1,
                  energy_variance_scale=1.1).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -29.0)
        wav = float2pcm(wav)
        sf.write(file=f"2023-FH1_submission_directory/NEB_test_par/wav/{filename}.wav",
                 data=wav,
                 samplerate=sr)

    print("generating NEB_test_list")
    with open("2023-FH1_submission_directory/NEB_test_list/NEB_test_list.csv", encoding="utf8", mode="r") as f:
        prompts = f.read()
    filename_to_prompt = dict()
    for prompt in prompts.split("\n"):
        if prompt.strip() != "":
            filename_to_prompt[prompt.split("|")[0]] = prompt.split("|")[1]

    for filename in filename_to_prompt:
        prompt = filename_to_prompt[filename][1:].replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        if verbose:
            print(f"Now synthesizing: {prompt}")
        silence = torch.zeros([6000])
        wav = tts(prompt,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.1,
                  energy_variance_scale=1.1).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -29.0)
        wav = float2pcm(wav)
        sf.write(file=f"2023-FH1_submission_directory/NEB_test_list/wav/{filename}.wav",
                 data=wav,
                 samplerate=sr)

    print("generating NEB_test_homos")
    with open("2023-FH1_submission_directory/NEB_test_homos/NEB_test_homos.csv", encoding="utf8", mode="r") as f:
        prompts = f.read()
    filename_to_prompt = dict()
    for prompt in prompts.split("\n"):
        if prompt.strip() != "":
            filename_to_prompt[prompt.split("|")[0]] = prompt.split("|")[1]

    for filename in filename_to_prompt:
        prompt = filename_to_prompt[filename][1:].replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        if verbose:
            print(f"Now synthesizing: {prompt}")
        silence = torch.zeros([6000])
        wav = tts(prompt,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.1,
                  energy_variance_scale=1.1).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -29.0)
        wav = float2pcm(wav)
        sf.write(file=f"2023-FH1_submission_directory/NEB_test_homos/wav/{filename}.wav",
                 data=wav,
                 samplerate=sr)


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    ad_submission(device=exec_device)

    neb_submission(device=exec_device)
