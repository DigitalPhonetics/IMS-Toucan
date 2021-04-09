import torch
import torchaudio

wav2mel = torch.jit.load("../Models/Use/SpeakerEmbedding/wav2mel.pt")
dvector = torch.jit.load("../Models/Use/SpeakerEmbedding/dvector-step250000.pt").eval()

wav_tensor, sample_rate = torchaudio.load("../audios/default_spemb.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)
emb_tensor = dvector.embed_utterance(mel_tensor)
cached_spemb = emb_tensor.detach()

torch.save(cached_spemb, "../Models/Use/default_spemb.pt")
