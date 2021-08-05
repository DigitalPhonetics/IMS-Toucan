import torch
import torchaudio

wav2mel = torch.jit.load("../Models/SpeakerEmbedding/wav2mel.pt")
dvector = torch.jit.load("../Models/SpeakerEmbedding/dvector-step250000.pt").eval()

wav_tensor, sample_rate = torchaudio.load("../audios/default_speaker_embedding.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)
emb_tensor = dvector.embed_utterance(mel_tensor)
cached_speaker_embedding = emb_tensor.detach()

torch.save(cached_speaker_embedding, "../Models/SpeakerEmbedding/default_speaker_embedding.pt")
