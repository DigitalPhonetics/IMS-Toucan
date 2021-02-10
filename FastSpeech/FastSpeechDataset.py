from torch.utils.data import Dataset


class FastSpeechDataset(Dataset):

    def __init__(self, device="cuda"):
        pass

    def __getitem__(self, index):
        # create tensor on correct device
        # return text_tensor, text_length, gold_speech, speech_length, gold_duration, duration_length, gold_pitch, pitch_length, gold_energy, energy_length, spemb
        pass

    def __len__(self):
        pass
