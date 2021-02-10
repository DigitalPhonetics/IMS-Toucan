from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class SpeakerEmbeddingDataset(Dataset):
    def __init__(self):
        self.ap = AudioPreprocessor(input_sr=16000, melspec_buckets=512)

    def __getitem__(self, item):
        pass
