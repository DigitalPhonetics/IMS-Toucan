from torch.utils.data import Dataset


class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, path_to_feature_dump):
        self.len = True
        self.id_list = None
        # load json with IDs mapping to list of melspecs

    def __getitem__(self, item):
        self.same_different_toggle = not self.same_different_toggle
        if self.same_different_toggle:
            # return a sample pair where the speaker ID matches
            pass
        else:
            # return a sample pair with different speaker IDs
            pass

    def __len__(self):
        return self.len
