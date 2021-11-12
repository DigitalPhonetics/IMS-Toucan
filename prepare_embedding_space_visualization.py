import json
import os

import torch

from InferenceInterfaces.InferenceArchitectures.InferenceFastSpeech2 import FastSpeech2
from InferenceInterfaces.InferenceArchitectures.InferenceTacotron2 import Tacotron2
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend


class ZeroShot_Vis(torch.nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.text2phone = ArticulatoryCombinedTextFrontend(language="de", inference=True)
        self.phone2mel = Tacotron2(path_to_weights=os.path.join("Models", "Singe_Step_LAML_Tacotron2", "best.pt")).to(torch.device(device))
        self.to(torch.device(device))
        phone_to_embedding = dict()
        for phone in self.text2phone.phone_to_vector:
            print(phone)
            phone_to_embedding[phone] = self.phone2mel.enc.embed(torch.tensor(self.text2phone.phone_to_vector[phone]).float()).detach().numpy().tolist()
        with open("Preprocessing/embedding_table_512dim.json", 'w', encoding="utf8") as fp:
            json.dump(phone_to_embedding, fp)

        self.phone2mel = FastSpeech2(path_to_weights=os.path.join("Models", "Singe_Step_LAML_FastSpeech2", "best.pt")).to(torch.device(device))
        self.to(torch.device(device))
        phone_to_embedding = dict()
        for phone in self.text2phone.phone_to_vector:
            print(phone)
            phone_to_embedding[phone] = self.phone2mel.encoder.embed(torch.tensor(self.text2phone.phone_to_vector[phone]).float()).detach().numpy().tolist()
        with open("Preprocessing/embedding_table_384dim.json", 'w', encoding="utf8") as fp:
            json.dump(phone_to_embedding, fp)


if __name__ == '__main__':
    ZeroShot_Vis()
