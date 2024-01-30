import os

import pyloudnorm
import torch

from Architectures.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Architectures.Vocoder.HiFiGAN_Generator import HiFiGAN
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Utility.storage_config import MODELS_DIR


class ToucanTTSInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",  # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, f"ToucanTTS_Meta", "best.pt"),  # path to the ToucanTTS checkpoint or just a shorthand if run standalone
                 vocoder_model_path=os.path.join(MODELS_DIR, f"Vocoder", "best.pt"),  # path to the Vocoder checkpoint
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            tts_model_path = os.path.join(MODELS_DIR, f"ToucanTTS_{tts_model_path}", "best.pt")

        #####################################
        #   load phone to features model    #
        #####################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')
        self.use_lang_id = True
        self.phone2mel = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"])
        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        self.phone2mel = self.phone2mel.to(torch.device(device))

        ################################
        #  load mel to wave model      #
        ################################
        vocoder_checkpoint = torch.load(vocoder_model_path, map_location="cpu")
        self.vocoder = HiFiGAN()
        self.vocoder.load_state_dict(vocoder_checkpoint)
        self.vocoder = self.vocoder.to(device).eval()
        self.vocoder.remove_weight_norm()
        self.meter = pyloudnorm.Meter(24000)

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, device=device)
        self.phone2mel.eval()
        self.vocoder.eval()
        self.to(torch.device(device))
        self.eval()

    def forward(self,
                latent_sequence,
                spk_embed,
                loudness_in_db=-24.0,
                glow_sampling_temperature=0.2):
        with torch.inference_mode():
            mel = self.phone2mel(latent_sequence,
                                 spk_embed,
                                 glow_sampling_temperature=glow_sampling_temperature)
            wave, _, _ = self.vocoder(mel.unsqueeze(0))
            wave = wave.squeeze().cpu().numpy()
        try:
            loudness = self.meter.integrated_loudness(wave)
            wave = pyloudnorm.normalize.loudness(wave, loudness, loudness_in_db)
        except ValueError:
            # if the audio is too short, a value error will arise
            pass
        return wave  # returns a 24kHz wave with normalized loudness
