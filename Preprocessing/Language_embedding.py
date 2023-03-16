from speechbrain.pretrained import EncoderClassifier
import torch


class LanguageEmbedding:

    def __init__(self, device=torch.device("cpu")):
        # Use pretrained model:
        # self.language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa",
        #                                                   run_opts={
        #                                                       "device": str(device)},
        #                                                   savedir="pretrained_models/lang-id-commonlanguage_ecapa")

        #  Use 45+WASS model:
        self.language_id = EncoderClassifier.from_hparams(source="/data/vokquant/speechbrain/recipes/CommonLanguage/lang_id/results/ECAPA-TDNN/1986/save/ckpt/",
                                                          run_opts={"device": str(device)},
                                                          savedir="/data/vokquant/IMS-Toucan_lang_emb/Preprocessing/pretrained_models/lang-id-commonlanguage_ecapa")
    def get_language_embedding(self, input_waves=None):
        # print(input_waves.size())
        embeddings = self.language_id.encode_batch(input_waves)
        # print((embeddings.size()))
        return embeddings

    def get_emb_from_path(self, path_to_wavfile=None):
        audio = self.language_id.load_audio(path_to_wavfile)
        print(self.language_id.encode_batch(audio).size())
        return self.language_id.encode_batch(audio).squeeze(0)
        
if __name__ == '__main__':
    print('hi')
