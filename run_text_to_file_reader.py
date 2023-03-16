import os

import torch

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

def read_goethe(model_id="Austrian", device="cpu", language="at", speaker=None):
    os.makedirs(os.path.join("audios",model_id,speaker), exist_ok=True)
    read_texts(model_id, "Meine Ruh ist hin.", os.path.join("audios",model_id,speaker,"goethe1.wav"), device="cpu", language="at")
    read_texts(model_id, "Mein Herz ist schwer.", os.path.join("audios",model_id,speaker,"goethe2.wav"), device="cpu", language="at")
    read_texts(model_id, "Ich finde sie nimmer und nimmermehr.",os.path.join("audios",model_id,speaker,"goethe3.wav"), device="cpu", language="at")
    read_texts(model_id, "Wo ich ihn nicht hab ist mir das Grab.", os.path.join("audios",model_id,speaker,"goethe4.wav"), device="cpu", language="at")
    read_texts(model_id, "Die ganze Welt ist mir vergällt.", os.path.join("audios",model_id,speaker,"goethe5.wav"), device="cpu", language="at")
    read_texts(model_id, "Mein armer Kopf ist mir verrückt.", os.path.join("audios",model_id,speaker,"goethe6.wav"), device="cpu", language="at")
    read_texts(model_id, "Mein armer Sinn ist mir zerstückt.", os.path.join("audios",model_id,speaker,"goethe7.wav"), device="cpu", language="at")
    read_texts(model_id, "Hallo, ich hoffe du, bist gut in deine Wohnung gekommen und hast heute einen interessanten Abend. Ich bin fertig für heute, oder was meinst du? Bis bald!", os.path.join("audios",model_id,speaker,"Halllo.wav"), device="cpu", language="at")

def read_goethe_at_lab(model_id="Austrian_From_Labels", device="cpu", language="at-lab", speaker=None):
    print("speaker in read is: " + speaker)
    os.makedirs(os.path.join("audios",model_id,speaker), exist_ok=True)
    read_texts(model_id, "Meine Ruh ist hin.", os.path.join("audios",model_id,speaker,"goethe1.wav"), device="cpu", language="at-lab",speaker=speaker)
    #read_texts(model_id, "Mein Herz ist schwer.", os.path.join("audios",model_id,speaker,"goethe2.wav"), device="cpu", language="at-lab")
    #read_texts(model_id, "Ich finde sie nimmer und nimmermehr.",os.path.join("audios",model_id,speaker,"goethe3.wav"), device="cpu", language="at-lab")
    #read_texts(model_id, "Wo ich ihn nicht hab ist mir das Grab.", os.path.join("audios",model_id,speaker,"goethe4.wav"), device="cpu", language="at-lab")
    #read_texts(model_id, "Die ganze Welt ist mir vergällt.", os.path.join("audios",model_id,speaker,"goethe5.wav"), device="cpu", language="at-lab")
    #read_texts(model_id, "Mein armer Kopf ist mir verrückt.", os.path.join("audios",model_id,speaker,"goethe6.wav"), device="cpu", language="at-lab")
    #read_texts(model_id, "Mein armer Sinn ist mir zerstückt.", os.path.join("audios",model_id,speaker,"goethe7.wav"), device="cpu", language="at-lab")
    #read_texts(model_id, "Hallo, ich hoffe du, bist gut in deine Wohnung gekommen und hast heute einen interessanten Abend. Ich bin fertig für heute, oder was meinst du? Bis bald!", os.path.join("audios",model_id,speaker,"Halllo.wav"), device="cpu", language="at-lab")

###
def read_austrian_test_set(model_id, device="cpu", language="", speaker=None, input_is_phones=True):
    tts = InferenceFastSpeech2(device=device, model_name=model_id, language=language)
    print(language)
    tts.set_language(language)
    print("model_id is: " + model_id)
    with open("/data/vokquant/data/aridialect/test-text_small_phoneme_input_mini.txt", "r") as test_files_list:
        for line in test_files_list:
            # stripped line = name of each txt_test_file"
            test_file_path = line.strip()
            test_file = os.path.split(test_file_path)[1]
            with open(test_file_path, "r") as txt_filename:
                sentence = txt_filename.read().replace('\n', '')
                filename_out=os.path.join("audios","Avocodo_" + model_id, f"{test_file.replace('.lab', '.wav')}")
                speaker = test_file.split("_")[0]
                print(filename_out)
                read_texts(model_id=model_id, sentence=sentence, filename=filename_out, device="cuda", language=language, speaker=speaker, input_is_phones=input_is_phones)
###

### Victor ################################################################
# I have added a flag in the third line of this function named Avocodo.   #
# It is meant to decide whether Avocodo or other train of HiFiGAN is used #
###########################################################################
def read_texts(model_id, sentence, filename, speaker=None, device="cpu", language="", input_is_phones=None):
    print("speaker in read texts is: ")
    print(str(speaker))
    tts = InferenceFastSpeech2(device=device, model_name=model_id,language=language,Avocodo=True)
    tts.set_language(language)
    tts.set_phoneme_input(input_is_phones)
    # select between {at_emb_trained, vd_emb_trained, ivg_emb_trained, goi_emb_trained, interp_at_vd_emb, spanish_emb, fr_emb }
    #lang_emb_avg = '/data/vokquant/IMS-Toucan_lang_emb/Preprocessing/embeds_mls_test/embeddings_from_45_model/at_emb.pt' # specify average language_embedding file (*.pt)
    lang_emb_avg = '/data/vokquant/IMS-Toucan_lang_emb/Preprocessing/embeds_mls_test/at_emb_trained.pt' # specify average language_embedding file (*.pt)
    #utt_emb = "/data/vokquant/data/aridialect/aridialect_wav16000/hga_vd_berlin_003.wav"
    if speaker == "spo":
        print("utterance embedding is set to spo_at_berlin_001:")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/spo_at_berlin_001.wav")
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    elif speaker == "hpo":
        print("utterance embedding is set to hpo_vd_wean_0002:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/hpo_vd_wean_0002.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    elif speaker== "hga":
        print("utterance embedding is set to hga_vd_berlin_003:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/hga_vd_berlin_003.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    elif speaker == "joe":
        print("utterance embedding is set to joe_vd_fritz_048:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/joe_vd_fritz_048.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    elif speaker == "hoi":
        print("utterance embedding is set to hoi_at_berlin_001:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/hoi_at_berlin_001.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    elif speaker == "gun":
        print("utterance embedding is set to gun_goi_goi_001:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/gun_goi_goi_001.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    elif speaker =="lsc":
        print("utterance embedding is set to lsc_ivg_ivg_009:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/lsc_ivg_ivg_009.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    else:
        print("speaker is None, I will set it to hpo_vd_wean_0002:")
        tts.set_utterance_embedding("/data/vokquant/data/aridialect/aridialect_wav16000/hpo_vd_wean_0002.wav")
        #tts.set_utterance_embedding(utt_emb)
        tts.set_language_embedding(lang_emb_avg,use_avg=True)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def read_texts_as_ensemble(model_id, sentence, filename, device="cpu", language="at", amount=10):
    """
    for this function, the filename should NOT contain the .wav ending, it's added automatically
    """
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language(language)
    if type(sentence) == str:
        sentence = [sentence]
    for index in range(amount):
        tts.default_utterance_embedding = torch.zeros(704).float().random_(-40, 40).to(device)
        tts.read_to_file(text_list=sentence, file_location=filename + f"_{index}" + ".wav")


def read_harvard_sentences(model_id, device):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


def read_contrastive_focus_sentences(model_id, device):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)

    with open("Utility/contrastive_focus_test_sentences.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/focus_{}".format(model_id)
    os.makedirs(output_dir, exist_ok=True)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("audios", exist_ok=True)
    #for speaker in {"hga", "joe", "hpo", "spo"}:
    #for speaker in {"hpo"}:
    #    print(speaker)
    #    read_goethe_at_lab(model_id="Austrian_From_Labels",speaker=speaker)
    #~nɑːxŋʔaxtsentnfiɐdlʔisɐʔeksɡɑːnŋɐhɛːnʔinʃbekdɐ~#
    read_austrian_test_set(model_id="Austrian_From_Labels_avg_lang_emb_trained_with_WASS", device="cpu", language="de", speaker="hpo", input_is_phones=True)
    #read_austrian_test_set(model_id="Austrian_From_Labels", language="de", speaker="hpo")
    #read_austrian_test_set(model_id="Meta", language="de", speaker="hpo", input_is_phones=False)
    #read_texts(model_id="Austrian_From_Labels", sentence="Viktor! Ich will nachhause.", filename="audios/will_nachhause.wav", device="cpu", language="at", speaker="hpo", input_is_phones=False)
    #read_texts(model_id="2Austrian_extended_phonemes", sentence="Hello everybody! This is the model speaking english with an english voice. I hope it works as it should.", filename="audios/english_multimodel_from_lab.wav", device="cpu", language="en")
    #read_texts(model_id="Austrian", sentence="Hallo, ich hoffe du, bist gut in deine Wohnung gekommen und hast heute einen interessanten Abend. Ich bin fertig für heute, oder was meinst du? Bis bald!", filename="audios/hallo_lia.wav", device="cpu", language="at")
    #read_texts(model_id="Austrian", sentence="Servus Freunde. Was sagts ihr zu meiner neuen Stimme?", filename="audios/servas.wav", device="cpu", language="at")
    #read_texts_as_ensemble(model_id="Austrian", sentence="Meine Ruh ist hin.", filename="audios/ensemble.wav")
