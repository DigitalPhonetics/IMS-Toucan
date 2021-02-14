import json
import os
import random
import time

import soundfile as sf
import torch
import torchviz

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from SpeakerEmbedding.SiameseSpeakerEmbedding import SiameseSpeakerEmbedding
from SpeakerEmbedding.SpeakerEmbeddingDataset import SpeakerEmbeddingDataset


def build_sub_corpus(path_to_raw_corpus, path_to_dump, amount_of_samples_per_speaker=10):
    # make a dict with key being speaker and values
    # being lists of all their utterances as cleaned
    # waves. Dump them as jsons.
    ap = None
    done_with_speaker = False
    for speaker in os.listdir(path_to_raw_corpus):
        print("Collecting and normalizing speaker {}".format(speaker))
        speaker_to_melspecs = dict()
        for sub in os.listdir(os.path.join(path_to_raw_corpus, speaker)):
            for wav in os.listdir(os.path.join(path_to_raw_corpus, speaker, sub)):
                if ".wav" in wav:
                    try:
                        wave, sr = sf.read(os.path.join(path_to_raw_corpus, speaker, sub, wav))
                    except RuntimeError:
                        print("File {} seems to be faulty".format(os.path.join(speaker, sub, wav)))
                        continue
                    if ap is None:
                        ap = AudioPreprocessor(input_sr=sr, melspec_buckets=80, output_sr=16000)
                    # yeet the file if the audio is too short
                    if len(wave) < 6000:
                        continue
                    clean_wave = ap.audio_to_wave_tensor(wave)
                    if speaker not in speaker_to_melspecs:
                        speaker_to_melspecs[speaker] = list()
                    speaker_to_melspecs[speaker].append(clean_wave.numpy().tolist())
                    if len(speaker_to_melspecs[speaker]) >= amount_of_samples_per_speaker:
                        done_with_speaker = True
                        break
            if done_with_speaker:
                done_with_speaker = False
                break
        with open(os.path.join(path_to_dump, speaker + ".json"), 'w') as fp:
            json.dump(speaker_to_melspecs, fp)


def train_loop(net, train_dataset, eval_dataset, save_directory, epochs=100, batchsize=64, device="cuda"):
    start_time = time.time()
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(net.get_conf())
    val_loss_highscore = 100.0
    batch_counter = 0
    net.train()
    net.to_device(device)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        index_list = random.sample(range(len(train_dataset)), len(train_dataset))
        train_losses = list()
        # train one epoch
        for index in index_list:
            train_datapoint = train_dataset[index]
            train_loss = net(train_datapoint[0], train_datapoint[1], train_datapoint[2])
            train_losses.append(train_loss / batchsize)  # for accumulative gradient
            train_losses[-1].backward()
            batch_counter += 1
            if batch_counter % batchsize == 0:
                print("Step:         {}".format(batch_counter))
                optimizer.step()
                optimizer.zero_grad()
        # evaluate after epoch
        with torch.no_grad():
            net.eval()
            val_losses = list()
            for validation_datapoint_index in range(len(eval_dataset)):
                eval_datapoint = eval_dataset[validation_datapoint_index]
                val_losses.append(net(eval_datapoint[0], eval_datapoint[1], eval_datapoint[2]))
            val_loss = sum(val_losses) / len(val_losses)
            if val_loss_highscore > val_loss:
                val_loss_highscore = val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           save_directory / "checkpoint_{}.pt".format(round(val_loss, 4)))
            print("Epoch:        {}".format(epoch))
            print("Train Loss:   {}".format(sum(train_losses)))
            print("Valid Loss:   {}".format(val_loss))
            print("Time elapsed: {}".format(start_time - time.time()))
            net.train()


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show_model(net):
    print(net)
    print("\n\nNumber of Parameters: {}".format(count_parameters(net)))


def plot_model():
    sse = SiameseSpeakerEmbedding()
    out = sse(torch.rand((1, 1, 80, 2721)), torch.rand((1, 1, 80, 1233)), torch.Tensor([-1]))
    torchviz.make_dot(out.mean(), dict(sse.named_parameters())).render("speaker_emb_graph", format="png")


if __name__ == '__main__':
    start_stage = 3
    stop_stage = 99

    if start_stage <= 1 < stop_stage:
        print("Stage 1: Preparation")
        device = torch.device("cpu")
        if not os.path.exists("Corpora"):
            os.mkdir("Corpora")
        if not os.path.exists("Corpora/SpeakerEmbedding"):
            os.mkdir("Corpora/SpeakerEmbedding")
        if not os.path.exists("Corpora/SpeakerEmbedding/train"):
            os.mkdir("Corpora/SpeakerEmbedding/train")
        if not os.path.exists("Corpora/SpeakerEmbedding/valid"):
            os.mkdir("Corpora/SpeakerEmbedding/valid")
        if not os.path.exists("Models"):
            os.mkdir("Models")
        if not os.path.exists("Models/SpeakerEmbedding"):
            os.mkdir("Models/SpeakerEmbedding")
        path_to_feature_dump_train = "Corpora/SpeakerEmbedding/train/"
        path_to_feature_dump_valid = "Corpora/SpeakerEmbedding/valid/"
        path_to_raw_corpus_train = "/mount/arbeitsdaten46/projekte/dialog-1/tillipl/" \
                                   "datasets/VoxCeleb2/audio-files/train/dev/aac/"
        path_to_raw_corpus_valid = "/mount/arbeitsdaten46/projekte/dialog-1/tillipl/" \
                                   "datasets/VoxCeleb2/audio-files/test/aac/"

        if start_stage <= 2 < stop_stage:
            print("Stage 2: Feature Extraction")
            build_sub_corpus(path_to_raw_corpus_train, path_to_feature_dump_train)
            build_sub_corpus(path_to_raw_corpus_valid, path_to_feature_dump_valid)

            if start_stage <= 3 < stop_stage:
                print("Stage 3: Data Loading")
                train_data = SpeakerEmbeddingDataset(path_to_feature_dump_train, size=100000, device=device)
                valid_data = SpeakerEmbeddingDataset(path_to_feature_dump_valid, size=5000, device=device)

                if start_stage <= 4 < stop_stage:
                    print("Stage 4: Model Training")
                    model = SiameseSpeakerEmbedding()
                    train_loop(model, train_data, valid_data, "Models/SpeakerEmbedding", device=device)
