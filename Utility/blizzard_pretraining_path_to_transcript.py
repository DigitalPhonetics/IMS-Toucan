import csv
import os
from pathlib import Path


# helper
def read_sep_data(filename, sep='\t', field=1):
    data = {}
    with open(filename) as f:
        for line in f:
            line = line.split(sep)
            data[line[0].strip()] = line[field].strip()
    return data


# PFC
def read_pfc():
    pfc_dict = read_sep_data('Utility/female_pfc_utterances.csv')
    filtered_dict = dict()
    for key in pfc_dict:
        if key.split("_")[-2].endswith("t"):  # only read speech subset
            filtered_dict[key] = pfc_dict[key]
    return filtered_dict


# MLS
def read_mls():
    transcripts = {split: read_sep_data(f'/resources/speech/corpora/MultiLingLibriSpeech/mls_french/{split}/transcripts.txt') for split in {'train', 'dev', 'test'}}
    file2spk = read_sep_data('Utility/female_mls.csv', sep=',')
    spk2purity = read_sep_data('Utility/female_mls_stats.csv', sep=',', field=-1)
    file_2_french_only_speakers = dict()
    for key in file2spk:
        if spk2purity[file2spk[key]] == "True":
            file_2_french_only_speakers[key] = file2spk[key]
    paths = [Path(filepath) for filepath in file_2_french_only_speakers.keys()]
    file2text = {str(filepath): transcripts[filepath.parts[6]][filepath.stem] for filepath in paths}
    return file2text


# Voxpopuli
# for raw data: row['raw_text'] in read_tsv()
# for normalized data: row['normalized_text'] in read_tsv()
def read_voxpopuli():
    def read_tsv(filename):
        data = {}
        with open(filename) as f:
            for row in csv.DictReader(f, delimiter='\t', quotechar='|'):
                data[row['id']] = row['raw_text']
        return data

    transcripts = {}
    for split in {'train', 'dev', 'test'}:
        transcripts.update(read_tsv(f'/resources/asr-data/voxpopuli/transcribed_data/fr/asr_{split}.tsv'))
    file2spk = read_sep_data('Utility/female_voxpopuli.csv', sep=',')
    paths = [Path(filepath) for filepath in file2spk.keys()]
    file2text = {filepath: transcripts[Path(filepath).stem] for filepath, _ in file2spk.items()}
    return file2text


def build_path_to_transcript_dict_blizzard2023_ad_double():
    root = "/mount/arbeitsdaten45/projekte/asr-4/denisopl/Blizzard2023/2sentences/output/AD"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_double():
    root = "/mount/arbeitsdaten45/projekte/asr-4/denisopl/Blizzard2023/2sentences/output/NEB"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace("»", '"').replace("«", '"')
    return path_to_transcript


if __name__ == '__main__':
    print(read_voxpopuli())
    print(read_pfc())
    print(read_mls())
