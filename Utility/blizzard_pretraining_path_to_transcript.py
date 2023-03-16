import csv
from pathlib import Path


# helper
def read_sep_data(filename, sep='\t'):
    data = {}
    with open(filename) as f:
        for line in f:
            line = line.split(sep)
            data[line[0].strip()] = line[1].strip()
    return data


# PFC
def read_pfc():
    pfc_dict = read_sep_data('female_pfc_utterances.csv')
    for key in pfc_dict:
        if not key.split("_")[-2].endswith("t"):  # only read speech subset
            pfc_dict.pop(key)
    return pfc_dict


# MLS
def read_mls():
    transcripts = {split: read_sep_data(f'/resources/speech/corpora/MultiLingLibriSpeech/mls_french/{split}/transcripts.txt') for split in {'train', 'dev', 'test'}}
    file2spk = read_sep_data('female_mls.csv', sep=',')
    paths = [Path(filepath) for filepath in file2spk.keys()]
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
    file2spk = read_sep_data('female_voxpopuli.csv', sep=',')
    paths = [Path(filepath) for filepath in file2spk.keys()]
    file2text = {filepath: transcripts[Path(filepath).stem] for filepath, _ in file2spk.items()}
    return file2text


if __name__ == '__main__':
    print(read_voxpopuli())
    print(read_pfc())
    print(read_mls())
