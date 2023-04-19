from tqdm import tqdm

def extract_sent_embs(train_set, sent_emb_extractor, promptspeech=False):
    sent_embs = {}
    print("Extracting sentence embeddings.")
    if promptspeech:
        from Utility.path_to_transcript_dicts import build_sent_to_prompt_dict_promptspeech
        sent_to_prompt_dict = build_sent_to_prompt_dict_promptspeech()
    for index in tqdm(range(len(train_set))):
        sentence = train_set[index][9]
        if promptspeech:
            prompt = sent_to_prompt_dict[sentence]
            sent_emb = sent_emb_extractor.encode(sentences=[prompt]).squeeze()
        else:
            sent_emb = sent_emb_extractor.encode(sentences=[sentence]).squeeze()
        sent_embs[sentence] = sent_emb
    return sent_embs