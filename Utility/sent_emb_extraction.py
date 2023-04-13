from tqdm import tqdm

def extract_sent_embs(train_set, sent_emb_extractor):
    sent_embs = {}
    print("Extracting sentence embeddings.")
    for index in tqdm(range(len(train_set))):
        sentence = train_set[index][9]
        sent_emb = sent_emb_extractor.encode(sentences=[sentence]).squeeze()
        sent_embs[sentence] = sent_emb
    return sent_embs