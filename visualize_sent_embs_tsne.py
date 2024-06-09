import os

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor as SentenceEmbeddingExtractor

def visualize_sent_embs(sent_embs, save_dir):
    # Prepare the data for t-SNE
    data_points = np.vstack([embedding.numpy() for embeddings in sent_embs.values() for embedding in embeddings])
    labels = np.concatenate([[i] * len(sent_embs[emotion]) for i, emotion in enumerate(sent_embs)])

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_result = tsne.fit_transform(data_points)

    # Plot the t-SNE points with colors corresponding to emotions
    color_mapping = {
    "anger": "red",
    "disgust": "purple",
    "fear": "black",
    "joy": "green",
    "neutral": "blue",
    "sadness": "gray",
    "surprise": "orange"
    }
    plt.figure(figsize=(10, 8))
    for i, emotion in enumerate(sent_embs):
        indices = np.where(labels == i)[0]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=emotion, color=color_mapping[emotion])

    plt.legend()
    # Save the figure
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation", "plots"), exist_ok=True)
    save_dir = os.path.join(PREPROCESSING_DIR, "Evaluation", "plots")

    #train_sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_balanced_10000_sent_embs_emoBERTcls.pt"), map_location='cpu')
    #visualize_sent_embs(train_sent_embs, os.path.join(save_dir, 'tsne_train_sent_embs.png'))

    #train_sent_embs_BERT = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "emotion_prompts_balanced_10000_sent_embs_BERT.pt"), map_location='cpu')
    #visualize_sent_embs(train_sent_embs_BERT, os.path.join(save_dir, 'tsne_train_sent_embs_BERT.png'))

    train_sent_embs_BERT = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "emotion_prompts_balanced_10000_sent_embs_stpara.pt"), map_location='cpu')
    visualize_sent_embs(train_sent_embs_BERT, os.path.join(save_dir, 'tsne_train_sent_embs_stpara.png'))

    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "test_dailydialogues_sent_embs_emoBERTcls.pt")):
        print("Extracting test sent embs...")
        sent_emb_extractor = SentenceEmbeddingExtractor(pooling='cls', device='cuda:5')
        test_sentences = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "test_sentences.pt"), map_location='cpu')
        for dataset, emotion_to_sents in test_sentences.items():
            for emotion, sentences in emotion_to_sents.items():
                test_sentences[dataset][emotion] = sentences[:50]

        test_dailydialogues_sent_embs = {"anger":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
        test_tales_sent_embs = {"anger":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}

        # dailydialogues
        for emotion, sents in tqdm(list(test_sentences['dailydialogues'].items())):
            for sent in sents:
                sent_emb = sent_emb_extractor.encode(sentences=[sent]).squeeze()
                test_dailydialogues_sent_embs[emotion].append(sent_emb)
        torch.save(test_dailydialogues_sent_embs, os.path.join(PREPROCESSING_DIR, "Evaluation", "test_dailydialogues_sent_embs_emoBERTcls.pt"))

        # tales
        for emotion, sents in tqdm(list(test_sentences['tales'].items())):
            for sent in sents:
                sent_emb = sent_emb_extractor.encode(sentences=[sent]).squeeze()
                test_tales_sent_embs[emotion].append(sent_emb)
        torch.save(test_tales_sent_embs, os.path.join(PREPROCESSING_DIR, "Evaluation", "test_tales_sent_embs_emoBERTcls.pt"))
    else:
        test_dailydialogues_sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "test_dailydialogues_sent_embs_emoBERTcls.pt"), map_location='cpu')
        test_tales_sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "test_tales_sent_embs_emoBERTcls.pt"), map_location='cpu')
    #visualize_sent_embs(test_dailydialogues_sent_embs, os.path.join(save_dir, 'tsne_test_dailydialogues_sent_embs.png'))
    #visualize_sent_embs(test_tales_sent_embs, os.path.join(save_dir, 'tsne_test_tales_sent_embs.png'))

    test_combined_sent_embs = {}
    for emotion in test_dailydialogues_sent_embs:
        test_combined_sent_embs[emotion] = test_dailydialogues_sent_embs[emotion] + test_tales_sent_embs[emotion]
    #visualize_sent_embs(test_combined_sent_embs, os.path.join(save_dir, 'tsne_test_combined_sent_embs.png')) 
