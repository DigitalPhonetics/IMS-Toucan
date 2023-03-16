# Description: loads pretrained or self trained embedding model and 
#               extracts the embedding vectors from audio and calculates
#               the average (np.mean) for each bundle (file_list)
# Output:   *.pt file
# Author:   Victor Garcia, Lorenz Gutscher
# year:     2022

from speechbrain.pretrained import EncoderClassifier
import torch
import os
from tqdm import tqdm
import numpy as np
from glob import glob
from numpy import trim_zeros
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import seaborn as sns
import soundfile as sf
from AudioPreprocessor import AudioPreprocessor
from pathlib import Path

device=torch.device("cpu")

# language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa",
#                                                         run_opts={"device": str(device)},
#                                                         savedir="pretrained_models/lang-id-commonlanguage_ecapa")

language_id = EncoderClassifier.from_hparams(source="/data/vokquant/speechbrain/recipes/CommonLanguage/lang_id/results/ECAPA-TDNN/1986/save/ckpt/",
                                                    run_opts={"device": str(device)},
                                                    hparams_file="/data/vokquant/speechbrain/recipes/CommonLanguage/lang_id/inference/hyperparams_inference.yaml",
                                                    savedir="pretrained_models/lang-id-commonlanguage_ecapa")

# It takes as input a list containing the paths to all wavs of a same language or variety and returns the average embedding (centroid (np.mean))
def get_language_embedding(filelist,path_to_wavs,language_id):
    temp=[]
    _, sr = sf.read(os.path.join(path_to_wavs,filelist[0]))
    print(sr)
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False, device=device)
    for wav in tqdm(filelist):
        wave, sr = sf.read(os.path.join(path_to_wavs,wav))
        # audio = language_id.load_audio(os.path.join(path_to_wavs,wav),savedir='/data/vokquant/IMS-Toucan_lang_emb/Preprocessing/softlinks_embedding')
        try:
            norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            continue
        norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
        embedding = language_id.encode_batch(norm_wave.cpu().detach())
        # add embedding of each wav to temp:
        temp.append(embedding)

    stack=np.stack(temp)
    # average all embeddings:
    embedding=np.mean(stack,axis=0)
    print(embedding)
    return embedding

WASS_corpora_generation = False
MLS_generation_spanish = True
MLS_generation_english = True
MLS_generation_dutch = True
MLS_generation_polish = True
MLS_generation_italian = True
MLS_generation_portuguese = True
MLS_generation_french = False
MLS_generation_german = True

root_path = '../../data/mls/common_voice_kpd/'

# generate spanish(sp) language_embedding average:
if MLS_generation_spanish==True:
    language_name='Spanish'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1] # only 1 file per folder is used [0:1]
    print(files)

    # Once you have the input list to the function, you call it
    embedding=get_language_embedding(files,path_to_wavs,language_id)

    # Finally, you save the embedding in the computer
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate english(en) language_embedding average:
if MLS_generation_english==True:
    language_name='English'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate dutch(du) language_embedding average:
if MLS_generation_dutch==True:
    language_name='Dutch'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate polish(pl) language_embedding average:
if MLS_generation_polish==True:
    language_name='Polish'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate italian(it) language_embedding average:
if MLS_generation_italian==True:
    language_name='Italian'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate portuguese(pt) language_embedding average:
if MLS_generation_portuguese==True:
    language_name='Portuguese'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate german(de) language_embedding average:
if MLS_generation_german==True:
    language_name='German'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

# generate french(fr) language_embedding average:
if MLS_generation_french==True:
    language_name='French'
    path_to_wavs = ''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(root_path+language_name+'/test/'):
        files += [os.path.join(dirpath, file) for file in filenames if file[-4:]=='.wav'][0:1]
    print(files)
    embedding=get_language_embedding(files,path_to_wavs,language_id)
    torch.save(embedding,language_name+'_emb_trained.pt')

if WASS_corpora_generation==True:

    path_to_wavs = '/data/vokquant/data/aridialect/aridialect_wav16000/'

    #This is used to generate the input list to the function get_language_embedding()
    ########################################################################################
    for paths,dirs,files in tqdm(os.walk(path_to_wavs)):
        at_files = [x for x in files if x.split('_')[1] == 'at']
        vd_files = [x for x in files if x.split('_')[1] == 'vd']
        goi_files = [x for x in files if x.split('_')[1] == 'goi']
        ivg_files = [x for x in files if x.split('_')[1] == 'ivg']
    ########################################################################################

    at_embedding=get_language_embedding(filelist=at_files,path_to_wavs=path_to_wavs,language_id=language_id)
    vd_embedding=get_language_embedding(vd_files,path_to_wavs,language_id)
    goi_embedding=get_language_embedding(goi_files,path_to_wavs,language_id)
    ivg_embedding=get_language_embedding(ivg_files,path_to_wavs,language_id)

    # Finally, you save the embedding in the computer
    torch.save(at_embedding,'at_emb_trained.pt')
    torch.save(vd_embedding,'vd_emb_trained.pt')
    torch.save(goi_embedding,'goi_emb_trained.pt')
    torch.save(ivg_embedding,'ivg_emb_trained.pt')
# torch.save(sp_embedding,'fr_emb.pt')

        # for j,wav in enumerate(wavs):
        #     if j>99:
        #         break
        #     spkr_names.append(dir)
        #     emb = generate_embedding(wav)
        #     print(emb.shape)
        #     temp.append(emb.squeeze().detach().cpu().numpy()) 
        #     all_embeddings.append(emb.squeeze().detach().cpu().numpy())
        # centroid = np.stack(temp)
        # print(centroid)
        # centroid = np.mean(centroid,axis=0)
        # print(centroid.shape)
        # spkr_names.append(dir + '_centroid')
        # # torch.save(centroid,os.path.join('./centroids_16k_denoised',dir+'.pt'))
        # all_embeddings.append(centroid)

# print(len(all_embeddings))
# print(len(spkr_names))
# centroids = [i.detach().cpu().numpy() for i in centroids]
# print(centroids)
# at_embedding = torch.load('./embeds/at_emb.pt')
# vd_embedding = torch.load('./embeds/vd_emb.pt')
# goi_embedding = torch.load('./embeds/goi_emb.pt')
# ivg_embedding = torch.load('./embeds/ivg_emb.pt')
# print(at_embedding.shape)
# print(vd_embedding.shape)
# print(goi_embedding.shape)
# print(ivg_embedding.shape)
# print(type(at_embedding))
# print(type(vd_embedding))
# print(type(goi_embedding))
# print(type(ivg_embedding))
# centroids = []
# centroids.append(at_embedding)
# centroids.append(vd_embedding)
# centroids.append(goi_embedding)
# centroids.append(ivg_embedding)
# print(len(centroids))


# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# transformed = tsne.fit_transform(centroids)

# data = {
#     "dim-1": transformed[:, 0],
#     "dim-2": transformed[:, 1],
#     "label": ['at','vd','goi','ivg'],
# }

# plt.figure()
# sns.scatterplot(
#     x="dim-1",
#     y="dim-2",
#     hue="label",
#     palette=sns.color_palette(n_colors=121),
#     data=data,
#     legend="full",
# )
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.savefig('imagen_centroids.png')

# print(all_embeddings)