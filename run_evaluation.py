import os

from tqdm import tqdm

from Utility.EvaluationScripts.audio_vs_audio import ffe
from Utility.EvaluationScripts.audio_vs_audio import mcd_with_warping

mcd_lyric_cloned = list()
mcd_lyric_uncond = list()
mcd_prose_cloned = list()
mcd_prose_uncond = list()

ffe_lyric_cloned = list()
ffe_lyric_uncond = list()
ffe_prose_cloned = list()
ffe_prose_uncond = list()

for file in tqdm(os.listdir("audios/evaluation/human")):
    if file.endswith(".wav"):
        mcd_lyric_cloned.append(mcd_with_warping(f"audios/evaluation/human/{file}",
                                                 f"audios/evaluation/poetry_cloned/{file.split('.')[0]}_poetic_cloned.wav"))
        mcd_lyric_uncond.append(mcd_with_warping(f"audios/evaluation/human/{file}",
                                                 f"audios/evaluation/poetry_unconditional/{file.split('.')[0]}_poetic_uncond.wav"))
        mcd_prose_cloned.append(mcd_with_warping(f"audios/evaluation/human/{file}",
                                                 f"audios/evaluation/prosa_cloned/{file.split('.')[0]}_prosa_cloned.wav"))
        mcd_prose_uncond.append(mcd_with_warping(f"audios/evaluation/human/{file}",
                                                 f"audios/evaluation/prosa_unconditional/{file.split('.')[0]}_prosa_uncond.wav"))

        ffe_lyric_cloned.append(ffe(f"audios/evaluation/human/{file}", f"audios/evaluation/poetry_cloned/{file.split('.')[0]}_poetic_cloned.wav"))
        ffe_lyric_uncond.append(ffe(f"audios/evaluation/human/{file}", f"audios/evaluation/poetry_unconditional/{file.split('.')[0]}_poetic_uncond.wav"))
        ffe_prose_cloned.append(ffe(f"audios/evaluation/human/{file}", f"audios/evaluation/prosa_cloned/{file.split('.')[0]}_prosa_cloned.wav"))
        ffe_prose_uncond.append(ffe(f"audios/evaluation/human/{file}", f"audios/evaluation/prosa_unconditional/{file.split('.')[0]}_prosa_uncond.wav"))

print((1 / len(mcd_lyric_cloned)) * sum(mcd_lyric_cloned))
print((1 / len(mcd_lyric_uncond)) * sum(mcd_lyric_uncond))
print((1 / len(mcd_prose_cloned)) * sum(mcd_prose_cloned))
print((1 / len(mcd_prose_uncond)) * sum(mcd_prose_uncond))

print((1 / len(ffe_lyric_cloned)) * sum(ffe_lyric_cloned))
print((1 / len(ffe_lyric_uncond)) * sum(ffe_lyric_uncond))
print((1 / len(ffe_prose_cloned)) * sum(ffe_prose_cloned))
print((1 / len(ffe_prose_uncond)) * sum(ffe_prose_uncond))
