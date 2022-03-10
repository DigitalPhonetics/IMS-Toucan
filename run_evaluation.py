import os

from tqdm import tqdm

from Utility.EvaluationScripts.audio_vs_audio import ffe
# from Utility.EvaluationScripts.playground_audio_vs_audio_poem import get_pitch_curves_abc
from Utility.EvaluationScripts.audio_vs_audio import gpe
from Utility.EvaluationScripts.audio_vs_audio import mcd_with_warping
from Utility.EvaluationScripts.audio_vs_audio import vde

# get_pitch_curves_abc(f"audios/ps2/PoetryStudy/Set5/s5_p1_ref2.wav", f"audios/ps2/PoetryStudy/Set5/s5_p1_ref1.wav", f"audios/ps2/PoetryStudy/Set5/s5_p1_base2_pros_1.wav")

mcd_same_style = list()
mcd_diff_style = list()

ffe_same_style = list()
ffe_diff_style = list()

gpe_same_style = list()
gpe_diff_style = list()

vde_same_style = list()
vde_diff_style = list()

for file in tqdm(os.listdir("audios/adept/human")):
    if file.endswith(".wav"):
        mcd_same_style.append(mcd_with_warping(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))
        vde_same_style.append(vde(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))
        gpe_same_style.append(gpe(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))
        ffe_same_style.append(ffe(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))

        mcd_diff_style.append(mcd_with_warping(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))
        vde_diff_style.append(vde(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))
        gpe_diff_style.append(gpe(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))
        ffe_diff_style.append(ffe(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))

print(mcd_same_style)
print(vde_same_style)
print(gpe_same_style)
print(ffe_same_style)

print(mcd_diff_style)
print(vde_diff_style)
print(gpe_diff_style)
print(ffe_diff_style)

print((1 / len(mcd_same_style)) * sum(mcd_same_style))
print((1 / len(vde_same_style)) * sum(vde_same_style))
print((1 / len(gpe_same_style)) * sum(gpe_same_style))
print((1 / len(ffe_same_style)) * sum(ffe_same_style))

print((1 / len(mcd_diff_style)) * sum(mcd_diff_style))
print((1 / len(vde_diff_style)) * sum(vde_diff_style))
print((1 / len(gpe_diff_style)) * sum(gpe_diff_style))
print((1 / len(ffe_diff_style)) * sum(ffe_diff_style))

"""
Results on ADEPT

25.628487893016743
0.3197801087172897
0.6701621670474431
0.6701621670474431

5.585193124795971
0.4236358253409133
0.7842741112107358
0.7842741112107358
"""
