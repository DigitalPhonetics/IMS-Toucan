import os

from tqdm import tqdm

from Utility.EvaluationScripts.audio_vs_audio import ffe
from Utility.EvaluationScripts.audio_vs_audio import gpe
from Utility.EvaluationScripts.audio_vs_audio import mcd_with_warping
from Utility.EvaluationScripts.audio_vs_audio import vde

ffe(f"audios/adept/human/1.wav", f"audios/adept/same_voice_same_style/1.wav")
gpe(f"audios/adept/human/1.wav", f"audios/adept/same_voice_same_style/1.wav")
vde(f"audios/adept/human/1.wav", f"audios/adept/same_voice_same_style/1.wav")

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
        # vde_same_style.append(vde(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))
        # gpe_same_style.append(gpe(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))
        # ffe_same_style.append(ffe(f"audios/adept/human/{file}", f"audios/adept/same_voice_same_style/{file}"))

        mcd_diff_style.append(mcd_with_warping(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))
        # vde_diff_style.append(vde(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))
        # gpe_diff_style.append(gpe(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))
        # ffe_diff_style.append(ffe(f"audios/adept/human/{file}", f"audios/adept/same_voice_diff_style/{file}"))

print(mcd_same_style)
# print(vde_same_style)
# print(gpe_same_style)
# print(ffe_same_style)

print(mcd_diff_style)
# print(vde_diff_style)
# print(gpe_diff_style)
# print(ffe_diff_style)

print((1 / len(mcd_same_style)) * sum(mcd_same_style))
# print((1 / len(vde_same_style)) * sum(vde_same_style))
# print((1 / len(gpe_same_style)) * sum(gpe_same_style))
# print((1 / len(ffe_same_style)) * sum(ffe_same_style))

print((1 / len(mcd_diff_style)) * sum(mcd_diff_style))
# print((1 / len(vde_diff_style)) * sum(vde_diff_style))
# print((1 / len(gpe_diff_style)) * sum(gpe_diff_style))
# print((1 / len(ffe_diff_style)) * sum(ffe_diff_style))

"""
0.3197801087172897
0.6701621670474431
0.6701621670474431
0.4236358253409133
0.7842741112107358
0.7842741112107358
"""
