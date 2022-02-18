"""
https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german
"""

import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor

LANG_ID = "de"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
SAMPLES = 10

test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array = batch["audio"]["array"]
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch


test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence)
