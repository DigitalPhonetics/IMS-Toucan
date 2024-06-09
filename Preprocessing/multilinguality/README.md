## Zero-Shot Approximation of Language Embeddings
This directory contains all scripts that are needed to reproduce the meta learning for zero-shot part of our system. These scripts allow you to predict representations of languages purely based on distances between them, as measured by a variety of linguistically informed metrics, or even better, a learned combination thereof.

### Learned distance metric
If you want to use a learned distance metric, you need to run `MetricMetaLearner.py` first to generate a lookup file for the learned distances.

Note: **The learned distances are (obviously) only useful for the model it was trained on**, i.e., different Toucan models require different learned-distance lookups.


### Applying zero-shot approximation to a trained model

Use `run_zero_shot_lang_emb_injection.py` to update the language embeddings of a trained model for all languages that were *not* seen during training (by default, `supervised_languages.json` is used to determine which languages *were* seen).
See the script for arguments that can be passed (e.g. to use a custom model path). Here is an example:
```
python run_zero_shot_lang_emb_injection.py --model_path <model_path> --distance_type <distance_type> --n_closest <number_of_nearest_neighbors>
```

By default, the updated model is saved with a modified filename in the same directory.