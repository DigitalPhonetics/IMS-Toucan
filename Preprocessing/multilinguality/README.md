## Zero-Shot Approximation of Language Embeddings
This directory contains all scripts that are needed to reproduce the meta learning for zero-shot part of our system. These scripts allow you to predict representations of languages purely based on distances between them, as measured by a variety of linguistically informed metrics, or even better, a learned combination thereof.


### Applying zero-shot approximation to a trained model

Use `run_zero_shot_lang_emb_injection.py` to update the language embeddings of a trained model for all languages that were *not* seen during training (by default, `supervised_languages.json` is used to determine which languages *were* seen).
See the script for arguments that can be passed (e.g. to use a custom model path). Here is an example:
```
cd IMS-Toucan/
python run_zero_shot_lang_emb_injection.py -m <model_path> -d <distance_type> -k <number_of_nearest_neighbors>
```

By default, the updated model is saved with a modified filename in the same directory.

### Cached distance lookups
In order to apply any zero-shot approximation, cache files for distance lookups are required. 

The ASP lookup file (`asp_dict.pkl`) needs to be downloaded from the release page. All other cache files are automatically generated as required when running `run_zero_shot_lang_emb_injection.py`.

**Note:** While the map, tree, and inverse ASP distances are model independent, **the learned distance lookup is only applicable for the model it was trained on**, i.e., different Toucan models require different learned-distance lookups. If you want to apply zero-shot approximation to a new model, make sure that you are not using an outdated, pre-existing learned distance lookup, but instead train a new learned distance metric.
