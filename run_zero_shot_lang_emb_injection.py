import argparse

from huggingface_hub import hf_hub_download

from Preprocessing.multilinguality.create_distance_lookups import CacheCreator
from Preprocessing.multilinguality.create_lang_dist_dataset import LangDistDatasetCreator
from Preprocessing.multilinguality.generate_zero_shot_lang_embs import approximate_and_inject_language_embeddings

if __name__ == "__main__":
    default_model_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default=default_model_path, help="model path from which to obtain pretrained language embeddings")
    parser.add_argument("--distance_type", "-d", type=str, choices=["map", "tree", "asp", "learned", "combined"], default="learned",
                        help="which type of distance to use for finding nearest languages")
    parser.add_argument("--n_closest", "-k", type=int, default=50, help="how many nearest languages to select for language embedding approximation")

    args = parser.parse_args()

    # make sure that cache files exist
    cc = CacheCreator(cache_root="Preprocessing/multilinguality")
    cc.create_required_files(model_path=hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt"))

    # create distance dataset
    dc = LangDistDatasetCreator(args.model_path, cache_root="Preprocessing/multilinguality")
    distance_dataset = dc.create_dataset(args.distance_type, n_closest=args.n_closest, zero_shot=True)

    # generate zero-shot lang embs and inject into pretrained model, then save to modified model path
    approximate_and_inject_language_embeddings(model_path=args.model_path,
                                               df=distance_dataset,
                                               iso_lookup=dc.iso_lookup)
