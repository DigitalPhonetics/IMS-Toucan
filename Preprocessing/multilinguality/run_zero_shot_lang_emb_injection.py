import os
import argparse
from Utility.storage_config import MODELS_DIR
from create_distance_lookups import CacheCreator
from create_lang_dist_dataset import LangDistDatasetCreator
from generate_zero_shot_lang_embs import approximate_and_inject_language_embeddings

if __name__ == "__main__":
    default_model_path = os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt") # MODELS_DIR must be absolute path, the relative path will fail at this location    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default=default_model_path, help="model path from which to obtain pretrained language embeddings")
    parser.add_argument("--distance_type", "-d", type=str, choices=["map", "tree", "asp", "learned", "combined"], default="learned", 
                        help="which type of distance to use for finding nearest languages")
    parser.add_argument("--n_closest", "-k", type=int, default=50, help="how many nearest languages to select for language embedding approximation")
    parser.add_argument("--learned_dist_path", type=str, default="lang_1_to_lang_2_to_learned_dist.json", 
                        help="filepath of JSON file containing the meta-learned pairwise distances")

    args = parser.parse_args()

    # make sure that cache files exist
    cc = CacheCreator()
    cc.create_required_files()

    # create distance dataset
    dc = LangDistDatasetCreator(args.model_path, args.learned_dist_path)
    distance_dataset = dc.create_dataset(args.distance_type, n_closest=args.n_closest, zero_shot=True)

    # generate zero-shot lang embs and inject into pretrained model, then save to modified model path
    approximate_and_inject_language_embeddings(model_path=args.model_path, 
                                               df=distance_dataset,
                                               iso_lookup=dc.iso_lookup)

