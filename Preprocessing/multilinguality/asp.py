import pickle
import json

def load_asp_dict(path_to_dict):
    if isinstance(path_to_dict, dict):
        return path_to_dict
    else:
        with open(path_to_dict, 'rb') as dictfile:
            asp_dict = pickle.load(dictfile)
        return asp_dict

def asp(lang_a, lang_b, path_to_dict):
    """Look up and return the ASP between lang_a and lang_b from (pre-calculated) dictionary at path_to_dict"""
    asp_dict = load_asp_dict(path_to_dict)
    lang_list = list(asp_dict.keys()) # list of all languages, to get lang_b's index
    lang_b_idx = lang_list.index(lang_b) # lang_b's index
    asp = asp_dict[lang_a][lang_b_idx] # asp_dict's structure: {lang: numpy array of all corresponding ASPs}
    
    return asp


if __name__ == "__main__":
    print(asp('deu', 'nld', './asp_dict.pkl')) # example: ASP between German and Dutch