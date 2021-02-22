from TransformerTTS.TransformerTTS import build_transformertts_model as build_trans, visualize_sanity_check as vis

if __name__ == '__main__':
    vis(model=build_trans(), sentence="Hallo")
