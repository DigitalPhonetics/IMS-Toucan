import os

import gdown

os.makedirs('../Models/PretrainedModel/', exist_ok=True)

url = 'google drive url goes here'
output_dir = '../Models/PretrainedModel/meta_taco_checkpoint.pt'
gdown.download(url, output_dir, quiet=False)

url = 'google drive url goes here'
output_dir = '../Models/PretrainedModel/meta_fast_checkpoint.pt'
gdown.download(url, output_dir, quiet=False)
