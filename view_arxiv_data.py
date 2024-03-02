# %%

import numpy as np
import json
from PIL import Image



def modify_path(path, mode):

    # str_from = "/scratch/gpfs/DANQIC/mm-science/data/arxiv_untar_gzs/"
    str_from = "/scratch/gpfs/DANQIC/mm-science/data/arxiv_jpg_1024/"   

    if mode == "feynman_in_caption":
        str_to = "data_arxiv/feynman_in_caption/"
    
    elif mode == "feynman_in_paragraph":
        str_to = "data_arxiv/feynman_in_paragraph_10k/"

    else:
        raise NotImplementedError

    return path.replace(str_from, str_to)





def get_random_image(mode, show_json: bool = False):

    if mode == "feynman_in_caption":
        path = "data_arxiv/feynman_in_caption.json"
    elif mode == "feynman_in_paragraph":
        path = "data_arxiv/feynman_in_paragraph_10k.json"
    else:
        raise NotImplementedError

    with open(path, "r") as f:
        labels = json.load(f)

    i = np.random.randint(low=0, high=len(labels))
    figure_path = modify_path(labels[i]['figure_path'], mode=mode)

    if show_json:
        print(f"Sampled index: {i}")
        for k in labels[i].keys():
            print("-----------------------------")
            print(k, labels[i][k])
    else:
        print(f"Sampled index: {i} | {figure_path}")

    img = Image.open(figure_path)
    return img


# %%

display(get_random_image(mode="feynman_in_paragraph", show_json=False))

# %%
