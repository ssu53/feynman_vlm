# %%

import json
from pathlib import Path
from PIL import Image
from easydict import EasyDict


# %%

name_dataset = "FeynEval-E"
name_dataset = "FeynEval-M"

if name_dataset == "FeynEval-E":
    path_dataset = Path("dataset/FeynEval-E/all")
    path_dataset_index = Path("dataset_index/FeynEval-E")

if name_dataset == "FeynEval-M":
    path_dataset = Path("dataset/FeynEval-M/diagrams")
    path_dataset_index = Path("dataset_index/FeynEval-M")

# %%

import pandas as pd

index_text = pd.read_csv(path_dataset_index / 'index_text.csv', index_col=0)
index_text

# %%


if name_dataset == "FeynEval-E":


    labels = pd.DataFrame(
        columns=['src'] + index_text.question.tolist())


    for path_sub in ['no', 'yes_single', 'yes_multiple']:

        path = path_dataset / path_sub

        for fn in path.iterdir():
            
            assert fn.name not in labels.index, f"Name confict {fn.name}"
            labels.loc[fn.name, 'src'] = path_sub

            # img = Image.open(fn)
            # display(img)


    labels["Does this image show a Feynman diagram(s)?"] = labels.src.apply(lambda src: src != 'no')
    # labels["Does this image show something other than a Feynman diagram(s)?"] = labels.src.apply(lambda src: src == 'no')


if name_dataset == "FeynEval-M":
    
    # get answers
    with open(f"dataset/{name_dataset}/answers.json") as f:
        answers = json.load(f)
    answers = {item['figure']: EasyDict(item) for item in answers}

    labels = pd.DataFrame(answers).T
    labels.drop(columns=["figure"], inplace=True)
    labels.sort_index(inplace=True)

    display(labels)



# %%

labels.to_csv(path_dataset_index / 'labels.csv')
# %%

index_image = pd.DataFrame(labels.index, columns=['name'])
index_image.to_csv(path_dataset_index / 'index_image.csv')
# %%

data = pd.DataFrame(columns=['image', 'text'])

for fn in labels.index:
    for txt in index_text.question:
        data = pd.concat(
            [data, pd.DataFrame([[fn, txt]], columns=data.columns)],
            ignore_index=True)

data.to_csv(path_dataset_index / 'data.csv')

# %%