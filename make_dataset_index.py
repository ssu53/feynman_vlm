# %%

import json
from pathlib import Path
from PIL import Image
from easydict import EasyDict
import pandas as pd


# %%


def get_path_dataset(name_dataset):

    if name_dataset == "FeynEval-E":
        path_dataset = Path("dataset/FeynEval-E")

    elif name_dataset == "FeynEval-M":
        path_dataset = Path("dataset/FeynEval-M")

    elif name_dataset == "FeynEval-M-v2":
        path_dataset = Path("dataset/FeynEval-MH-March17")

    elif name_dataset == "FeynEval-H":
        path_dataset = Path("dataset/FeynEval-MH-March17")

    else:
        raise NotImplementedError

    return path_dataset



def get_path_dataset_index(name_dataset):

    if name_dataset == "FeynEval-E":
        path_dataset_index = Path("dataset_index/FeynEval-E")

    elif name_dataset == "FeynEval-M":
        path_dataset_index = Path("dataset_index/FeynEval-M")

    elif name_dataset == "FeynEval-M-v2":
        path_dataset_index = Path("dataset_index/FeynEval-M-v2")

    elif name_dataset == "FeynEval-H":
        path_dataset_index = Path("dataset_index/FeynEval-H")

    else:
        raise NotImplementedError

    return path_dataset_index



def get_labels_FeynEvalE(path_dataset):

    index_text = pd.read_csv("dataset_index/FeynEval-E/index_text.csv", index_col=0)
    
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

    return labels



def get_labels_FeynEvalM(path_dataset):
    
    # get answers
    with open(f"{path_dataset}/answers.json") as f:
        answers = json.load(f)
    answers = {item['figure']: EasyDict(item) for item in answers}

    labels = pd.DataFrame(answers).T
    labels.drop(columns=["figure"], inplace=True)
    labels.sort_index(inplace=True)

    return labels



def get_labels_FeynEvalMv2(path_dataset):
    
    # get answers
    with open(f"{path_dataset}/FeynEval-M-answers.json") as f:
        answers = json.load(f)
    answers = {item['figure']: EasyDict(item) for item in answers}

    labels = pd.DataFrame(answers).T
    labels.drop(columns=["figure"], inplace=True)
    labels.sort_index(inplace=True)

    return labels


def get_data_and_labels_FeynEvalH(path_dataset):

    # get answers
    with open(f"{path_dataset}/FeynEval-H.json") as f:
        answers = json.load(f)
    
    data = pd.DataFrame(answers)
    data.sort_index(inplace=True)
    data.drop(columns=["answer"], inplace=True)
    data.rename(columns={"figure": "image", "question": "text"}, inplace=True)
    
    labels = pd.DataFrame(answers)
    labels = labels.set_index('figure')
    labels.index.name = None
    labels.sort_index(inplace=True)
    labels.drop(columns=["question"], inplace=True)
    labels.rename(columns={"figure": "name", "answer": "integral"}, inplace=True)

    return data, labels



# %%


if __name__ == "__main__":

    # --------------------------------------------------
    # dataset to be indexed

    # name_dataset = "FeynEval-E"
    # name_dataset = "FeynEval-M"
    # name_dataset = "FeynEval-M-v2"
    name_dataset = "FeynEval-H"


    # --------------------------------------------------
    # get paths

    path_dataset = get_path_dataset(name_dataset)
    path_dataset_index = get_path_dataset_index(name_dataset)


    # --------------------------------------------------
    # get labels

    if name_dataset == "FeynEval-E":
        labels = get_labels_FeynEvalE(path_dataset)
    elif name_dataset == "FeynEval-M":
        lables = get_labels_FeynEvalM(path_dataset)
    elif name_dataset == "FeynEval-M-v2":
        labels = get_labels_FeynEvalMv2(path_dataset)
    elif name_dataset == "FeynEval-H":
        data, labels = get_data_and_labels_FeynEvalH(path_dataset)
    else:
        raise NotImplementedError

    labels.to_csv(path_dataset_index / 'labels.csv')

    
    # --------------------------------------------------
    # save image index

    index_image = pd.DataFrame(labels.index, columns=['name'])
    index_image.to_csv(path_dataset_index / 'index_image.csv')


    # --------------------------------------------------
    # make full index (text and image) to be queried

    if name_dataset == "FeynEval-H":
        pass

    else:
        
        index_text = pd.read_csv(path_dataset_index / 'index_text.csv', index_col=0)

        data = pd.DataFrame(columns=['image', 'text'])

        for fn in labels.index:
            for txt in index_text.question:
                data = pd.concat(
                    [data, pd.DataFrame([[fn, txt]], columns=data.columns)],
                    ignore_index=True)

    data.to_csv(path_dataset_index / 'data.csv')

