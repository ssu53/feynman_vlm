# %%

import pandas as pd
from utils import INTEGER_QUESTIONS, YESNO_QUESTIONS, KEYWORD_QUESTIONS



def subset(results, images=None, texts=None):

    res = results
    if images is not None: res = res[res.image.isin(images)]
    if texts is not None: res = res[res.text.isin(texts)]

    return res


def subset_acc(results, images=None, texts=None):

    res = subset(results, images, texts)
    # print(f"{res.acc.mean() * 100 : .0f}%")

    return res.acc.mean()


# %%

df = pd.DataFrame()

results_7b = pd.read_csv(f"results_graded/llava-1.5-7b-hf.csv", index_col=0)
results_13b = pd.read_csv(f"results_graded/llava-1.5-13b-hf.csv", index_col=0)

for question in KEYWORD_QUESTIONS + YESNO_QUESTIONS + INTEGER_QUESTIONS:
    df.loc[question, "llava-1.5-7b"] = subset_acc(results_7b, texts=[question])
    df.loc[question, "llava-1.5-13b"] = subset_acc(results_13b, texts=[question])

display((df*100).style.format('{:.0f}%'))
# %%
