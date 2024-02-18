# %%

import pandas as pd
import re
from utils import INTEGER_QUESTIONS, YESNO_QUESTIONS, KEYWORD_QUESTIONS


PARSE_COl = "output_parsed"
LABEL_COL = "labels"
ACC_COL = "acc"


# results_file = "llava-1.5-7b-hf.csv"
results_file = "llava-1.5-13b-hf.csv"
results = pd.read_csv(f"results_raw/{results_file}", index_col=0)


def parse_integer(text):

    ans = [int(s) for s in re.findall(r'\b\d+\b', text)]

    if len(ans) == 1:
        return ans.pop()

    if len(ans) > 1:
        print(text)
        print(ans)
        return None


    integers = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten":  10,
    }

    ans = []
    for integer in integers:
        if integer in text:
            ans.append(integers[integer])
    
    if len(ans) != 1:
        print(text)
        print(ans)
        return None
    
    return ans.pop()


def parse_yesno(text):

    ans = []

    if "yes" in text or "Yes" in text:
        ans.append(True)
    if "no" in text or "No" in text:
        ans.append(False)
    
    if len(ans) != 1:
        print(text)
        print(ans)
        return None

    return ans.pop()


def parse_keyword(text, keywords=["feynman", "Feynman"]):
    for kw in keywords:
        if kw in text: return True
    return False

# %%

labels = pd.read_csv("data/labels.csv", index_col=0).set_index('name')

for ind,row in results.iterrows():

    if row.text in INTEGER_QUESTIONS:
        results.loc[ind,PARSE_COl] = parse_integer(row.output)
    
    if row.text in YESNO_QUESTIONS:
        results.loc[ind,PARSE_COl] = parse_yesno(row.output)
    
    if row.text in KEYWORD_QUESTIONS:
        results.loc[ind,PARSE_COl] = parse_keyword(row.output)

    if row.text in INTEGER_QUESTIONS or row.text in YESNO_QUESTIONS:
        results.loc[ind,LABEL_COL] = labels.loc[row.image, row.text]

    if row.text in KEYWORD_QUESTIONS:
        results.loc[ind,LABEL_COL] = True
    
# %%

results.to_csv(f"results_{results_file}")

# %%

results = pd.read_csv(f"results_{results_file}", index_col=0)

# %%

results[ACC_COL] = (results[PARSE_COl] == results[LABEL_COL])

# %%
results.to_csv(f"results_{results_file}")


# %%


