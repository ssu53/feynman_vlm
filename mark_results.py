# %%

import pandas as pd
import re
from utils import INTEGER_QUESTIONS, YESNO_QUESTIONS, KEYWORD_QUESTIONS


PARSE_COL = "output_parsed"
LABEL_COL = "labels"
ACC_COL = "acc"


# benchmark = "pld"
# benchmark = "FeynEval-E"
# benchmark = "FeynEval-M"
# benchmark = "FeynEval-M-v2"
benchmark = "FeynEval-H"

# model = "llava-1.5-7b-hf"
model = "llava-1.5-13b-hf"
# model = "gpt4v_instructify_int"
# model = "gpt4v_instructify_mc4"

results_file = f"{benchmark}_{model}.csv"

print(results_file)

results = pd.read_csv(f"results_raw/{results_file}", index_col=0)

# %%

def parse_integer(text, suffix=None):

    ans = [int(s) for s in re.findall(r'\b\d+\b', text)]

    if len(ans) == 1:
        return ans.pop()

    if len(ans) > 1:
        if suffix is not None:
            # search for number followed by a space and the suffix, e.g. "12 vertices"
            ans = [int(s.replace(f" {suffix}", "")) for s in re.findall(r'\b\d+ ' + suffix + r'\b', text)]
            if len(ans) == 1: return ans.pop()
        print(text)
        print(ans)
        return None


    integers = {
        "no": 0,
        "none": 0,
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
        if bool(re.search(r'\b' + integer + r'\b', text)):
            ans.append(integers[integer])
    
    if len(ans) == 1:
        return ans.pop()

    if len(ans) > 1:
        if suffix is not None:
            # search for number followed by a space and the suffix, e.g. "twelve vertices"
            ans = []
            for integer in integers:
                ans.extend(integers[s.replace(f" {suffix}", "")] for s in re.findall(r'\b' + integer + " " + suffix + r'\b', text))
            if len(ans) == 1: return ans.pop()
        print(text)
        print(ans)
        print("---------------------------------")
        return None

    # failed to find any matches
    print(text)
    print(ans)
    print("---------------------------------")
    return None
    


def parse_yesno(text):

    ans = []

    if bool(re.search(r'\byes\b', text)) or bool(re.search(r'\bYes\b', text)):
        ans.append(True)
    if bool(re.search(r'\bno\b', text)) or bool(re.search(r'\bNo\b', text)):
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



def parse_mc4(text):

    if text in ["A", "B", "C", "D"]:
        return text

    ans = []
    if "A)" in text: ans.append("A")
    if "B)" in text: ans.append("B")
    if "C)" in text: ans.append("C")
    if "D)" in text: ans.append("D")

    if len(ans) != 1:
        print(text)
        print(ans)
        return None

    return ans.pop()

# %%

if benchmark == "pld":

    labels = pd.read_csv(f"dataset_index/{benchmark}/labels.csv", index_col=0).set_index('name')


    for ind,row in results.iterrows():

        if row.text in INTEGER_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_integer(row.output)
        
        if row.text in YESNO_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_yesno(row.output)
        
        if row.text in KEYWORD_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_keyword(row.output)

        if row.text in INTEGER_QUESTIONS or row.text in YESNO_QUESTIONS:
            results.loc[ind,LABEL_COL] = labels.loc[row.image, row.text]

        if row.text in KEYWORD_QUESTIONS:
            results.loc[ind,LABEL_COL] = True


if benchmark == "FeynEval-E":

    labels = pd.read_csv(f"dataset_index/{benchmark}/labels.csv", index_col=0)
    results['src'] = [labels.src[x] for x in results.image]


    for ind,row in results.iterrows():
        
        if row.text in YESNO_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_yesno(row.output)
        
        if row.text in KEYWORD_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_keyword(row.output)

        if row.text in YESNO_QUESTIONS:
            results.loc[ind,LABEL_COL] = labels.loc[row.image, row.text]

        if row.text in KEYWORD_QUESTIONS:
            results.loc[ind,LABEL_COL] = False if row.src == "no" else True


if benchmark == "FeynEval-M" or benchmark == "FeynEval-M-v2":

    labels = pd.read_csv(f"dataset_index/{benchmark}/labels.csv", index_col=0)

    mapping = pd.read_csv(f"dataset_index/{benchmark}/index_text.csv", index_col=0).question.to_dict()
    mapping = {v:k for k,v in mapping.items()}

    for ind,row in results.iterrows():

        if row.output.startswith("I'm sorry") or row.output.startswith("Sorry, I can't"):
            results.loc[ind,PARSE_COL] = None
            continue
        
        if row.text in INTEGER_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_integer(
                row.output,
                suffix = 
                "vertices" if mapping[row.text] == "vertices" else 
                "internal" if mapping[row.text] == "edges" else
                "external" if mapping[row.text] == "legs" else
                None
            )
        
        if row.text in YESNO_QUESTIONS:
            results.loc[ind,PARSE_COL] = parse_yesno(row.output)

        if row.text in INTEGER_QUESTIONS or row.text in YESNO_QUESTIONS:
            results.loc[ind,LABEL_COL] = labels.loc[row.image, mapping[row.text]]


if benchmark == "FeynEval-H":

    labels = pd.read_csv(f"dataset_index/{benchmark}/labels.csv", index_col=0)

    for ind,row in results.iterrows():
        results.loc[ind,PARSE_COL] = parse_mc4(row.output)
        results.loc[ind,LABEL_COL] = labels.loc[row.image, "integral"]


# %%

# ------------------------------------------------------
# save down results
# do any manual parsing
# place final results in results_graded/

results.to_csv(f"results_scratch_{results_file}")
results = pd.read_csv(f"results_scratch_{results_file}", index_col=0)

# DO ANY MANUAL PARSING AND PUT RESULTS IN results_graded/

# %%

results = pd.read_csv(f"results_scratch_{results_file}", index_col=0)

results[ACC_COL] = (results[PARSE_COL] == results[LABEL_COL]).mask(results[PARSE_COL].isna(), None)

results.to_csv(f"results_scratch_{results_file}")
# %%