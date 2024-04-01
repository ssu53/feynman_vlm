# %%

import pandas as pd
from PIL import Image
from make_dataset_index import get_path_dataset



PARSE_COL = "output_parsed"
LABEL_COL = "labels"
ACC_COL = "acc"
NO_RESP_COL = "NO RESPONSE PARSED"
ACC_EXCL_NAN_COL = "Acc.*"
ACC_INCL_NAN_COL = "Acc."



def subset(results, images=None, texts=None):

    res = results
    if images is not None: res = res[res.image.isin(images)]
    if texts is not None: res = res[res.text.isin(texts)]

    return res



def subset_acc(results, images=None, texts=None):

    res = subset(results, images, texts)
    # print(f"{res.acc.mean() * 100 : .0f}%")

    return res.acc.mean()



def summarise_pld_():

    df = pd.DataFrame()

    results_7b = pd.read_csv(f"results_graded/pld_llava-1.5-7b-hf.csv", index_col=0)
    results_13b = pd.read_csv(f"results_graded/pld_llava-1.5-13b-hf.csv", index_col=0)

    questions = set()
    questions.update(results_7b.text.unique())
    questions.update(results_13b.text.unique())

    for question in questions:
        df.loc[question, "llava-1.5-7b"] = subset_acc(results_7b, texts=[question])
        df.loc[question, "llava-1.5-13b"] = subset_acc(results_13b, texts=[question])

    display((df*100).style.format('{:.0f}%'))


   
def summarise_pld(results):
    df_summary = pd.DataFrame(
        index=results.text.unique(),
        columns=[ACC_EXCL_NAN_COL, ACC_INCL_NAN_COL, NO_RESP_COL],
    )

    for ind in df_summary.index:
        res = results[results.text == ind]
        df_summary.loc[ind,ACC_EXCL_NAN_COL] = res[ACC_COL].mean()
        df_summary.loc[ind,ACC_INCL_NAN_COL] = res[ACC_COL].fillna(False).mean()
        df_summary.loc[ind,NO_RESP_COL] = res[ACC_COL].isna().mean()


    return df_summary



def summarise_feynevale(results, by_src=True):

    if by_src:
        
        srcs = results.src.unique()
        dfs = []
        
        for src in srcs:
            df_summary = pd.DataFrame(
                index=results.text.unique(),
                columns=[ACC_EXCL_NAN_COL, ACC_INCL_NAN_COL, NO_RESP_COL],
            )

            for ind in df_summary.index:
                res = results[(results.text == ind) & (results.src == src)]
                df_summary.loc[ind,ACC_EXCL_NAN_COL] = res[ACC_COL].mean()
                df_summary.loc[ind,ACC_INCL_NAN_COL] = res[ACC_COL].fillna(False).mean()
                df_summary.loc[ind,NO_RESP_COL] = res[ACC_COL].isna().mean()

            dfs.append(df_summary)
        
        df_summary = pd.concat(dfs, axis=0, keys=srcs)
    
    else:
        df_summary = pd.DataFrame(
            index=results.text.unique(),
            columns=[ACC_EXCL_NAN_COL, NO_RESP_COL],
        )

        for ind in df_summary.index:
            res = results[results.text == ind]
            df_summary.loc[ind,ACC_EXCL_NAN_COL] = res[ACC_COL].mean()
            df_summary.loc[ind,ACC_INCL_NAN_COL] = res[ACC_COL].fillna(False).mean()
            df_summary.loc[ind,NO_RESP_COL] = res[ACC_COL].isna().mean()
    
    return df_summary



def summarise_feynevalm(results, by_loop=False):

    if by_loop:
        
        image_to_loop =  {x.image: int(x.labels) for _,x in results[results.text == "How many independent loops are there in the diagram?"].iterrows()}

        loops = set(image_to_loop.values())
        dfs = []

        for loop in loops:
            df_summary = pd.DataFrame(
                index=results.text.unique(),
                columns=[ACC_EXCL_NAN_COL, ACC_INCL_NAN_COL, NO_RESP_COL],
            )

            for ind in df_summary.index:
                res = results[(results.text == ind) & (results.image.apply(lambda x: image_to_loop[x] == loop))]                
                df_summary.loc[ind,ACC_EXCL_NAN_COL] = res[ACC_COL].mean()
                df_summary.loc[ind,ACC_INCL_NAN_COL] = res[ACC_COL].fillna(False).mean()
                df_summary.loc[ind,NO_RESP_COL] = res[ACC_COL].isna().mean()

            dfs.append(df_summary)
        
        df_summary = pd.concat(dfs, axis=0, keys=loops)


    else:
        df_summary = pd.DataFrame(
            index=results.text.unique(),
            columns=[ACC_EXCL_NAN_COL, ACC_INCL_NAN_COL, NO_RESP_COL],
        )

        for ind in df_summary.index:
            res = results[results.text == ind]
            df_summary.loc[ind,ACC_EXCL_NAN_COL] = res[ACC_COL].mean()
            df_summary.loc[ind,ACC_INCL_NAN_COL] = res[ACC_COL].fillna(False).mean()
            df_summary.loc[ind,NO_RESP_COL] = res[ACC_COL].isna().mean()


    return df_summary



def summarise_feynevalh(results):

    df_summary = pd.DataFrame(
        index=["integral"],
        columns=[ACC_EXCL_NAN_COL, ACC_INCL_NAN_COL, NO_RESP_COL],
    )

    df_summary.loc["integral", ACC_EXCL_NAN_COL] = results[ACC_COL].mean()
    df_summary.loc["integral", ACC_INCL_NAN_COL] = results[ACC_COL].fillna(False).mean()
    df_summary.loc["integral", NO_RESP_COL] = results[ACC_COL].isna().mean()

    return df_summary



def see_example(results, num_samples=1):

    path = str(get_path_dataset(benchmark)) + "/"
    path += "all/" if benchmark == "FeynEval-E" else \
            "diagrams/" if benchmark == "FeynEval-M" else \
            "diagrams/" if benchmark == "FeynEval-M-v2" else \
            "diagrams/" if benchmark == "FeynEval-H" else \
            ""

    for ind in results.sample(num_samples).index:
        print("ind", ind)
        img = Image.open(path + results.image[ind])
        display(img)
        for col in results.columns:
            print(f"{col}: {results.loc[ind,col]}")

    
# %%

if __name__ == "__main__":

    # benchmark = "pld"
    # benchmark = "FeynEval-E"
    # benchmark = "FeynEval-M"
    # benchmark = "FeynEval-M-v2"
    benchmark = "FeynEval-H"

    # model = "llava-1.5-7b-hf"
    # model = "llava-1.5-13b-hf"
    # model = "gpt4v"
    # model = "gpt4v_instructify_int"
    model = "gpt4v_instructify_mc4"

    results_file = f"{benchmark}_{model}.csv"
    results = pd.read_csv(f"results_graded/{results_file}", index_col=0)

    print(results_file)

    if benchmark == "pld":
        df_summary = summarise_pld(results)
    if benchmark == "FeynEval-E":
        df_summary = summarise_feynevale(results, by_src=True)
    if benchmark == "FeynEval-M" or benchmark == "FeynEval-M-v2":
        df_summary = summarise_feynevalm(results, by_loop=True)
    if benchmark == "FeynEval-H":
        df_summary = summarise_feynevalh(results)

    display((df_summary*100).style.format('{:.0f}%'))

    see_example(results, num_samples=3)


# %%
