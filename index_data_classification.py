# %%

from pathlib import Path
from PIL import Image


# %%


path_data = Path('data_classification')

# %%

import pandas as pd

index_text = pd.read_csv(path_data / 'index_text.csv', index_col=0)
index_text

# %%

labels = pd.DataFrame(
    columns=['src'] + index_text.question.tolist())


for path_sub in ['no', 'yes_single', 'yes_multiple']:

    path = path_data / path_sub

    for fn in path.iterdir():
        
        assert fn.name not in labels.index, f"Name confict {fn.name}"
        labels.loc[fn.name, 'src'] = path_sub

        # img = Image.open(fn)
        # display(img)


labels["Does this image show a Feynman diagram(s)?"] = labels.src.apply(lambda src: src != 'no')
labels["Does this image show something other than a Feynman diagram(s)?"] = labels.src.apply(lambda src: src == 'no')


# %%

labels.to_csv(path_data / 'labels.csv')
# %%

index_image = pd.DataFrame(labels.index, columns=['name'])
index_image.to_csv(path_data / 'index_image.csv')
# %%

data = pd.DataFrame(columns=['image', 'text'])

for fn in labels.index:
    for txt in index_text.question:
        data = pd.concat(
            [data, pd.DataFrame([[fn, txt]], columns=data.columns)],
            ignore_index=True)

data.to_csv(path_data / 'data.csv')

# %%