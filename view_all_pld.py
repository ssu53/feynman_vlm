# %%

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("data/pld")

plt.figure(figsize=(15,8))
rows = 4
cols = 6

for i,fn in enumerate(path.iterdir()):
    img = Image.open(fn)

    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# %%
