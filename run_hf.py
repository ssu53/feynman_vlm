# %%

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer

# %%

# model_path = "llava-hf/llava-1.5-7b-hf"
model_path = "llava-hf/llava-1.5-13b-hf"

print(model_path)

cache_dir = "/scratch/shiyesu/.cache/huggingface/hub"

model = LlavaForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
model.eval()
processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)

# %%

# device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model.to(device)

# %%

def prompt(text):
    return f"<image>\nUSER: {text}\nASSISTANT:"


def parse_output(text):
    return text.split('ASSISTANT: ')[-1]


# %%

data = pd.read_csv("data/data.csv", index_col=0)

for ind,dat in tqdm(data.iterrows()):

    text = prompt(dat.text)
    image = Image.open(f"data/pld/{dat.image}.png")

    inputs = processor(text=text, images=image, return_tensors="pt").to(device)

    generate_ids = model.generate(**inputs, max_length=128)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    data.loc[ind,"output"] = parse_output(output)


    data.to_csv(f"results/{model_path.split('/')[-1]}.csv")


# %%

def transparent_to_white(image):
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image.convert('RGB')
    return new_image

transparent_to_white(image)
# %%
