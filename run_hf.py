# %%

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
import argparse
from tqdm import tqdm



def prompt(text):
    return f"<image>\nUSER: {text}\nASSISTANT:"



def parse_output(text):
    return text.split('ASSISTANT: ')[-1]



def transparent_to_white(image):
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image.convert('RGB')
    return new_image




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Model path", type=str, 
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--cache_dir", help="Hugginface cache dir", type=str, 
                        default="/scratch/shiyesu/.cache/huggingface/hub")
    parser.add_argument("--data_index_fn", type=str, default="data/data.csv")
    parser.add_argument("--image_dir", type=str, default="data/pld")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--quantise", type=bool, help="Whether to run with bitsandbytes quantisation", default=True)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    print(args)


    assert args.model_path in ["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"]


    if args.quantise:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
        )
        device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        model.to(device)
    
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, cache_dir=args.cache_dir)


    data = pd.read_csv(args.data_index_fn, index_col=0)


    pbar = tqdm(
        total=len(data),
        desc=f"Querying...",
        disable=not args.verbose,
    )

    for ind,dat in data.iterrows():

        text = prompt(dat.text)
        image = Image.open(f"{args.image_dir}/{dat.image}")

        inputs = processor(text=text, images=image, return_tensors="pt")
        if args.quantise: inputs = inputs.to('cuda')
        else: inputs = inputs.to(device)

        generate_ids = model.generate(**inputs, max_length=128)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        data.loc[ind,"output"] = parse_output(output)


        data.to_csv(f"{args.out_dir}/{args.model_path.split('/')[-1]}.csv")

        pbar.update(1)



if __name__ == "__main__":
    main()
