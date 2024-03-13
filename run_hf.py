import numpy as np
import random
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
import argparse
from tqdm import tqdm

from easydict import EasyDict



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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


    # args = EasyDict(
    #     model_path = "llava-hf/llava-1.5-7b-hf",
    #     cache_dir = "/scratch/shiyesu/.cache/huggingface/hub",
    #     data_index_fn = "dataset_index/FeynEval-E/data.csv",
    #     image_dir = "dataset/FeynEval-E/all",
    #     out_fn = "results/llava-1.5-7b-hf",
    #     seed = 42,
    #     quantise = False,
    #     verbose = True,
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Model path", type=str, 
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--cache_dir", help="Hugging Face cache dir", type=str, 
                        default="/scratch/shiyesu/.cache/huggingface/hub")
    parser.add_argument("--data_index_fn", type=str, default="dataset_index/FeynEval-E/data.csv")
    parser.add_argument("--image_dir", type=str, default="dataset/FeynEval-E/all")
    parser.add_argument("--out_fn", type=str, default="results/llava")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantise", type=bool, help="Run with bitsandbytes quantisation", default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    print(args)


    assert args.model_path in ["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"]


    set_seed(args.seed)


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

        try: 

            inputs = processor(text=text, images=image, return_tensors="pt")
            if args.quantise: inputs = inputs.to('cuda')
            else: inputs = inputs.to(device)

            generate_ids = model.generate(**inputs, max_length=128)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            data.loc[ind,"output"] = parse_output(output)


            data.to_csv(f"{args.out_fn}.csv")
        
        except:
            # TODO: handle
            print(f"FAILED at index {ind}, {dat.image}, {dat.text}")

        pbar.update(1)



if __name__ == "__main__":
    main()
