import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
import argparse
from tqdm import tqdm
from utils import instructify, set_seed

from easydict import EasyDict



def prompt_llava(text):
    return f"<image>\nUSER: {text}\nASSISTANT:"



def parse_output_llava(text):
    return text.split('ASSISTANT: ')[-1]




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
    parser.add_argument("--instructify", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantise", help="Run with bitsandbytes quantisation", action='store_true')
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
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

        text = prompt_llava(instructify(dat.text, args.instructify))
        image = Image.open(f"{args.image_dir}/{dat.image}")

        try: 

            inputs = processor(text=text, images=image, return_tensors="pt")
            if args.quantise: inputs = inputs.to('cuda')
            else: inputs = inputs.to(device)

            generate_ids = model.generate(**inputs, max_new_tokens=128)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            data.loc[ind,"output"] = parse_output_llava(output)


            data.to_csv(f"{args.out_fn}.csv")
        
        except:
            # TODO: handle
            print(f"FAILED at index {ind}, {dat.image}, {text}")

        pbar.update(1)



if __name__ == "__main__":
    main()
