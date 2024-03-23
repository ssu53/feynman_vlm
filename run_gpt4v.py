import os
import base64
import requests

import pandas as pd

from easydict import EasyDict
from tqdm import tqdm
import argparse


# TODO: try/catch rate limit and waiting with asyncio



def set_openai_api_key(fn = 'openai_api_key.txt'):

    with open(fn) as f:
        api_key = f.read()

    os.environ['OPENAI_API_KEY'] = api_key

    print(f"Set the API key from {fn}.")



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



def get_header(api_key = None, reset_api_key: bool = False):

    if reset_api_key:
        set_openai_api_key()

    if api_key is None: 
       api_key = os.environ['OPENAI_API_KEY']
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    return headers



def get_payload(text, image, image_dir, max_tokens: int = 300, detail: str = "auto"):
    """
    Args
        image_dir: directory to fetch the images
        max_tokens: 300 is default from OpenAI docs examples
    """

    image_path = f"{image_dir}/{image}"
    base64_image = encode_image(image_path)
    # img = Image.open(image_path)

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{text}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail,
                }
                }
            ]
            }
        ],
        "max_tokens": max_tokens
    }

    return payload



def main():


    # args = EasyDict(
    #     data_index_fn = "data_classification/data.csv",
    #     image_dir = "data_classification/all",
    #     out_dir = "results",
    #     verbose = True,
    # )


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index_fn", type=str, default="dataset_index/FeynEval-E/data.csv")
    parser.add_argument("--image_dir", type=str, default="dataset/FeynEval-E/all")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--detail", type=str, default="auto")
    parser.add_argument("--out_fn", type=str, default="results/gpt4v")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    print(args)



    data = pd.read_csv(args.data_index_fn, index_col=0)

    # Subsample!
    # data = data.sample(12, replace=False)

    headers = get_header()
   

    pbar = tqdm(
        total=len(data),
        desc=f"Querying...",
        disable=not args.verbose,
    )

    for ind,dat in data.iterrows():

        payload = get_payload(
            dat.text, 
            dat.image, 
            image_dir = args.image_dir, 
            max_tokens = args.max_tokens,
            detail = args.detail,
        )

        print(dat.image, dat.text)
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        print(response.json())

        choices = response.json()['choices']
        assert len(choices) == 1
        output = choices[0]['message']['content']

        data.loc[ind,"output"] = output

        data.to_csv(f"{args.out_fn}.csv")

        pbar.update(1)



if __name__ == "__main__":
    main()
