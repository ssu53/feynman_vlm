# Feynman diagram evaluation for multimodal VLM


## Installation

Tested in Python 3.8.10.

Subset of the complete list of packages...

```
pip install torch
pip install transformers
pip install tqdm
pip install numpy
pip install pandas
pip install pillow
pip install sentencepiece
pip install protobuf
pip install bitsandbytes
pip install accelerate
pip install openai
pip install easydict
```


## Evaluating llava

Run `run_hf.py` e.g. 

```
python run_hf.py --data_index_fn data_classification/data.csv --image_dir data_classification/all
```

By default quantises with bitandbytes using `bnb_4bit_compute_dtype=torch.bfloat16`.

