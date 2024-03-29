import numpy as np
import random
import torch
from PIL import Image



def instructify(text, format="none"):
    if format == "none":
        return text
    if format == "int":
        instruction = "State your answer as a single integer."
        return text + " " + instruction
    if format == "int_abstain":
        instruction = "State your answer as a single integer, or `None` if unsure."
        return text + " " + instruction
    if format == "mc4":
        instruction = "State your answer simply as a one of: A, B, C, D. Do NOT explain or justify."
        return text + " " + instruction
    if format == "mc4_abstain":
        instruction = "State your answer simply as one of: A, B, C, D, or `None` if unsure. Do NOT explain or justify."
        return text + " " + instruction
    raise NotImplementedError



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def transparent_to_white(image):
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image.convert('RGB')
    return new_image



INTEGER_QUESTIONS = [
    "How many external legs are in this Feynman diagram?",
    "How many loops are in this Feynman diagram?",
    "How many internal edges are in this Feynman diagram?",
    "How many vertices are in this Feynman diagram?",
    "How many different particle types are in this Feynman diagram?",

    "How many independent loops are there in the diagram?",
    "How many external legs are there in the diagram?",
    "How many internal edges are there in the diagram?",
    "How many interaction vertices are there in the diagram?",
    "How many different kinds of particles are there in the diagram?",
]

YESNO_QUESTIONS = [
    "Does this image show a Feynman diagram?",
    "Does this image show a Feynman diagram(s)?",
]

KEYWORD_QUESTIONS = [
    "What is the content of this image?",
]