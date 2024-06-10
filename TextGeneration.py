import torch
from fastai.text.all import *
from DataPreparation import *
import pathlib


class TextGeneration:
    def generate_poem(url, input_text):
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        tokenizer = DataPreparation.load_tokenizer()
        model = load_learner(url)

        prompt_ids = tokenizer.encode(input_text)
        inp = torch.tensor(prompt_ids)[None]
        preds = model.generate(
            inp, max_length=60, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        poem = tokenizer.decode(
            preds[0].cpu().numpy(), skip_special_tokens=True)

        pathlib.PosixPath = temp
        print(poem)
        return poem

