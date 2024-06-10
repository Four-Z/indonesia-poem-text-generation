from transformers import AutoTokenizer, AutoModelForCausalLM
from fastai.text.all import *
import pandas as pd
from Preprocessing import *


# Turunan class dari library FASTAI yang berfungsi untuk mengubah teks menjadi representasi yang sesuai untuk model transformer
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))

    def decodes(self, x):
        decoded_text = self.tokenizer.decode(x.cpu().numpy())
        return TitledStr(decoded_text)


class DataPreparation:
    def load_pretrained_model():
        # Tentukan direktori tempat Anda menyimpan model
        local_model_directory = "./pretrained/model"

        # Muat model dari penyimpanan lokal
        model = AutoModelForCausalLM.from_pretrained(
            local_model_directory, local_files_only=True)
        return model

    def load_tokenizer():
        # Tentukan direktori tempat Anda menyimpan tokenizer
        local_tokenizer_directory = "./pretrained/tokenizer"

        # Muat tokenizer dari penyimpanan lokal
        tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_directory, local_files_only=True)

        return tokenizer

    def tokenizing(tree, url):
        df = pd.read_csv(url, encoding="utf8")
        df['puisi'] = df['puisi'].apply(lambda w: Preprocessing.preprocess(w))
        all_puisi = df['puisi'].tolist()

        # Mentokenisasi dataset dan membagi data dengan ratio 8:2 untuk training dan validasi
        train_len = int(len(all_puisi) * 0.8)  # use a 80/20 split
        val_len = len(all_puisi) - train_len

        splits = [range_of(train_len), range(val_len)]

        tokenizer = DataPreparation.load_tokenizer()
        tls = TfmdLists(all_puisi, TransformersTokenizer(
            tokenizer), splits=splits, dl_type=LMDataLoader)

        Preprocessing.clear_tree(tree)

        for i in range(len(tls)):
            number = i+1
            puisi = tls[i].tolist()
            tree.insert("", "end", values=(number, puisi))

        return tls

    def tokenizing_without_tree(url):
        df = pd.read_csv(url, encoding="utf8")
        df['puisi'] = df['puisi'][:10].apply(
            lambda w: Preprocessing.preprocess(w))
        all_puisi = df['puisi'][:10].tolist()

        # Mentokenisasi dataset dan membagi data dengan ratio 8:2 untuk training dan validasi
        train_len = int(len(all_puisi) * 0.8)  # use a 80/20 split
        val_len = len(all_puisi) - train_len

        splits = [range_of(train_len), range(val_len)]

        tokenizer = DataPreparation.load_tokenizer()
        tls = TfmdLists(all_puisi, TransformersTokenizer(
            tokenizer), splits=splits, dl_type=LMDataLoader)

        return tls
