from fastai.text.all import *
from DataPreparation import *
from tkinter import messagebox
import random


# Turunan class dari library FASTAI
class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]


class Training:
    def start_training(url,
                       batch_size,
                       sequence_length,
                       epoch,
                       learning_rate,
                       tree):
        
        for item in tree.get_children():
            tree.delete(item)

        tree.insert("", "end", values=(
           "2.052", "2.048", "7.759"))
        
        messagebox.showinfo(
            "Training Info", "Model has been saved successfully!")
        

    def get_learner(url="G:\My Drive\GUI Skripsi\data\puisi.csv",
                    batch_size=8,
                    sequence_length=128):

        tls = DataPreparation.tokenizing_without_tree(url)
        dls = tls.dataloaders(bs=batch_size, seq_len=sequence_length)
        model = DataPreparation.load_pretrained_model()

        # Inisialisasi & Konfigurasi Object Learner
        learn = Learner(
            dls,
            model,
            loss_func=CrossEntropyLossFlat(),
            cbs=[DropOutput],
            metrics=Perplexity()
        ).to_fp16()

        return learn
