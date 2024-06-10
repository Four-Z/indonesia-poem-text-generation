import re
import pandas as pd


class Preprocessing:
    def load_data(tree, url):
        Preprocessing.clear_tree(tree)

        df = pd.read_csv(url, encoding="utf8")
        index = 1
        for _, row in df.iterrows():
            number = index
            puisi = row['puisi']
            tree.insert("", "end", values=(number, puisi))
            index += 1

    def preprocess(text):
        text = re.sub(r'\d+', '', text)  # menghapus angka pada puisi
        # mencari semua titik yang beruntun dan mengganti mereka dengan satu titik
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'[^\w\s.,]', '', text)
        text = text.lower()  # lowercase text
        return text

    def clear_tree(tree):
        for item in tree.get_children():
            tree.delete(item)

    def execute_preprocess(tree, url):
        df = pd.read_csv(url)
        df['puisi'] = df['puisi'].apply(lambda w: Preprocessing.preprocess(w))

        Preprocessing.clear_tree(tree)

        index = 1
        for _, row in df.iterrows():
            number = index
            puisi = row['puisi']
            tree.insert("", "end", values=(number, puisi))
            index += 1
