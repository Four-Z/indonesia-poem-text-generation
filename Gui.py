import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from Preprocessing import *
from DataPreparation import *
from Training import *
from TextGeneration import *
from tkinter import filedialog
from tkinter import messagebox


class TransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))

    def decodes(self, x):
        decoded_text = self.tokenizer.decode(x.cpu().numpy())
        return TitledStr(decoded_text)


class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]


class MainWindow:
    def __init__(self, root):
        self.root = root

        # variable storage
        self.dataset_url = tk.StringVar()
        self.batch_size = tk.StringVar()
        self.sequence_length = tk.StringVar()
        self.epoch = tk.StringVar()
        self.learning_rate = tk.StringVar()
        self.model_url = tk.StringVar()
        self.input_text = tk.StringVar()

        tabControl = ttk.Notebook(root)
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tabControl.add(tab1, text='Fine Tuning Model')
        tabControl.add(tab2, text='Teks Generasi')
        tabControl.pack(expand=1, fill="both")

        root.title("Generasi Teks Puisi Indonesia")
        root.configure(bg="#eeeeee")
        self.menu_bar = tk.Menu(root, bg="#eeeeee", relief=tk.FLAT)
        root.configure(menu=self.menu_bar)

        ##### TAB 1 ######

        ### Title ###
        self.title = tk.Label(
            tab1,
            text="Generasi Teks Puisi Indonesia",
            bg="#eeeeee",
            font=("Helvetica", 21, 'bold')
        ).place(
            x=30,
            y=20
        )

        ### Choose Dataset Label ###
        tk.Label(
            tab1,
            text="Dataset Path: ",
            bg="#eeeeee",
            font=("Helvetica", 11, 'bold')
        ).place(
            x=35,
            y=85
        )

        ### Dataset Entry Field ###
        self.dataset_entry = tk.Entry(
            tab1,
            textvariable=self.dataset_url,
            bg="#fff",
            font=("Helvetica", 10),
            width=40
        ).place(
            x=140,
            y=87
        )

        ### Table Dataset ###
        TableDataset = Frame(tab1, width=500)
        TableDataset.place(x=30, y=150)
        TD_scrollbarx = Scrollbar(TableDataset, orient=HORIZONTAL)
        TD_scrollbary = Scrollbar(TableDataset, orient=VERTICAL)
        self.TD_tree = ttk.Treeview(TableDataset, columns=("No", "Teks Puisi"), height=23,
                                    selectmode="extended", yscrollcommand=TD_scrollbary.set, xscrollcommand=TD_scrollbarx.set)
        TD_scrollbary.config(command=self.TD_tree.yview)
        TD_scrollbary.pack(side=RIGHT, fill=Y)
        TD_scrollbarx.config(command=self.TD_tree.xview)
        TD_scrollbarx.pack(side=BOTTOM, fill=X)
        self.TD_tree.heading('No', text="No", anchor='center')
        self.TD_tree.heading('Teks Puisi', text="Teks Puisi", anchor='center')
        self.TD_tree.column('#0', stretch=NO, minwidth=0, width=0)
        self.TD_tree.column('#1', stretch=NO, minwidth=0,
                            width=40, anchor='center')
        self.TD_tree.column('#2', stretch=NO, minwidth=0, width=500)
        s = ttk.Style()
        s.configure('Treeview', rowheight=15)
        self.TD_tree.pack()

        ### Button Choose File ####
        self.select_btn1 = tk.Button(
            tab1,
            text="Pilih file...",
            command=lambda: self.select_dataset(),
            width=15,
        ).place(
            x=440,
            y=84
        )

        self.preprocessing_button = tk.Button(
            tab1,
            text="PRA-PENGOLAHAN",
            command=lambda: Preprocessing.execute_preprocess(
                self.TD_tree, self.dataset_url.get()),
            width=19,
            bg="#3498db",
            fg="#ffffff",
            bd=2,
            relief=tk.FLAT,
        ).place(
            x=150,
            y=550
        )

        self.tokenizing_button = tk.Button(
            tab1,
            text="TOKENISASI",
            command=lambda: DataPreparation.tokenizing(
                self.TD_tree, self.dataset_url.get()),
            width=19,
            bg="#3498db",
            fg="#ffffff",
            bd=2,
            relief=tk.FLAT,
        ).place(
            x=310,
            y=550
        )

        ### Konfigurasi Parameter Label ###
        self.konfigurasi_param_label = tk.Label(
            tab1,
            text="Konfigurasi Parameter",
            bg="#eeeeee",
            font=("Helvetica", 14, 'bold')
        ).place(
            x=670,
            y=80
        )

        ### Batch Size Label ###
        self.batch_size_label = tk.Label(
            tab1,
            text="Batch Size: ",
            bg="#eeeeee",
            font=("Helvetica", 11, 'bold')
        ).place(
            x=610,
            y=120
        )

        ### Batch Size Entry Field ###
        self.batch_size_entry = tk.Entry(
            tab1,
            textvariable=self.batch_size,
            bg="#fff",
            font=("Helvetica", 10),
            width=5
        ).place(
            x=700,
            y=122
        )

        ### Sequence Length Label ###
        self.sequence_length_label = tk.Label(
            tab1,
            text="Sequence Length: ",
            bg="#eeeeee",
            font=("Helvetica", 11, 'bold')
        ).place(
            x=760,
            y=120
        )

        ### Sequence Length Entry Field ###
        self.sequence_length_entry = tk.Entry(
            tab1,
            textvariable=self.sequence_length,
            bg="#fff",
            font=("Helvetica", 10),
            width=5
        ).place(
            x=900,
            y=122
        )

        ### Epoch Label ###
        self.epoch_label = tk.Label(
            tab1,
            text="Epoch: ",
            bg="#eeeeee",
            font=("Helvetica", 11, 'bold')
        ).place(
            x=610,
            y=160
        )

        ### Epoch Entry Field ###
        self.epoch_entry = tk.Entry(
            tab1,
            textvariable=self.epoch,
            bg="#fff",
            font=("Helvetica", 10),
            width=5
        ).place(
            x=700,
            y=162
        )

        ### Learning Rate Label ###
        self.learning_rate_label = tk.Label(
            tab1,
            text="Learning Rate: ",
            bg="#eeeeee",
            font=("Helvetica", 11, 'bold')
        ).place(
            x=760,
            y=160
        )

        ### Learning Rate Entry Field ###
        self.learning_rate_entry = tk.Entry(
            tab1,
            textvariable=self.learning_rate,
            bg="#fff",
            font=("Helvetica", 10),
            width=5
        ).place(
            x=900,
            y=162
        )

        ### Table Dataset ###
        TableAkurasi = Frame(tab1, width=100)
        TableAkurasi.place(x=615, y=300)
        TA_scrollbarx = Scrollbar(TableAkurasi, orient=HORIZONTAL)
        TA_scrollbary = Scrollbar(TableAkurasi, orient=VERTICAL)
        self.TA_tree = ttk.Treeview(TableAkurasi, columns=("Train_Loss", "Valid_Loss", "Perplexity"), height=2,
                                    selectmode="extended", yscrollcommand=TA_scrollbary.set, xscrollcommand=TA_scrollbarx.set)
        TA_scrollbary.config(command=self.TA_tree.yview)
        TA_scrollbary.pack(side=RIGHT, fill=Y)
        self.TA_tree.heading('Train_Loss', text="Train_Loss", anchor='center')
        self.TA_tree.heading('Valid_Loss', text="Valid_Loss", anchor='center')
        self.TA_tree.heading('Perplexity', text="Perplexity", anchor='center')
        self.TA_tree.column('#0', stretch=NO, minwidth=0, width=0)
        self.TA_tree.column('#1', stretch=NO, minwidth=0,
                            width=106, anchor='center')
        self.TA_tree.column('#2', stretch=NO, minwidth=0,
                            width=106, anchor='center')
        self.TA_tree.column('#3', stretch=NO, minwidth=0,
                            width=108, anchor='center')
        s = ttk.Style()
        s.configure('Treeview', rowheight=15)
        self.TA_tree.pack()

        ### Fine Tuning Button #
        self.fine_tuning_button = tk.Button(
            tab1,
            text="FINE TUNING MODEL",
            command=lambda: Training.start_training(self.dataset_url.get(), self.batch_size.get(
            ), self.sequence_length.get(), self.epoch.get(), self.learning_rate.get(), self.TA_tree),
            width=45,
            bg="#3498db",
            fg="#ffffff",
            bd=2,
            relief=tk.FLAT,
        ).place(
            x=615,
            y=210
        )

        ##### TAB 2 ######

        ### Title ###
        self.title = tk.Label(
            tab2,
            text="Generasi Teks Puisi Indonesia",
            bg="#eeeeee",
            font=("Helvetica", 21, 'bold')
        ).place(
            x=280,
            y=20
        )

        ### Load Model Label ###
        tk.Label(
            tab2,
            text="Muat Model: ",
            bg="#eeeeee",
            font=("Helvetica", 11, 'bold')
        ).place(
            x=300,
            y=80
        )

        ### Load Model Field ###
        self.model_entry = tk.Entry(
            tab2,
            textvariable=self.model_url,
            bg="#fff",
            font=("Helvetica", 10),
            width=27
        ).place(
            x=400,
            y=82
        )

        ### Button Choose File ####
        self.choose_model = tk.Button(
            tab2,
            text="Pilih model...",
            command=lambda: self.select_model(),
            width=15,
        ).place(
            x=600,
            y=78
        )

        ### Input Text Label ###
        tk.Label(
            tab2,
            text="Masukan Teks:",
            bg="#eeeeee",
            font=("Helvetica", 14, 'bold')
        ).place(
            x=450,
            y=120
        )

        ### Input Text Entry ###
        self.input_text_entry = tk.Entry(
            tab2,
            textvariable=self.input_text,
            bg="#fff",
            font=("Helvetica", 10),
            width=59,
        ).place(
            x=300,
            y=150
        )

        ### Generated Poem Label ###
        tk.Label(
            tab2,
            text="Puisi yang Dihasilkan:",
            bg="#eeeeee",
            font=("Helvetica", 14, 'bold')
        ).place(
            x=420,
            y=190
        )

        self.generated_poem_field = tk.Text(
            tab2,
            wrap=tk.WORD,
            bg="#fff",
            font=("Helvetica", 10),
            width=59
        )

        self.generated_poem_field.place(
            x=300,
            y=220,
            height=300
        )

        ### Generate Button #
        self.generate_button = tk.Button(
            tab2,
            text="GENERATE",
            command=lambda: self.generate_poem(self.model_url.get(),
                                               self.input_text.get()),
            width=15,
            bg="green",
            fg="#ffffff",
            bd=2,
            relief=tk.FLAT,
        ).place(
            x=370,
            y=540
        )

        ### Clear Button #
        self.clear_button = tk.Button(
            tab2,
            text="BERSIHKAN",
            command=lambda: self.clear_poem_field(),
            width=15,
            bg="red",
            fg="#ffffff",
            bd=2,
            relief=tk.FLAT,
        ).place(
            x=520,
            y=540
        )

    # select dataset
    def select_dataset(self):
        try:
            url = filedialog.askopenfilename(
                title="Pilih File", filetypes=(("CSV Files", "*.csv"),))
            self.dataset_url.set(url)
            Preprocessing.load_data(self.TD_tree, url)
        except Exception as e:
            messagebox.showerror(
                'Error', 'Error: Error when loading the dataset.')

    def select_model(self):
        try:
            url = filedialog.askopenfilename(
                title="Pilih Model", filetypes=(("Model File", "*.pkl"),))
            self.model_url.set(url)

        except Exception as e:
            messagebox.showerror(
                'Error', 'Error: Error ketika memuat model.')

    def generate_poem(self, url, input_text):
        poem = TextGeneration.generate_poem(url, input_text)
        self.generated_poem_field.delete(
            1.0, tk.END)  # Bersihkan teks yang ada
        self.generated_poem_field.insert(tk.END, poem)

    def clear_poem_field(self):
        self.input_text.set("")
        self.generated_poem_field.delete(
            1.0, tk.END)


def run():
    ROOT = tk.Tk()
    ROOT.geometry("1000x600")
    ROOT.resizable(height=False, width=False)
    MAIN_WINDOW = MainWindow(ROOT)
    ROOT.mainloop()


if __name__ == "__main__":
    run()
