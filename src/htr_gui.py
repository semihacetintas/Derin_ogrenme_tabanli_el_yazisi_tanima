import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2

# Dosya yollarında sorun olmaması için bu scriptin bulunduğu klasöre geç.
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

from segment import segment_words
from model import Model, DecoderType
from main import char_list_for_inference
from dataloader_iam import Batch
from preprocessor import Preprocessor
from spelling_corrector import SpellingCorrector


class HTRGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('El Yazısı Tanıma')
        self.root.geometry('760x520')
        self.root.resizable(False, False)

        self.char_list = char_list_for_inference()
        self.model = None
        self.corrector = SpellingCorrector('../data/corpus.txt')
        self.preprocessor = Preprocessor((128, 32), dynamic_width=True, padding=16)

        self._build_ui()

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=16, pady=16)
        frame.pack(fill='both', expand=True)

        title = tk.Label(frame, text='El Yazısı Tanıma', font=('Segoe UI', 20, 'bold'))
        title.pack(anchor='w')

        info = tk.Label(frame, text='Bir resim seçin, sistem kelime segmentasyonunu yapacak ve metni tanıyacaktır.',
                        font=('Segoe UI', 11), wraplength=720, justify='left')
        info.pack(anchor='w', pady=(10, 16))

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill='x', pady=(0, 16))

        self.path_label = tk.Label(btn_frame, text='Henüz resim seçilmedi.', font=('Segoe UI', 10), anchor='w')
        self.path_label.pack(side='left', fill='x', expand=True)

        browse_btn = tk.Button(btn_frame, text='Resim Seç', width=14, command=self.choose_file)
        browse_btn.pack(side='right')

        self.result_text = tk.Text(frame, wrap='word', height=18, font=('Segoe UI', 10))
        self.result_text.pack(fill='both', expand=True)
        self.result_text.configure(state='disabled')

        self.status_label = tk.Label(frame, text='', font=('Segoe UI', 10, 'italic'), fg='#333333', anchor='w')
        self.status_label.pack(fill='x', pady=(10, 0))

    def choose_file(self):
        filename = filedialog.askopenfilename(
            title='El Yazısı Resmi Seçin',
            filetypes=[('Resimler', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff')]
        )
        if not filename:
            return

        self.path_label.config(text=filename)
        self.status_label.config(text='Model yükleniyor...')
        self.root.update_idletasks()

        try:
            self._ensure_model_loaded()
        except Exception as exc:
            self.status_label.config(text='Model yüklenemedi.')
            messagebox.showerror('Model Hatası', str(exc))
            return

        self.status_label.config(text='Resim işleniyor...')
        self.root.update_idletasks()

        try:
            raw, corrected = self.process_file(filename)
            self._show_result(raw, corrected)
            self.status_label.config(text='Tamamlandı.')
        except Exception as exc:
            self.status_label.config(text='İşleme sırasında hata oluştu.')
            messagebox.showerror('İşlem Hatası', str(exc))

    def _ensure_model_loaded(self):
        if self.model is not None:
            return
        self.model = Model(self.char_list, DecoderType.BestPath, must_restore=True)

    def process_file(self, image_path):
        print("SEÇİLEN DOSYA =", image_path)

        words = segment_words(image_path)
        if not words:
            raise ValueError('Görselde okunabilir kelime bulunamadı.')

        recognized_words = []
        corrected_words = []
        for word_img in words:
            processed_img = self.preprocessor.process_img(word_img)
            batch = Batch([processed_img], None, 1)
            recognized, _ = self.model.infer_batch(batch, calc_probability=False)
            text = recognized[0].strip()
            corrected = self.corrector.correction(text) if text else ''
            recognized_words.append(text)
            corrected_words.append(corrected)

        raw_sentence = ' '.join([w for w in recognized_words if w])
        corrected_sentence = ' '.join([w for w in corrected_words if w])
        return raw_sentence, corrected_sentence

    def _show_result(self, raw_text, corrected_text):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, 'Ham Metin:\n')
        self.result_text.insert(tk.END, raw_text or '(boş)')
        self.result_text.insert(tk.END, '\n\nDüzeltilmiş Metin:\n')
        self.result_text.insert(tk.END, corrected_text or '(boş)')
        self.result_text.configure(state='disabled')

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = HTRGui()
    app.run()
