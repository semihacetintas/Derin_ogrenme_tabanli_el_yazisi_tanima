import argparse
import os

import cv2

from segment import segment_words
from model import Model, DecoderType
from main import char_list_for_inference
from dataloader_iam import Batch
from preprocessor import Preprocessor
from debug_log import agent_log
from spelling_corrector import SpellingCorrector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, help='Okunacak cümle resmi')
    args = parser.parse_args()

    img_path = args.img_file or input("Resim yolunu gir: ")

    char_list = char_list_for_inference()
    # #region agent log
    agent_log('read_sentence.py', 'char_list_loaded', {
        'char_count': len(char_list),
        'preview': ''.join(char_list[:20]),
    }, 'F')
    # #endregion

    model = Model(
        char_list,
        DecoderType.BestPath,
        must_restore=True
    )

    # Akıllı Yazım Düzelticiyi yükle
    corrector = SpellingCorrector("../data/corpus.txt")

    words = segment_words(img_path)
    print("Bulunan kelime sayısı:", len(words))

    os.makedirs("output", exist_ok=True)
    results = []
    preprocessor = Preprocessor((128, 32), dynamic_width=True, padding=16)

    for i, w in enumerate(words):
        cv2.imwrite(f"output/word_{i}.png", w)
        img = preprocessor.process_img(w)
        batch = Batch([img], None, 1)
        recognized, _ = model.infer_batch(batch, calc_probability=False)

        text = recognized[0].strip()
        # #region agent log
        agent_log('read_sentence.py', 'word_recognized', {
            'index': i,
            'text': text,
            'raw': recognized[0],
        }, 'C')
        # #endregion

        if text:
            corrected_text = corrector.correction(text)
            print(f"Kelime {i}: {text} -> {corrected_text}")
            results.append(corrected_text)
        else:
            print(f"Kelime {i}: (bos/atlandi)")

    final_sentence = " ".join(results)
    print("\nSONUC:")
    print(final_sentence)

    with open("output/result.txt", "w", encoding="utf-8") as f:
        f.write(final_sentence)


if __name__ == '__main__':
    main()
