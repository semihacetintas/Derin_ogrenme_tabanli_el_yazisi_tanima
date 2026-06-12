import argparse
import json
import os
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(average_train_loss, char_error_rates, word_accuracies):
    with open(FilePaths.fn_summary, 'w', encoding='utf-8') as f:
        json.dump({
            'averageTrainLoss': average_train_loss,
            'charErrorRates': char_error_rates,
            'wordAccuracies': word_accuracies
        }, f)


# ✅ FIXED (EN KRİTİK KISIM)
def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Tek satır formatı (SimpleHTR varsayılanı)
    if '\n' not in content.strip():
        return list(content)
    return [line for line in content.splitlines() if line]


def char_list_for_inference() -> List[str]:
    """Model eğitiminde kullanılan karakter listesi (charList.txt)."""
    return char_list_from_file()


def char_list_from_dataset() -> List[str]:
    """Eğitimde kullanılan karakter kümesini ground truth dosyasından oluştur."""
    gt_path = Path('../data/gt/words.txt')
    if not gt_path.exists():
        return char_list_from_file()

    chars = set()
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            parts = line.split()
            if len(parts) >= 9:
                text = ' '.join(parts[8:])
            elif len(parts) >= 2:
                text = ' '.join(parts[1:])
            else:
                continue
            chars.update(text)

    return sorted(list(chars))


def train(model, loader, line_mode, early_stopping=25):
    epoch = 0
    summary_char_error_rates = []
    summary_word_accuracies = []
    train_loss_in_epoch = []
    average_train_loss = []

    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)

    best_char_error_rate = float('inf')
    no_improvement_since = 0

    while True:
        epoch += 1
        print('Epoch:', epoch)

        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)

            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
            train_loss_in_epoch.append(loss)

        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        average_train_loss.append(sum(train_loss_in_epoch) / len(train_loss_in_epoch))

        write_summary(average_train_loss, summary_char_error_rates, summary_word_accuracies)

        train_loss_in_epoch = []

        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'No improvement, best: {best_char_error_rate * 100:.2f}%')
            no_improvement_since += 1

        if no_improvement_since >= early_stopping:
            print(f'Stopped after {early_stopping} epochs without improvement')
            break


def validate(model, loader, line_mode):
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)

    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0

    while loader.has_next():
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        for i in range(len(recognized)):
            num_word_ok += (batch.gt_texts[i] == recognized[i])
            num_word_total += 1

            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])

            print('[OK]' if dist == 0 else f'[ERR:{dist}]',
                  f'"{batch.gt_texts[i]}" -> "{recognized[i]}"')

    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total

    print(f'CER: {char_error_rate*100:.2f}% | Word Acc: {word_accuracy*100:.2f}%')
    return char_error_rate, word_accuracy


def infer(model, fn_img):
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, calc_probability=False)

    print(f'Result: "{recognized[0]}"')
    if probability is not None:
        print(f'Prob: {probability[0]}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', type=int, default=50)  # ✅ daha stabil
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--line_mode', action='store_true')
    parser.add_argument('--img_file', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', type=int, default=25)
    parser.add_argument('--dump', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    decoder_mapping = {
        'bestpath': DecoderType.BestPath,
        'beamsearch': DecoderType.BeamSearch,
        'wordbeamsearch': DecoderType.WordBeamSearch
    }

    decoder_type = decoder_mapping[args.decoder]

    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=False)

        char_list = loader.char_list

        # ✅ HER TRAIN'DE YENİDEN YAZ (EN TEMİZ)
        with open(FilePaths.fn_char_list, 'w', encoding='utf-8') as f:
            f.write(''.join(char_list))
        with open(FilePaths.fn_corpus, 'w', encoding='utf-8') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, args.line_mode, args.early_stopping)

    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(loader.char_list, decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)

    elif args.mode == 'infer':
        model = Model(char_list_for_inference(), decoder_type, must_restore=True, dump=args.dump)
        infer(model, args.img_file)


if __name__ == '__main__':
    main()
