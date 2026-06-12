import albumentations as A
import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:

    def __init__(self, data_dir: Path, batch_size: int, data_split: float = 0.9, fast: bool = False) -> None:

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        # AUGMENTATION
        self.augment = A.Compose([
            A.Rotate(limit=3, p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2)
        ])

        f = open(data_dir / 'gt/words.txt', encoding="utf8")

        chars = set()

        for line in f:
            line = line.strip()

            if not line or line[0] == '#':
                continue

            line_split = line.split()

            if len(line_split) >= 9:
                img_id = line_split[0]
                img_split = img_id.split('-')

                subdir1 = img_split[0]
                subdir2 = img_split[0] + '-' + img_split[1]

                file_name = data_dir / 'words' / subdir1 / subdir2 / (img_id + '.png')
                gt_text = ' '.join(line_split[8:])

            elif len(line_split) >= 2:
                rel_path = line_split[0]
                gt_text = ' '.join(line_split[1:])
                file_name = data_dir / rel_path

            else:
                continue

            if not file_name.exists():
                continue

            chars = chars.union(set(list(gt_text)))
            self.samples.append(Sample(gt_text, file_name))

        print("Total samples:", len(self.samples))

        split_idx = int(data_split * len(self.samples))

        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        self.train_set()

        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))

        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)
        else:
            return self.curr_idx < len(self.samples)

    def _get_img(self, i: int) -> np.ndarray:

        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            file_path = self.samples[i].file_path
            try:
                with open(str(file_path), 'rb') as f:
                    img_data = f.read()
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = None

        # AUGMENTATION
        if self.data_augmentation and img is not None:
            img = self.augment(image=img)["image"]

        if img is None:
            img = np.zeros((32, 128), dtype=np.uint8)

        # RESIZE
        img = cv2.resize(img, (128, 32))

        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        img = img.astype(np.float32)

        return img

    def get_next(self) -> Batch:

        batch_range = range(self.curr_idx,
                            min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size

        return Batch(imgs, gt_texts, len(imgs))