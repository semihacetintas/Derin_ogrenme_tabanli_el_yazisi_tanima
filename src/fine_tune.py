import os
import random
import cv2
import numpy as np
from typing import List, Tuple
from collections import namedtuple

from model import Model, DecoderType
from main import char_list_from_file, get_img_size
from preprocessor import Preprocessor
from dataloader_iam import Batch, Sample

def load_dataset(data_dir: str) -> Tuple[List[Sample], List[Sample]]:
    """Türkçe sentetik görselleri ve İngilizce veri setinden dengeli bir alt kümeyi yükler."""
    gt_path = os.path.join(data_dir, "gt", "words.txt")
    
    tr_samples = []
    en_samples = []
    
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                rel_path = parts[0]
                gt_text = " ".join(parts[1:])
            else:
                continue
                
            full_path = os.path.join(data_dir, rel_path)
            if not os.path.exists(full_path):
                continue
                
            sample = Sample(gt_text, full_path)
            if "generated/tr_" in rel_path:
                tr_samples.append(sample)
            else:
                en_samples.append(sample)
                
    print(f"Bulunan Türkçe sentetik örnek sayısı: {len(tr_samples)}")
    print(f"Bulunan İngilizce örnek sayısı: {len(en_samples)}")
    
    # Türkçe örnekleri daha yüksek frekansta eğitmek için 4x çoğaltıyoruz (over-sampling)
    augmented_tr_samples = tr_samples * 4
    print(f"Çoğaltılmış Türkçe örnek sayısı (4x): {len(augmented_tr_samples)}")
    
    # İngilizce örneklerden rastgele 3000 adet seçiyoruz
    random.seed(42)
    selected_en_samples = random.sample(en_samples, min(3000, len(en_samples)))
    print(f"Eğitim için seçilen İngilizce örnek sayısı: {len(selected_en_samples)}")
    
    train_samples = augmented_tr_samples + selected_en_samples
    random.shuffle(train_samples)
    
    # %90 train, %10 validation
    split_idx = int(0.9 * len(train_samples))
    return train_samples[:split_idx], train_samples[split_idx:]

def get_img(sample: Sample) -> np.ndarray:
    """Görseli diskten okur ve gri tonlamaya çevirir."""
    try:
        with open(sample.file_path, "rb") as f:
            img_data = f.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None
        
    if img is None:
        img = np.zeros((32, 128), dtype=np.uint8)
        
    img = cv2.resize(img, (128, 32))
    return img.astype(np.float32)

class FineTuneDataset:
    def __init__(self, samples: List[Sample], batch_size: int, data_augmentation: bool = True):
        self.samples = samples
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.curr_idx = 0
        
    def shuffle(self):
        random.shuffle(self.samples)
        self.curr_idx = 0
        
    def has_next(self) -> bool:
        return self.curr_idx + self.batch_size <= len(self.samples)
        
    def get_next(self) -> Batch:
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))
        imgs = [get_img(self.samples[i]) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))

def main():
    data_dir = "../data"
    batch_size = 50
    epochs = 4
    
    # Karakter listesini yükle (mevcut snapshot ile tamamen uyumlu)
    char_list = char_list_from_file()
    print(f"Karakter listesi yüklendi. Toplam sınıf sayısı: {len(char_list)}")
    
    # Modelin yüklenmesi (must_restore=True sayesinde eski snapshot ağırlıkları korunur)
    print("Pre-trained model yükleniyor...")
    model = Model(char_list, DecoderType.BestPath, must_restore=True)
    
    # Veri setini oluştur
    train_samples, val_samples = load_dataset(data_dir)
    print(f"Toplam Eğitim Görseli: {len(train_samples)}")
    print(f"Toplam Doğrulama Görseli: {len(val_samples)}")
    
    train_dataset = FineTuneDataset(train_samples, batch_size, data_augmentation=True)
    val_dataset = FineTuneDataset(val_samples, batch_size, data_augmentation=False)
    
    preprocessor_train = Preprocessor(get_img_size(), data_augmentation=True)
    preprocessor_val = Preprocessor(get_img_size(), data_augmentation=False)
    
    print("\nFine-tuning başlıyor (CPU üzerinde hızlı eğitim)...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch} / {epochs} ---")
        train_dataset.shuffle()
        
        batch_idx = 0
        total_loss = 0
        
        while train_dataset.has_next():
            batch = train_dataset.get_next()
            batch = preprocessor_train.process_batch(batch)
            loss = model.train_batch(batch)
            total_loss += loss
            batch_idx += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch: {batch_idx} | Ortalama Kayıp (Loss): {loss:.4f}")
                
        avg_train_loss = total_loss / batch_idx
        print(f"Epoch {epoch} Tamamlandı. Ortalama Eğitim Kaybı: {avg_train_loss:.4f}")
        
        # Validation
        print("Doğrulama (Validation) yapılıyor...")
        val_dataset.shuffle()
        val_loss = 0
        val_batches = 0
        
        num_ok = 0
        num_total = 0
        
        while val_dataset.has_next():
            batch = val_dataset.get_next()
            batch_processed = preprocessor_val.process_batch(batch)
            recognized, _ = model.infer_batch(batch_processed, calc_probability=False)
            
            for i in range(len(recognized)):
                num_total += 1
                if recognized[i] == batch.gt_texts[i]:
                    num_ok += 1
            val_batches += 1
            
        val_acc = (num_ok / num_total) * 100 if num_total > 0 else 0
        print(f"Doğrulama Doğruluğu (Word Acc): %{val_acc:.2f} ({num_ok}/{num_total})")
        
        # Modeli kaydet
        print("Model kaydediliyor...")
        model.save()
        
    print("\nEğitim başarıyla tamamlandı ve yeni model snapshot-24 olarak kaydedildi! ✅")

if __name__ == "__main__":
    main()
