import os
import cv2
from segment import segment_words

img_path = input("Resim yolunu gir: ")

words = segment_words(img_path)

output_dir = "output"

# klasörü temizle
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
else:
    os.makedirs(output_dir)

# yeni kelimeleri kaydet
for i, w in enumerate(words):
    cv2.imwrite(f"{output_dir}/word_{i}.png", w)

print("Output güncellendi!")