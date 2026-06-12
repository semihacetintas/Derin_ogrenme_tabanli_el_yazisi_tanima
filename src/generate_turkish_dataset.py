import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

def main():
    # Türkçe kelimeler (sadece küçük karakterli, soru işaretli)
    turkce_kelimeler = [
        "çiçek", "çanta", "çorba", "çalışma", "çizgi", "çilek", "çamur", "çekiç", "çene", "çığ", "çizme",
        "şeker", "şapka", "şemsiye", "şirket", "şarkı", "şair", "şerit", "şişe", "şans", "şelale",
        "öğrenci", "ödev", "öykü", "öğretmen", "öfke", "ömür", "ördek", "örnek", "özlem", "önem",
        "gökyüzü", "gözlük", "görev", "gölge", "görüş", "gömlek", "göç", "gövde",
        "üzüm", "ülke", "ücret", "üretim", "ümit", "üniversite", "üye", "ürün", "üçgen", "üzeri",
        "ağaç", "yağmur", "güneş", "ışık", "soğuk", "yoğurt", "düğün", "kağıt", "sağlık", "doğal",
        "büyük", "küçük", "ılık", "ırmak", "ıspanak", "ısı", "ıslak", "ıstırap", "ıhlamur",
        "türkçe", "türkçe", "türkiye", "istanbul", "ankara", "izmir", "mustafa", "semiha",
        # Soru işaretli kelimeler ve cümle sonları
        "coming?", "you?", "are?", "here?", "why?", "what?", "when?", "how?", "who?", "this?",
        "geliyor?", "misin?", "mi?", "miyiz?", "neden?", "nasıl?", "kim?", "nerede?", "ne?",
        "çiçek?", "şeker?", "üzüm?", "öğrenci?", "ağaç?", "yoğurt?", "düğün?", "büyük?", "küçük?",
        "burada?", "tamam?", "hazır?", "oldu?", "bitti?", "gitti?", "miyim?", "şimdi?", "orada?"
    ]

    # Ekstra rastgele kelime kombinasyonları ve soru işareti testleri
    extra_words = [
        "are?", "you?", "coming?", "yoe?", "yes?", "no?", "ok?", "fine?", "hello?", "help?",
        "ç?", "ş?", "ğ?", "ü?", "ö?", "ı?", "ist?", "tr?"
    ]
    
    all_words = turkce_kelimeler * 15 + extra_words * 20
    random.shuffle(all_words)
    all_words = all_words[:2000] # Tam olarak 2000 görsel üreteceğiz

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    generated_dir = os.path.join(data_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    gt_file_path = os.path.join(data_dir, "gt", "words.txt")

    # Windows standart font yolları
    font_paths = [
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/calibrib.ttf",  # Calibri Bold
        "C:/Windows/Fonts/calibrii.ttf",  # Calibri Italic
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
        "C:/Windows/Fonts/ariali.ttf",    # Arial Italic
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",   # Times Bold
        "C:/Windows/Fonts/timesi.ttf",    # Times Italic
        "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/verdanab.ttf",  # Verdana Bold
        "C:/Windows/Fonts/verdanai.ttf",  # Verdana Italic
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/georgiab.ttf",  # Georgia Bold
        "C:/Windows/Fonts/georgiai.ttf",  # Georgia Italic
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/comicbd.ttf"    # Comic Sans Bold
    ]

    # Sadece var olan fontları filtrele
    valid_font_paths = [p for p in font_paths if os.path.exists(p)]
    if not valid_font_paths:
        print("Sistem fontları bulunamadı, varsayılan font kullanılacak!")
        valid_font_paths = [None]

    new_entries = []

    print(f"Sentetik veriler üretiliyor. Toplam kelime sayısı: {len(all_words)}")

    for i, word in enumerate(all_words):
        # 128x32 boyutlarında, modelin beklediği formatta (siyah yazı, beyaz arka plan)
        img = Image.new("L", (180, 48), color=255) # L = Grayscale (0-255)
        draw = ImageDraw.Draw(img)

        # Rastgele font seçimi
        font_path = random.choice(valid_font_paths)
        font_size = random.randint(22, 28)
        
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Metni merkeze hizala
        try:
            # Get text bounding box to center it
            bbox = draw.textbbox((0, 0), word, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = 100, 20
            
        x = max(2, (180 - text_w) // 2 + random.randint(-10, 10))
        y = max(2, (48 - text_h) // 2 - 4 + random.randint(-4, 4))

        # Metni çiz (Siyah renk, arka plan beyaz)
        draw.text((x, y), word, fill=0, font=font)

        # Hafif döndürme ve gürültü ekleme (El yazısı hissi vermek için)
        if random.random() > 0.3:
            # Rastgele -5 ile 5 derece arası döndürme
            angle = random.uniform(-4, 4)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

        # Resmi 128x32 formatına getir
        img = img.resize((128, 32), Image.Resampling.LANCZOS)

        file_name = f"tr_{i}.png"
        full_path = os.path.join(generated_dir, file_name)
        img.save(full_path)

        # gt/words.txt formatında entry hazırla: relative_path ground_truth
        new_entries.append(f"generated/{file_name} {word}")

    # words.txt dosyasını güncelle
    print("words.txt güncelleniyor...")
    existing_lines = []
    if os.path.exists(gt_file_path):
        with open(gt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Daha önce eklenmiş tr_*.png girdileri varsa temizle (idempotent yapmak için)
                if "generated/tr_" not in line:
                    existing_lines.append(line.rstrip())

    # Yeni girdileri ekle
    all_lines = existing_lines + new_entries

    with open(gt_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    print("Basariyla sentetik gorseller uretildi ve words.txt dosyasina eklendi.")

if __name__ == "__main__":
    main()
