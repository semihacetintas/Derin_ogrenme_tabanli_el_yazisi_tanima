from PIL import Image, ImageDraw, ImageFont
import os
import random

kelimeler = [
"çiçek","çanta","çorba","çalışma","çizgi",
"şeker","şapka","şemsiye","şirket","şarkı",
"öğrenci","ödev","öykü","öğretmen",
"gökyüzü","gözlük",
"üzüm","ülke","ücret",
"ağaç","yağmur","güneş","ışık",
"soğuk","yoğurt","düğün",
"büyük","küçük"
]

os.makedirs("generated", exist_ok=True)

font = ImageFont.truetype("C:/Windows/Fonts/calibri.ttf", 24)

for i, kelime in enumerate(kelimeler):
    img = Image.new("RGB", (200, 50), color="white")
    draw = ImageDraw.Draw(img)

    x = random.randint(5,20)
    y = random.randint(5,20)

    draw.text((x,y), kelime, fill="black", font=font)

    img.save(f"generated/{i}.png")

print("Bitti ✅")