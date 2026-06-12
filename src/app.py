import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from segment import segment_words
from model import Model, DecoderType
from main import char_list_for_inference
from dataloader_iam import Batch
from preprocessor import Preprocessor
from spelling_corrector import SpellingCorrector

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB Max

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# HTR Modelini ve Yazım Düzelticiyi başlat (Singleton gibi bir kez yüklenir, çok hızlıdır)
print("HTR Modeli ve Yazım Düzeltici yükleniyor...")
CHAR_LIST = char_list_for_inference()
MODEL = Model(CHAR_LIST, DecoderType.BestPath, must_restore=True)
CORRECTOR = SpellingCorrector("../data/corpus.txt")
PREPROCESSOR = Preprocessor((128, 32), dynamic_width=True, padding=16)
print("Sistem başarıyla yüklendi! ✅")

def process_handwritten_image(img_path) -> dict:
    """Görseli segmentlere ayırır, HTR ile okur ve yazım denetiminden geçirir."""
    words = segment_words(img_path)
    if not words:
        return {
            "success": False,
            "message": "Görselde okunabilir kelime bulunamadı veya segmentasyon başarısız oldu.",
            "raw_text": "",
            "corrected_text": "",
            "words": []
        }
        
    raw_results = []
    corrected_results = []
    word_details = []
    
    for i, w in enumerate(words):
        # Resmi HTR preprocessor'dan geçir
        img = PREPROCESSOR.process_img(w)
        batch = Batch([img], None, 1)
        recognized, _ = MODEL.infer_batch(batch, calc_probability=False)
        
        text = recognized[0].strip()
        if text:
            corrected = CORRECTOR.correction(text)
            raw_results.append(text)
            corrected_results.append(corrected)
            
            # Kelime bazlı detaylar (Arayüzde eşleşme gösterebilmek için)
            word_details.append({
                "index": i,
                "raw": text,
                "corrected": corrected
            })
        else:
            word_details.append({
                "index": i,
                "raw": "",
                "corrected": ""
            })
            
    raw_sentence = " ".join(raw_results)
    corrected_sentence = " ".join(corrected_results)
    
    return {
        "success": True,
        "message": f"Başarıyla {len(raw_results)} kelime algılandı.",
        "raw_text": raw_sentence,
        "corrected_text": corrected_sentence,
        "words": word_details
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Dosya gönderilmedi."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Dosya seçilmedi."}), 400
        
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Resmi işle
        result = process_handwritten_image(file_path)
        return jsonify(result)

@app.route('/capture', methods=['POST'])
def capture_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "Görüntü verisi gönderilmedi."}), 400
        
    try:
        # Base64 verisini decode et
        image_data = data['image'].split(',')[1]
        decoded_data = base64.b64decode(image_data)
        
        filename = "capture.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(file_path, "wb") as f:
            f.write(decoded_data)
            
        # Resmi işle
        result = process_handwritten_image(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Kamera görüntüsü işlenirken hata oluştu: {str(e)}"}), 500

if __name__ == '__main__':
    # Flask sunucusunu yerel ağda da erişilebilir şekilde başlat
    app.run(host='0.0.0.0', port=5000, debug=False)
