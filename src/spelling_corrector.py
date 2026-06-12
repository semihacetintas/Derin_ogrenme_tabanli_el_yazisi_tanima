import os
import re
from collections import Counter

class SpellingCorrector:
    def __init__(self, corpus_path: str = None):
        self.WORDS = Counter()
        
        # 1. Varsayılan çok sık kullanılan İngilizce ve Türkçe kelimeleri ekle (Soğuk başlangıç)
        common_words = {
            # İngilizce en yaygın kelimeler
            "you": 100000, "are": 90000, "coming": 80000, "the": 200000, "and": 150000, "to": 140000, 
            "is": 120000, "in": 110000, "it": 100000, "that": 90000, "he": 80000, "was": 75000, 
            "for": 70000, "on": 65000, "are": 60000, "as": 55000, "with": 50000, "his": 48000,
            "they": 45000, "i": 42000, "at": 40000, "be": 38000, "this": 36000, "have": 34000, 
            "from": 32000, "one": 30000, "had": 28000, "by": 26000, "word": 24000, "but": 22000,
            "not": 20000, "what": 18000, "all": 16000, "were": 15000, "we": 14000, "when": 13000,
            # Türkçe en yaygın kelimeler (Türkçe karakterlerin korunması ve düzeltilmesi için)
            "çiçek": 1500, "çanta": 1200, "çorba": 1100, "çalışma": 2000, "çizgi": 1000, 
            "şeker": 1400, "şapka": 900, "şemsiye": 800, "şirket": 1600, "şarkı": 1300, 
            "öğrenci": 2500, "ödev": 1800, "öykü": 1200, "öğretmen": 2200, "gökyüzü": 1100, 
            "gözlük": 950, "üzüm": 1050, "ülke": 1900, "ücret": 1400, "ağaç": 1350, 
            "yağmur": 1250, "güneş": 1700, "ışık": 1450, "soğuk": 1300, "yoğurt": 900, 
            "düğün": 1150, "büyük": 2200, "küçük": 2000, "merhaba": 3000, "nasılsın": 2500, 
            "geliyor": 4000, "musun": 3500, "mi": 5000, "mu": 4500, "evet": 4800, "hayır": 4200, 
            "ve": 10000, "bir": 12000, "bu": 9000, "ne": 8000, "nasıl": 6000, "neden": 5000, 
            "kim": 4000, "nerede": 3500, "tamam": 3800, "hazır": 2200, "bitti": 2500, "gitti": 2400
        }
        self.WORDS.update(common_words)

        # 2. Eğer corpus.txt varsa, oradaki kelimeleri de yükle
        if corpus_path and os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
                    words = re.findall(r'[a-zA-ZçöüğışÇÖÜĞİŞı]+', f.read().lower())
                    self.WORDS.update(words)
            except Exception as e:
                print(f"Corpus yüklenirken hata oluştu: {e}")

    def P(self, word): 
        """Kelimenin frekansa dayalı olasılık değeri."""
        return self.WORDS[word] / sum(self.WORDS.values())

    def correction(self, word): 
        """Kelimenin en olası doğru halini döndürür."""
        if not word:
            return ""
            
        # Kelimenin başındaki ve sonundaki noktalama işaretlerini koru
        match_start = re.match(r'^([^a-zA-ZçöüğışÇÖÜĞİŞı]+)', word)
        match_end = re.search(r'([^a-zA-ZçöüğışÇÖÜĞİŞı]+)$', word)
        
        prefix = match_start.group(1) if match_start else ""
        suffix = match_end.group(1) if match_end else ""
        
        # Temiz kelimeyi elde et
        clean_word = word
        if prefix:
            clean_word = clean_word[len(prefix):]
        if suffix:
            clean_word = clean_word[:-len(suffix)]
            
        if not clean_word:
            return word

        # Küçük harfle arama yap (ancak orijinal case'i korumaya çalış)
        is_upper = clean_word.isupper()
        is_title = clean_word.istitle()
        
        low_word = clean_word.lower()
        low_corrected = max(self._candidates(low_word), key=self.P)
        
        # Case restorasyonu
        if is_upper:
            corrected = low_corrected.upper()
        elif is_title:
            corrected = low_corrected.capitalize()
        else:
            corrected = low_corrected
            
        return prefix + corrected + suffix

    def _candidates(self, word): 
        """Düzeltme aday kümesini döndürür."""
        return (self._known([word]) or self._known(self._edits1(word)) or self._known(self._edits2(word)) or [word])

    def _known(self, words): 
        """Sözlükte tanımlı olan kelimelerin alt kümesini döndürür."""
        return set(w for w in words if w in self.WORDS)

    def _edits1(self, word):
        """Kelimeye 1 edit uzaklığındaki tüm kelimeleri üretir."""
        letters    = 'abcdefghijklmnopqrstuvwxyzçöüğış'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word): 
        """Kelimeye 2 edit uzaklığındaki tüm kelimeleri üretir."""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

if __name__ == "__main__":
    # Testler
    corrector = SpellingCorrector("../data/corpus.txt")
    print("yoe ->", corrector.correction("yoe"))
    print("are yoe coming? ->", " ".join(corrector.correction(w) for w in ["are", "yoe", "coming?"]))
    print("sizek ->", corrector.correction("sizek"))
    print("bgrenci ->", corrector.correction("bgrenci"))
    print("seker? ->", corrector.correction("seker?"))
