from pathlib import Path

p = Path('model/charList.txt')
raw = p.read_text(encoding='utf8')
print(repr(raw))
for ch in 'çğıöşüÇĞİÖŞÜ':
    print(f'{ch}: {ch in raw}')
