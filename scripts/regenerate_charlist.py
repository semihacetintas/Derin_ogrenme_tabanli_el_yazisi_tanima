#!/usr/bin/env python3
"""Regenerate model/charList.txt from dataset (UTF-8 safe).

Usage: python scripts/regenerate_charlist.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUT = ROOT / 'model' / 'charList.txt'


def main():
    chars = set()
    gt = DATA_DIR / 'gt' / 'words.txt'
    if not gt.exists():
        print('Ground truth file not found:', gt)
        return 1

    with gt.open('r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 9:
                text = ' '.join(parts[8:])
            elif len(parts) >= 2:
                text = ' '.join(parts[1:])
            else:
                continue
            chars.update(text)

    char_list = ''.join(sorted(chars))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf8') as f:
        f.write(char_list)

    print('Wrote', OUT)
    print('Chars:', len(char_list), 'characters')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
