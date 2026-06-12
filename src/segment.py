import cv2
import numpy as np

from typing import Union

from debug_log import agent_log


def prepare_word(word_img: np.ndarray) -> Union[np.ndarray, None]:
    """Kelimeyi IAM/SimpleHTR formatına getir: sıkı kırp, polariteyi düzelt."""
    if word_img is None or word_img.size == 0:
        return None

    _, binary = cv2.threshold(
        word_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    pad = 8
    y1 = max(0, y - pad)
    y2 = min(word_img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(word_img.shape[1], x + w + pad)
    crop = word_img[y1:y2, x1:x2].copy()

    # Model siyah yazı + beyaz arka plan bekler
    if float(np.mean(crop)) < 127:
        crop = 255 - crop

    if crop.shape[1] < 25 or crop.shape[0] < 10:
        return None

    _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(ink) < 50:
        return None

    # #region agent log
    agent_log('segment.py:prepare_word', 'prepared_word', {
        'shape': list(crop.shape),
        'mean': float(np.mean(crop)),
        'is_dark_background': bool(np.mean(crop) < 127),
        'ink_pixels': int(cv2.countNonZero(ink)),
    }, 'A')
    # #endregion

    return crop


def segment_words(img_path):

    try:
        file_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    except Exception:
        original = None

    if original is None:
        print("Resim okunamadı!")
        agent_log(
            'segment.py:segment_words',
            'image_read_failed',
            {'img_path': str(img_path)},
            'E'
        )
        return []

    agent_log('segment.py:segment_words', 'source_image_stats', {
        'img_path': str(img_path),
        'shape': list(original.shape),
        'mean': float(np.mean(original)),
        'std': float(np.std(original)),
    }, 'A')

    img = cv2.GaussianBlur(original, (5, 5), 0)

    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 60), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    projection = np.sum(thresh, axis=0)

    words = []
    start = None

    for i, val in enumerate(projection):

        if val > 0 and start is None:
            start = i

        elif val == 0 and start is not None:

            end = i

            if end - start > 20:
                prepared = prepare_word(original[:, start:end])

                if prepared is not None:

                    agent_log('segment.py:segment_words', 'segment_slice', {
                        'index': len(words),
                        'start': int(start),
                        'end': int(end),
                        'width': int(prepared.shape[1]),
                        'height': int(prepared.shape[0]),
                        'mean': float(np.mean(prepared)),
                    }, 'B')

                    words.append(prepared)

            start = None

    if start is not None:

        prepared = prepare_word(original[:, start:])

        if prepared is not None:

            agent_log('segment.py:segment_words', 'segment_slice', {
                'index': len(words),
                'start': int(start),
                'end': int(original.shape[1]),
                'width': int(prepared.shape[1]),
                'height': int(prepared.shape[0]),
                'mean': float(np.mean(prepared)),
            }, 'B')

            words.append(prepared)

    print("Bulunan kelime sayısı:", len(words))

    agent_log(
        'segment.py:segment_words',
        'segment_complete',
        {'word_count': len(words)},
        'D'
    )

    return words