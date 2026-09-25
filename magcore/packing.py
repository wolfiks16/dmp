from __future__ import annotations

import base64

import numpy as np

# Массив ↔ текст для браузера и файла расчёта (2D и 3D): base64, порядок байтов little-endian. Числа
# передаются без округления — по ним файл расчёта сверяется с текущей версией решателя (Л-80).


def pack(values, dtype) -> str:
    """Массив → base64, порядок байтов little-endian (передача в браузер и в файл расчёта)."""
    a = np.ascontiguousarray(np.asarray(values), dtype=np.dtype(dtype).newbyteorder("<"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def unpack(text: str, dtype) -> np.ndarray:
    """Обратное к `pack` (плоский массив)."""
    return np.frombuffer(base64.b64decode(text), dtype=np.dtype(dtype).newbyteorder("<"))
