import re
from pathlib import Path

import numpy as np
import pytest

from magcore.post.palette import field_rainbow

# Палитра поля в проекте одна — радуга (решение Sergey 2026-09-23): экран (webapp) и рисунки
# (pilot, docs/figures) должны красить поле одинаково. Формула в интерфейсе на JS, для рисунков — на
# Python; тест держит обе копии вместе.

INDEX_HTML = Path(__file__).resolve().parents[2] / "webapp" / "static" / "index.html"


@pytest.mark.parametrize("t, rgb", [
    (0.0, (0.0, 0.0, 0.5)),        # тёмно-синий — ноль шкалы (в интерфейсе rgb(0, 0, 127))
    (0.25, (0.0, 0.5, 1.0)),
    (0.5, (0.5, 1.0, 0.5)),
    (0.75, (1.0, 0.5, 0.0)),
    (1.0, (0.5, 0.0, 0.0)),        # тёмно-красный — верх шкалы
])
def test_reference_points(t, rgb):
    assert np.allclose(field_rainbow(t), rgb, atol=1e-15)


def test_values_outside_the_scale_are_clipped_and_shapes_follow_input():
    assert np.array_equal(field_rainbow(-3.0), field_rainbow(0.0))
    assert np.array_equal(field_rainbow(7.0), field_rainbow(1.0))
    assert field_rainbow(np.zeros((4, 5))).shape == (4, 5, 3)
    t = np.linspace(0.0, 1.0, 1001)
    c = field_rainbow(t)
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_same_formula_as_the_interface():
    """Функция `rainbow` в index.html — та же формула: 1,5 − |4t − k| для R, G, B при k = 3, 2, 1."""
    src = INDEX_HTML.read_text(encoding="utf-8")
    m = re.search(r"function rainbow\(t\)\{(.*?)\}", src)
    assert m, "в index.html нет функции rainbow"
    ks = [int(k) for k in re.findall(r"1\.5-Math\.abs\(4\*t-(\d)\)", m.group(1))]
    assert ks == [3, 2, 1]
    assert "const ramp=rainbow;" in src                     # в интерфейсе поле красится только ею


def test_matplotlib_colormap_is_built_from_the_same_formula():
    pytest.importorskip("matplotlib")
    from magcore.post.palette import field_rainbow_cmap

    cmap = field_rainbow_cmap(256)
    t = np.arange(256) / 255.0                              # ровно в образцах палитры (между ними — ступенька)
    assert np.allclose(np.asarray(cmap(t))[:, :3], field_rainbow(t), atol=1e-12)
