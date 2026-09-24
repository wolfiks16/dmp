from __future__ import annotations

import threading
from contextlib import contextmanager

# ОТМЕНА ДОЛГОГО РАСЧЁТА — «по-хорошему». Поток Python нельзя убить на полуслове, поэтому решатели сами
# проверяют флаг в своих циклах (итерации Ньютона и Пикара, шаги по времени, положения ротора) и выходят
# исключением `Cancelled`. Флаг ставит тот, кто запустил расчёт (веб-сервер: кнопка «Отменить»), через
# `cancel_scope(event)` — он действует только в своём потоке. Вне такой области `check()` ничего не
# делает: в тестах и скриптах решатели работают как прежде, на числа проверка не влияет.


class Cancelled(Exception):
    """Расчёт отменён пользователем."""


_local = threading.local()


@contextmanager
def cancel_scope(event: threading.Event):
    """Внутри области `check()` в ЭТОМ потоке бросает `Cancelled`, как только `event` установлен."""
    prev = getattr(_local, "event", None)
    _local.event = event
    try:
        yield event
    finally:
        _local.event = prev


def check() -> None:
    """Точка отмены: если расчёт в этом потоке отменён — бросить `Cancelled`, иначе ничего не делать."""
    ev = getattr(_local, "event", None)
    if ev is not None and ev.is_set():
        raise Cancelled("расчёт отменён")
