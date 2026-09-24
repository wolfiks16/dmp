from __future__ import annotations

import threading

# СЕАНС gmsh — ОДИН НА ВЕСЬ ПРОЦЕСС. У gmsh глобальное состояние (модель, опции, сетка): два потока,
# работающие с ним одновременно, портят друг другу построение, а finalize одного обрывает сеанс другого.
# В веб-сервере сетки строятся и по запросам интерфейса, и внутри фоновых расчётов (прогонка ротора
# строит сетку на каждом положении), поэтому ЛЮБОЕ построение идёт под этим замком: сетки строятся по
# очереди, а сервер в это время отвечает на остальные запросы.
#
# interruptible=False — gmsh не ставит свой обработчик Ctrl+C: иначе инициализация падает вне главного
# потока («signal only works in main thread»). На главном потоке поведение то же.
#
# Использование (пара строго в try/finally):
#     gmsh = open_gmsh()
#     try:
#         ...
#     finally:
#         close_gmsh()

GMSH_LOCK = threading.RLock()


def open_gmsh():
    """Занять замок и открыть сеанс gmsh; вернуть модуль gmsh. Вложенный сеанс — явная ошибка."""
    import gmsh

    GMSH_LOCK.acquire()
    try:
        if gmsh.isInitialized():
            # RLock пускает тот же поток повторно: вложенный initialize/finalize оборвал бы внешний сеанс
            raise RuntimeError("сеанс gmsh уже открыт в этом потоке: вложенные построения не поддерживаются.")
        gmsh.initialize(interruptible=False)
    except BaseException:
        GMSH_LOCK.release()                  # сеанс не открылся — замок не должен остаться занятым
        raise
    return gmsh


def close_gmsh() -> None:
    """Закрыть сеанс gmsh и освободить замок (даже если закрытие упало)."""
    import gmsh

    try:
        gmsh.finalize()
    finally:
        GMSH_LOCK.release()
