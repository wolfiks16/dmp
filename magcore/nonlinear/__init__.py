"""
Размерно-независимое ядро нелинейных решателей (общее для 2D/3D backends).

Здесь живёт математическая структура, не зависящая от геометрии дискретизации:
хордовый Picard фиксированной точки (релаксация ν + критерий сходимости по ‖ΔB‖)
и приведение источника намагниченности к функции состояния. Конкретная сборка,
решатель и извлечение B=curl A инкапсулируются backend'ом через callback `step`.

См. docs/novelty/pivot_2D_thermal_SmCo_2026-07-07.md §1.6 (общее ядро + backends).
"""
from magcore.nonlinear.picard import (
    PicardLoopResult,
    resolve_magnetization,
    run_picard_fixed_point,
)

__all__ = [
    "PicardLoopResult",
    "resolve_magnetization",
    "run_picard_fixed_point",
]
