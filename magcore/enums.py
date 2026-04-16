from __future__ import annotations

from enum import Enum


class MaterialKind(str, Enum):
    AIR = "air"
    LINEAR = "linear"
    PERMANENT_MAGNET = "permanent_magnet"


class FormulationKind(str, Enum):
    MULTITRACE = "multitrace"
    SCHUR_PHI_REDUCED = "schur_phi_reduced"


class BasisKind(str, Enum):
    P0 = "P0"
    P1 = "P1"


class LinearSolverKind(str, Enum):
    DIRECT = "direct"
    GMRES = "gmres"


class ValidationLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"