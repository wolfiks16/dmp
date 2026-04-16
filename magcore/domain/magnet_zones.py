from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class MagnetZoneSpec:
    zone_id: str
    region_id: str
    curve_id: str
    easy_axis: tuple[float, float, float]

    def __post_init__(self) -> None:
        a = np.asarray(self.easy_axis, dtype=float)
        if a.shape != (3,):
            raise ValueError("easy_axis must be a 3D vector.")
        if not np.isfinite(a).all():
            raise ValueError("easy_axis must contain finite values.")

        nrm = float(np.linalg.norm(a))
        if nrm == 0.0:
            raise ValueError("easy_axis must be nonzero.")

        a_unit = tuple((a / nrm).tolist())
        object.__setattr__(self, "easy_axis", a_unit)

    @property
    def easy_axis_array(self) -> np.ndarray:
        return np.asarray(self.easy_axis, dtype=float)


@dataclass(frozen=True, slots=True)
class MagnetAssemblySpec:
    zones: tuple[MagnetZoneSpec, ...]

    def __post_init__(self) -> None:
        zone_ids = [z.zone_id for z in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("Duplicate zone_id in MagnetAssemblySpec.")

        region_ids = [z.region_id for z in self.zones]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("Each region_id may belong to at most one magnet zone.")

    def zone_map(self) -> dict[str, MagnetZoneSpec]:
        return {z.zone_id: z for z in self.zones}

    def zone_by_region_id(self, region_id: str) -> MagnetZoneSpec | None:
        for z in self.zones:
            if z.region_id == region_id:
                return z
        return None

    def zone_ids(self) -> tuple[str, ...]:
        return tuple(z.zone_id for z in self.zones)

    def region_ids(self) -> tuple[str, ...]:
        return tuple(z.region_id for z in self.zones)