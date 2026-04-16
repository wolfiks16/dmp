from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.domain.magnet_zones import MagnetAssemblySpec
from magcore.domain.problem import MagnetostaticProblem
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class FaceRegionMap:
    face_indices: tuple[int, ...]
    region_minus_ids: tuple[str, ...]
    region_plus_ids: tuple[str, ...]

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)


@dataclass(frozen=True, slots=True)
class FaceZoneMap:
    face_indices: tuple[int, ...]
    region_minus_ids: tuple[str, ...]
    region_plus_ids: tuple[str, ...]
    zone_minus_ids: tuple[str | None, ...]
    zone_plus_ids: tuple[str | None, ...]
    normals: np.ndarray

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)


def _patch_lookup(problem: MagnetostaticProblem) -> dict[int, tuple[str, str]]:
    """
    Build a face -> (region_minus_id, region_plus_id) lookup from topology patches.
    """
    lookup: dict[int, tuple[str, str]] = {}

    for patch in problem.topology.interface_patches:
        for f_idx in patch.face_indices:
            if f_idx in lookup:
                raise ValueError(f"Face {f_idx} appears in more than one interface patch.")
            lookup[f_idx] = (patch.region_minus_id, patch.region_plus_id)

    return lookup


def build_face_region_maps(
    problem: MagnetostaticProblem,
    face_indices: tuple[int, ...],
) -> FaceRegionMap:
    """
    Map selected interface faces to minus/plus region ids based on topology patches.
    """
    face_indices = tuple(sorted(face_indices))
    lookup = _patch_lookup(problem)

    minus_ids = []
    plus_ids = []

    for f_idx in face_indices:
        if f_idx not in lookup:
            raise KeyError(f"Face {f_idx} is not covered by topology interface patches.")
        r_minus, r_plus = lookup[f_idx]
        minus_ids.append(r_minus)
        plus_ids.append(r_plus)

    return FaceRegionMap(
        face_indices=face_indices,
        region_minus_ids=tuple(minus_ids),
        region_plus_ids=tuple(plus_ids),
    )


def build_face_zone_maps(
    problem: MagnetostaticProblem,
    magnet_assembly: MagnetAssemblySpec,
    face_indices: tuple[int, ...],
) -> FaceZoneMap:
    """
    Extend region-side face mapping with magnet-zone ids on each side.

    If a side belongs to a non-magnetic region, the corresponding zone id is None.
    """
    face_indices = tuple(sorted(face_indices))
    fr = build_face_region_maps(problem, face_indices)
    normals = np.array([problem.surface_mesh.face_normal(f) for f in face_indices], dtype=float)

    zone_minus = []
    zone_plus = []

    for r_minus, r_plus in zip(fr.region_minus_ids, fr.region_plus_ids, strict=False):
        z_minus = magnet_assembly.zone_by_region_id(r_minus)
        z_plus = magnet_assembly.zone_by_region_id(r_plus)

        zone_minus.append(None if z_minus is None else z_minus.zone_id)
        zone_plus.append(None if z_plus is None else z_plus.zone_id)

    return FaceZoneMap(
        face_indices=face_indices,
        region_minus_ids=fr.region_minus_ids,
        region_plus_ids=fr.region_plus_ids,
        zone_minus_ids=tuple(zone_minus),
        zone_plus_ids=tuple(zone_plus),
        normals=normals,
    )


def zone_face_groups(
    problem: MagnetostaticProblem,
    magnet_assembly: MagnetAssemblySpec,
    face_indices: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    """
    Build zone -> face_indices map.

    A face is associated with a magnetic zone if that zone appears on either side
    of the interface face.
    """
    fzm = build_face_zone_maps(problem, magnet_assembly, face_indices)

    groups: dict[str, list[int]] = {}
    for f_idx, z_minus, z_plus in zip(
        fzm.face_indices,
        fzm.zone_minus_ids,
        fzm.zone_plus_ids,
        strict=False,
    ):
        if z_minus is not None:
            groups.setdefault(z_minus, []).append(f_idx)
        if z_plus is not None and z_plus != z_minus:
            groups.setdefault(z_plus, []).append(f_idx)

    return {k: tuple(v) for k, v in groups.items()}


def build_mu_side_vectors(
    face_zone_map: FaceZoneMap,
    zone_state_map: dict,
    mu_air: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build side-dependent effective permeability vectors for the selected faces.

    Returns
    -------
    mu_minus, mu_plus:
        arrays of shape (N_faces,)
    """
    n = face_zone_map.n_faces
    mu_minus = np.full(n, mu_air, dtype=float)
    mu_plus = np.full(n, mu_air, dtype=float)

    for i, (z_minus, z_plus) in enumerate(
        zip(face_zone_map.zone_minus_ids, face_zone_map.zone_plus_ids, strict=False)
    ):
        if z_minus is not None:
            mu_minus[i] = float(zone_state_map[z_minus].mu_eff)
        if z_plus is not None:
            mu_plus[i] = float(zone_state_map[z_plus].mu_eff)

    return mu_minus, mu_plus


def build_source_jump_vector(
    face_zone_map: FaceZoneMap,
    zone_state_map: dict,
) -> np.ndarray:
    """
    Build the magnetic source jump on each face:

        s_i = (B_src^- - B_src^+) · n_i

    where n_i is the face normal oriented from minus to plus side.
    """
    n_faces = face_zone_map.n_faces
    out = np.zeros(n_faces, dtype=float)

    for i, (z_minus, z_plus, n_i) in enumerate(
        zip(
            face_zone_map.zone_minus_ids,
            face_zone_map.zone_plus_ids,
            face_zone_map.normals,
            strict=False,
        )
    ):
        Bm = np.zeros(3, dtype=float)
        Bp = np.zeros(3, dtype=float)

        if z_minus is not None:
            Bm = np.asarray(zone_state_map[z_minus].B_src_vector, dtype=float)
        if z_plus is not None:
            Bp = np.asarray(zone_state_map[z_plus].B_src_vector, dtype=float)

        out[i] = float(np.dot(Bm - Bp, n_i))

    return out