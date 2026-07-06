from __future__ import annotations

import numpy as np

from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupling_block
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def test_coupling_block_shape_and_nontrivial() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    B = assemble_coupling_block(ci, vs)

    assert B.shape == (vs.ndofs, ci.n_phi_dofs)
    assert np.isfinite(B).all()
    assert np.linalg.norm(B) > 0.0


def test_coupling_block_constant_psi_is_zero_divergence_identity() -> None:
    # For every edge: ⟨1, curl w·n⟩_Γ = ∮_Γ curl w·n dS = ∫_Ω div(curl w) dV = 0.
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    B = assemble_coupling_block(ci, vs)

    row_sums = B @ np.ones(ci.n_phi_dofs)
    assert np.linalg.norm(row_sums) < 1e-10


def test_coupling_block_matches_independent_curl_reconstruction() -> None:
    # (B^T a)_m = ∮_Γ λ_m (curl A_h · n) dS, recomputed via the post-processing curl path.
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    B = assemble_coupling_block(ci, vs)

    rng = np.random.default_rng(3)
    a = rng.standard_normal(vs.ndofs)
    got = B.T @ a

    expected = np.zeros(ci.n_phi_dofs, dtype=float)
    vertex_to_dof = ci.phi_space.vertex_to_dof
    for f in range(ci.n_faces):
        c = int(ci.face_to_cell[f])
        flux = float(np.dot(evaluate_curl_on_cell(vs, a, c), ci.outward_normals[f]))
        third_area = float(ci.face_areas[f]) / 3.0
        for v in ci.surface_mesh.faces[f]:
            expected[int(vertex_to_dof[int(v)])] += flux * third_area

    assert np.allclose(got, expected, atol=1e-12)


def test_coupling_block_is_structurally_local() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(3)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    B = assemble_coupling_block(ci, vs)

    boundary_owner_cells = {int(c) for c in ci.face_to_cell}
    edges_touching: set[int] = set()
    for c in boundary_owner_cells:
        edges_touching.update(int(d) for d in vs.cell_dof_indices(c))

    # Genuinely local: interior cells exist, so not every edge couples to Γ.
    assert len(edges_touching) < vs.ndofs
    # Every nonzero row is an edge of a boundary-owning cell (no spurious coupling).
    nonzero_rows = {r for r in range(vs.ndofs) if not np.allclose(B[r], 0.0)}
    assert nonzero_rows.issubset(edges_touching)
