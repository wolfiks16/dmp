from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.assembly import assemble_single_layer_p0p0_operator_full
from magcore.bem.closed_surface_checks import check_closed_face_set
from magcore.bem.evaluation import evaluate_single_layer_potential_p0
from magcore.bem.reference_fields import evaluate_reference_on_points
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(slots=True)
class SingleLayerDirichletSolveResult:
    sigma: np.ndarray
    boundary_data: np.ndarray
    face_centroids: np.ndarray
    matrix: np.ndarray
    residual_norm: float
    condition_number_est: float
    metadata: dict


def face_centroids(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> np.ndarray:
    face_indices = tuple(sorted(face_indices))
    return np.array([mesh.face_centroid(f) for f in face_indices], dtype=float)


def build_boundary_data_from_reference(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    ref_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build collocation-style Dirichlet data g from a reference harmonic field
    evaluated at face centroids.
    """
    ctrs = face_centroids(mesh, face_indices)
    g = evaluate_reference_on_points(ctrs, ref_fn)
    return ctrs, g


def solve_single_layer_dirichlet_p0(
    problem,
    face_indices: tuple[int, ...],
    ref_fn,
    adaptive_config: AdaptiveIntegrationConfig | None = None,
) -> SingleLayerDirichletSolveResult:
    """
    Solve:
        V sigma = g

    where:
    - V is the full dense P0/P0 single-layer operator
    - g is the reference Dirichlet data sampled at face centroids

    This is a collocation-style verification solve on a closed surface.
    """
    if adaptive_config is None:
        adaptive_config = AdaptiveIntegrationConfig()

    face_indices = tuple(sorted(face_indices))
    closed_report = check_closed_face_set(problem.surface_mesh, face_indices)
    closed_report.raise_if_not_closed()

    op = assemble_single_layer_p0p0_operator_full(
        problem=problem,
        face_indices=face_indices,
        config=adaptive_config,
    )

    ctrs, g = build_boundary_data_from_reference(
        problem.surface_mesh,
        face_indices,
        ref_fn,
    )

    A = np.asarray(op.matrix, dtype=float)
    sigma = np.linalg.solve(A, g)
    residual = A @ sigma - g

    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = float("inf")

    return SingleLayerDirichletSolveResult(
        sigma=sigma,
        boundary_data=g,
        face_centroids=ctrs,
        matrix=A,
        residual_norm=float(np.linalg.norm(residual)),
        condition_number_est=cond,
        metadata={
            "operator_label": op.label,
            "operator_backend": op.metadata.get("backend"),
            "closed_surface_components": closed_report.n_components,
            "n_faces": len(face_indices),
        },
    )


def reconstruct_potential_from_density(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    sigma: np.ndarray,
    target_points: np.ndarray,
    quadrature_order: int = 2,
) -> np.ndarray:
    return evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=sigma,
        target_points=target_points,
        quadrature_order=quadrature_order,
    )


@dataclass(frozen=True, slots=True)
class ReconstructionErrorReport:
    l2: float
    linf: float
    abs_errors: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray


def compute_reference_error(
    predicted: np.ndarray,
    target_points: np.ndarray,
    ref_fn,
) -> ReconstructionErrorReport:
    predicted = np.asarray(predicted, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    reference = evaluate_reference_on_points(target_points, ref_fn)

    abs_err = np.abs(predicted - reference)
    l2 = float(np.sqrt(np.mean(abs_err ** 2)))
    linf = float(np.max(abs_err))

    return ReconstructionErrorReport(
        l2=l2,
        linf=linf,
        abs_errors=abs_err,
        prediction=predicted,
        reference=reference,
    )