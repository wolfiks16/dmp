from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.adjoint_trace_collocation import (
    OffsetTraceConfig,
    assemble_single_layer_normal_trace_matrices,
)
from magcore.bem.assembly import assemble_single_layer_p0p0_operator_full
from magcore.bem.background_fields import (
    evaluate_background_on_points,
    linear_background_normal_flux,
    linear_background_potential,
)
from magcore.bem.closed_surface_checks import check_closed_face_set
from magcore.bem.evaluation import evaluate_single_layer_potential_p0


@dataclass(frozen=True, slots=True)
class TransmissionContrastConfig:
    adaptive_config: AdaptiveIntegrationConfig
    offset_config: OffsetTraceConfig
    mu_in: float
    mu_out: float
    H0: np.ndarray


@dataclass(slots=True)
class TransmissionSolveResult:
    lambda_in: np.ndarray
    lambda_out: np.ndarray
    system_matrix: np.ndarray
    rhs: np.ndarray
    residual_norm: float
    condition_number_est: float
    face_centroids: np.ndarray
    face_normals: np.ndarray
    g_bg: np.ndarray
    q_bg: np.ndarray
    metadata: dict


def build_transmission_rhs(
    centroids: np.ndarray,
    normals: np.ndarray,
    H0: np.ndarray,
    mu_in: float,
    mu_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build background data and transmission RHS.

    With background included on both sides:
        phi^- = phi_bg + S lambda^-
        phi^+ = phi_bg + S lambda^+

    continuity gives:
        V(lambda^- - lambda^+) = 0

    flux continuity gives:
        mu_in (1/2 I + K') lambda^-
      - mu_out(-1/2 I + K') lambda^+
      = (mu_out - mu_in) q_bg
    """
    phi_bg = linear_background_potential(H0)
    g_bg = evaluate_background_on_points(centroids, phi_bg)
    q_bg = linear_background_normal_flux(H0, normals)

    n = centroids.shape[0]
    rhs_top = np.zeros(n, dtype=float)
    rhs_bottom = (mu_out - mu_in) * q_bg
    rhs = np.concatenate([rhs_top, rhs_bottom])

    return g_bg, q_bg, rhs


def assemble_transmission_system(
    problem,
    face_indices: tuple[int, ...],
    cfg: TransmissionContrastConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Assemble the first physical transmission block system for mu-contrast
    without remanence.

    Unknowns:
        lambda_in, lambda_out

    Block system:
        [ V                              -V                           ] [lambda_in ] = [0]
        [ mu_in( 1/2 I + Kt_avg )   -mu_out(-1/2 I + Kt_avg) ] [lambda_out]   [(mu_out-mu_in) q_bg]
    """
    if cfg.mu_in <= 0.0 or cfg.mu_out <= 0.0:
        raise ValueError("mu_in and mu_out must be positive.")

    face_indices = tuple(sorted(face_indices))
    closed_report = check_closed_face_set(problem.surface_mesh, face_indices)
    closed_report.raise_if_not_closed()

    Vop = assemble_single_layer_p0p0_operator_full(
        problem=problem,
        face_indices=face_indices,
        config=cfg.adaptive_config,
    )
    trace = assemble_single_layer_normal_trace_matrices(
        mesh=problem.surface_mesh,
        face_indices=face_indices,
        offset_config=cfg.offset_config,
    )

    V = np.asarray(Vop.matrix, dtype=float)
    Kt = np.asarray(trace.Kt_avg, dtype=float)

    n = len(face_indices)
    I = np.eye(n, dtype=float)

    A11 = V
    A12 = -V
    A21 = cfg.mu_in * (0.5 * I + Kt)
    A22 = -cfg.mu_out * (-0.5 * I + Kt)

    A = np.block(
        [
            [A11, A12],
            [A21, A22],
        ]
    )

    g_bg, q_bg, rhs = build_transmission_rhs(
        centroids=trace.centroids,
        normals=trace.normals,
        H0=cfg.H0,
        mu_in=cfg.mu_in,
        mu_out=cfg.mu_out,
    )

    metadata = {
        "backend": "offset_trace_transmission_collocation",
        "mu_in": cfg.mu_in,
        "mu_out": cfg.mu_out,
        "n_faces": n,
        "V_backend": Vop.metadata.get("backend"),
        "trace_backend": trace.metadata.get("backend"),
    }

    aux = {
        "trace": trace,
        "g_bg": g_bg,
        "q_bg": q_bg,
        "metadata": metadata,
    }
    return A, rhs, aux


def solve_linear_transmission_problem(
    problem,
    face_indices: tuple[int, ...],
    cfg: TransmissionContrastConfig,
) -> TransmissionSolveResult:
    face_indices = tuple(sorted(face_indices))

    A, rhs, aux = assemble_transmission_system(
        problem=problem,
        face_indices=face_indices,
        cfg=cfg,
    )

    x = np.linalg.solve(A, rhs)
    residual = A @ x - rhs

    n = len(face_indices)
    lambda_in = x[:n]
    lambda_out = x[n:]

    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = float("inf")

    trace = aux["trace"]

    return TransmissionSolveResult(
        lambda_in=lambda_in,
        lambda_out=lambda_out,
        system_matrix=A,
        rhs=rhs,
        residual_norm=float(np.linalg.norm(residual)),
        condition_number_est=cond,
        face_centroids=trace.centroids,
        face_normals=trace.normals,
        g_bg=aux["g_bg"],
        q_bg=aux["q_bg"],
        metadata=aux["metadata"],
    )


def reconstruct_transmission_potential(
    problem,
    face_indices: tuple[int, ...],
    result: TransmissionSolveResult,
    target_points: np.ndarray,
    H0: np.ndarray,
    side: str = "exterior",
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Reconstruct total potential on either side:

        phi^- = phi_bg + S lambda_in
        phi^+ = phi_bg + S lambda_out

    side:
        "interior" or "exterior"
    """
    target_points = np.asarray(target_points, dtype=float)
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3).")

    side = side.lower()
    if side not in {"interior", "exterior"}:
        raise ValueError("side must be 'interior' or 'exterior'.")

    phi_bg_fn = linear_background_potential(H0)
    phi_bg = evaluate_background_on_points(target_points, phi_bg_fn)

    if side == "interior":
        sigma = result.lambda_in
    else:
        sigma = result.lambda_out

    scat = evaluate_single_layer_potential_p0(
        mesh=problem.surface_mesh,
        face_indices=tuple(sorted(face_indices)),
        density=sigma,
        target_points=target_points,
        quadrature_order=quadrature_order,
    )

    return phi_bg + scat