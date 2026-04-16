from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from magcore.domain.config import SolverConfig
from magcore.domain.problem import MagnetostaticProblem
from magcore.bem.spaces import FaceP0Space, VertexP1Space, DiscreteSpace
from magcore.bem.operators import (
    OperatorKind,
    BoundaryOperator,
    SingleLayerOperator,
    DoubleLayerOperator,
    AdjointDoubleLayerOperator,
    HypersingularOperator,
    AssembledBoundaryOperator,
)
from magcore.bem.regular_single_layer import assemble_single_layer_p0p0_regular
from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig, assemble_single_layer_p0p0_full

@dataclass(frozen=True, slots=True)
class AssemblyContext:
    problem: MagnetostaticProblem
    config: SolverConfig
    phi_space: VertexP1Space
    flux_space: FaceP0Space


@dataclass(slots=True)
class AssembledBlock:
    name: str
    matrix: np.ndarray
    operator_kind: str
    shape: tuple[int, int]


@dataclass(slots=True)
class AssembledSystem:
    blocks: dict[str, AssembledBlock]
    matrix: np.ndarray
    rhs: np.ndarray
    spaces: dict[str, DiscreteSpace]
    metadata: dict = field(default_factory=dict)


def build_trace_spaces(problem: MagnetostaticProblem, config: SolverConfig) -> tuple[VertexP1Space, FaceP0Space]:
    face_indices = tuple(range(problem.surface_mesh.n_faces))
    phi_space = VertexP1Space.from_faces(problem.surface_mesh, face_indices, name="phi_trace_space")
    flux_space = FaceP0Space.from_faces(problem.surface_mesh, face_indices, name="flux_trace_space")
    return phi_space, flux_space


def build_assembly_context(problem: MagnetostaticProblem, config: SolverConfig) -> AssemblyContext:
    phi_space, flux_space = build_trace_spaces(problem, config)
    return AssemblyContext(
        problem=problem,
        config=config,
        phi_space=phi_space,
        flux_space=flux_space,
    )


def build_placeholder_operators(ctx: AssemblyContext) -> dict[str, BoundaryOperator]:
    phi = ctx.phi_space
    flux = ctx.flux_space

    ops: dict[str, BoundaryOperator] = {
        "V": SingleLayerOperator(
            kind=OperatorKind.SINGLE_LAYER,
            domain_space=flux,
            range_space=phi,
            dual_space=None,
            label="V",
        ),
        "K": DoubleLayerOperator(
            kind=OperatorKind.DOUBLE_LAYER,
            domain_space=phi,
            range_space=phi,
            dual_space=None,
            label="K",
        ),
        "Kt": AdjointDoubleLayerOperator(
            kind=OperatorKind.ADJOINT_DOUBLE_LAYER,
            domain_space=flux,
            range_space=flux,
            dual_space=None,
            label="K'",
        ),
        "D": HypersingularOperator(
            kind=OperatorKind.HYPERSINGULAR,
            domain_space=phi,
            range_space=flux,
            dual_space=None,
            label="D",
        ),
    }
    return ops


def assemble_operator_placeholder(op: BoundaryOperator) -> np.ndarray:
    m, n = op.shape
    return np.zeros((m, n), dtype=float)


def assemble_multitrace_system_placeholder(problem: MagnetostaticProblem, config: SolverConfig) -> AssembledSystem:
    ctx = build_assembly_context(problem, config)
    ops = build_placeholder_operators(ctx)

    V = assemble_operator_placeholder(ops["V"])
    K = assemble_operator_placeholder(ops["K"])
    Kt = assemble_operator_placeholder(ops["Kt"])
    D = assemble_operator_placeholder(ops["D"])

    n_phi = ctx.phi_space.ndofs
    n_flux = ctx.flux_space.ndofs

    # Placeholder block layout:
    # [ I_phi   V      ]
    # [ D       I_flux ]
    I_phi = np.eye(n_phi, dtype=float)
    I_flux = np.eye(n_flux, dtype=float)

    top = np.hstack([I_phi, V])
    bottom = np.hstack([D, I_flux])
    A = np.vstack([top, bottom])

    rhs = np.zeros(n_phi + n_flux, dtype=float)

    blocks = {
        "I_phi": AssembledBlock("I_phi", I_phi, "identity_phi", I_phi.shape),
        "V": AssembledBlock("V", V, ops["V"].kind, V.shape),
        "D": AssembledBlock("D", D, ops["D"].kind, D.shape),
        "I_flux": AssembledBlock("I_flux", I_flux, "identity_flux", I_flux.shape),
        "K": AssembledBlock("K", K, ops["K"].kind, K.shape),
        "Kt": AssembledBlock("Kt", Kt, ops["Kt"].kind, Kt.shape),
    }

    return AssembledSystem(
        blocks=blocks,
        matrix=A,
        rhs=rhs,
        spaces={
            "phi": ctx.phi_space,
            "flux": ctx.flux_space,
        },
        metadata={
            "formulation": config.formulation.value if hasattr(config.formulation, "value") else str(config.formulation),
            "n_phi": n_phi,
            "n_flux": n_flux,
            "placeholder": True,
        },
    )


def assemble_single_layer_p0p0_operator_regular(
    problem: MagnetostaticProblem,
    face_indices: tuple[int, ...] | None = None,
    quadrature_order_target: int = 2,
    quadrature_order_source: int = 2,
    near_factor: float = 2.0,
    strict: bool = True,
) -> AssembledBoundaryOperator:
    """
    Assemble a real dense P0-P0 regular-only single-layer operator.

    This is an auxiliary verified backend for R1.4, not yet the final trace operator
    of the target P1/P0 SurfaceBEMSolver.
    """
    mesh = problem.surface_mesh
    if face_indices is None:
        face_indices = tuple(range(mesh.n_faces))

    space = FaceP0Space.from_faces(mesh, face_indices, name="p0_face_space")

    matrix, mask = assemble_single_layer_p0p0_regular(
        mesh=mesh,
        face_indices=face_indices,
        quadrature_order_target=quadrature_order_target,
        quadrature_order_source=quadrature_order_source,
        near_factor=near_factor,
        strict=strict,
    )

    return AssembledBoundaryOperator(
        kind=OperatorKind.SINGLE_LAYER,
        matrix=matrix,
        domain_space=space,
        range_space=space,
        label="V_reg_p0p0",
        metadata={
            "backend": "regular_p0p0_double_quadrature",
            "near_factor": near_factor,
            "strict": strict,
            "regular_pairs": len(mask.regular_pairs),
            "near_pairs": len(mask.near_pairs),
            "singular_pairs": len(mask.singular_pairs),
        },
    )

def assemble_single_layer_p0p0_operator_full(
    problem: MagnetostaticProblem,
    face_indices: tuple[int, ...] | None = None,
    config: AdaptiveIntegrationConfig | None = None,
) -> AssembledBoundaryOperator:
    """
    Assemble a complete dense P0-P0 single-layer operator including
    self / shared-edge / shared-vertex / near / regular interactions.
    """
    mesh = problem.surface_mesh
    if face_indices is None:
        face_indices = tuple(range(mesh.n_faces))
    if config is None:
        config = AdaptiveIntegrationConfig()

    space = FaceP0Space.from_faces(mesh, face_indices, name="p0_face_space")
    matrix, metadata = assemble_single_layer_p0p0_full(
        mesh=mesh,
        face_indices=face_indices,
        config=config,
    )

    return AssembledBoundaryOperator(
        kind=OperatorKind.SINGLE_LAYER,
        matrix=matrix,
        domain_space=space,
        range_space=space,
        label="V_full_p0p0",
        metadata=metadata,
    )